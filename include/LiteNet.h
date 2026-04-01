#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>

namespace MicroNAM {

// ─────────────────────────────────────────────────────────────────────────────
// Lite WaveNet inference engine
//
// Lite architecture (from neural-amp-modeler/nam/train/core.py):
//   LA0: 7 layers × 12-channel, kernel_size=3, dilations=[1,2,4,8,16,32,64]
//        head_size=6  (12-ch activations projected 12→6 to seed LA1 head)
//   LA1: 13 layers × 6-channel, kernel_size=3,
//        dilations=[128,256,512,1,2,4,8,16,32,64,128,256,512]
//        head_size=1  (6-ch head dot-producted → scalar output)
//   head_scale=0.02
//
// load_weights expects exactly 6554 floats in the NAM file weight order.
// forward processes exactly NAM_BLOCK samples per call.
// ─────────────────────────────────────────────────────────────────────────────

template<size_t NAM_BLOCK>
class LiteNet
{
public:
  // ── API ───────────────────────────────────────────────────────────────────
  void reset() { reset_impl(); }
  void load_weights(const float* model_weights) { load_weights_impl(model_weights); }
  void forward(const float* input, float* output) noexcept { forward_impl(input, output); }
  //   ^ Process exactly NAM_BLOCK samples.
  // ──────────────────────────────────────────────────────────────────────────

private:
  // ─────────────────────────────────────────────────────────────────────────────
  // Padé(3,2) tanh — same approximation as NanoNAM / FeatherNAM.
  // ─────────────────────────────────────────────────────────────────────────────
  static inline float nam_tanh(float x) noexcept
  {
    const float x2  = x * x;
    const float num = x * (27.0f + x2);
    const float den = 27.0f + 9.0f * x2;
    return num / den;
  }

  // ── 6-channel layer (used for LA1) ───────────────────────────────────────
  struct Layer6
  {
    float w0T[6][6];    // tap0 (current input),  [in][out]
    float w1T[6][6];    // tap1 = state[ptr-D],   [in][out]
    float w2T[6][6];    // tap2 = state[ptr-2D],  [in][out]
    float conv_b[6];
    float mixin_w[6];
    float w1x1T[6][6];  // 1×1 projection, [in][out]
    float b1x1[6];

    float* state;
    int    state_size;
    int    state_ptr;
    int    neg_dilation;
    int    neg_2dilation;
  };

  // ── 12-channel layer (used for LA0) ──────────────────────────────────────
  struct Layer12
  {
    float w0T[12][12];    // tap0 (current input),  [in][out]
    float w1T[12][12];    // tap1 = state[ptr-D],   [in][out]
    float w2T[12][12];    // tap2 = state[ptr-2D],  [in][out]
    float conv_b[12];
    float mixin_w[12];
    float w1x1T[12][12];  // 1×1 projection, [in][out]
    float b1x1[12];

    float* state;
    int    state_size;
    int    state_ptr;
    int    neg_dilation;
    int    neg_2dilation;
  };

  // ── Layer Array 0: 7 × 12-channel ────────────────────────────────────────
  float   la0_rechannel_w[12];    // 1→12 input projection
  float   la0_head_wT[12][6];     // 12→6 head projection, [in][out]
  Layer12 la0[7];
  // State: sum((2d+1)*12) for d in {1,2,4,8,16,32,64}
  // = 36+60+108+204+396+780+1548 = 3132
  float   la0_state[3132];

  // ── Layer Array 1: 13 × 6-channel ────────────────────────────────────────
  float  la1_rechannel_wT[12][6]; // 12→6 rechannel, [in][out]
  float  la1_head_w[6];
  float  la1_head_b;
  Layer6 la1[13];
  // State: sum((2d+1)*6) for d in {128,256,512,1,2,4,8,16,32,64,128,256,512}
  // = 1542+3078+6150+18+30+54+102+198+390+774+1542+3078+6150 = 23106
  float  la1_state[23106];

  float  head_scale;

  // ── Block working buffers ─────────────────────────────────────────────────
  float la0_buf_a[NAM_BLOCK][12];
  float la0_buf_b[NAM_BLOCK][12];
  float la0_head [NAM_BLOCK][12];  // LA0 skip-connection accumulator (12-ch)

  float la1_buf_a[NAM_BLOCK][6];
  float la1_buf_b[NAM_BLOCK][6];
  float la1_head [NAM_BLOCK][6];

  void init_state_ptrs()
  {
    static constexpr int la0_d[7]  = { 1, 2, 4, 8, 16, 32, 64 };
    static constexpr int la1_d[13] = { 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

    int off = 0;
    for (int i = 0; i < 7; ++i)
    {
      const int d  = la0_d[i];
      const int ss = (3-1)*d + 1;
      la0[i].state_size    = ss;
      la0[i].state_ptr     = 0;
      la0[i].neg_dilation  = ss - d;
      la0[i].neg_2dilation = ss - 2*d;
      la0[i].state         = la0_state + off;
      off += 12 * ss;
    }

    off = 0;
    for (int i = 0; i < 13; ++i)
    {
      const int d  = la1_d[i];
      const int ss = (3-1)*d + 1;
      la1[i].state_size    = ss;
      la1[i].state_ptr     = 0;
      la1[i].neg_dilation  = ss - d;
      la1[i].neg_2dilation = ss - 2*d;
      la1[i].state         = la1_state + off;
      off += 6 * ss;
    }
  }

  void reset_impl()
  {
    for (int i = 0; i < 3132;  ++i) la0_state[i] = 0.f;
    for (int i = 0; i < 23106; ++i) la1_state[i] = 0.f;
    for (int i = 0; i < 7;  ++i) la0[i].state_ptr = 0;
    for (int i = 0; i < 13; ++i) la1[i].state_ptr = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // load_weights — float[6554]
  //
  // File weight order: conv[out][in][k], k=0 oldest, k=2 newest.
  // Split at load time:
  //   file k=2 → w0T[in][out]  (tap0 = current input)
  //   file k=1 → w1T[in][out]
  //   file k=0 → w2T[in][out]
  //
  // Weight tally:
  //   la0_rechannel_w:  12
  //   LA0 × 7 layers:   7 × (12*12*3 + 12 + 12 + 12*12 + 12) = 7 × 612 = 4284
  //   la0_head_wT:      12×6 = 72
  //   la1_rechannel_wT: 12×6 = 72
  //   LA1 × 13 layers:  13 × (6*6*3 + 6 + 6 + 6*6 + 6) = 13 × 162 = 2106
  //   la1_head_w:       6
  //   la1_head_b:       1
  //   head_scale:       1
  //   Total:            6554
  // ─────────────────────────────────────────────────────────────────────────────
  void load_weights_impl(const float* model_weights)
  {
    init_state_ptrs();
    reset_impl();

    const float* w = model_weights;

    // ── LA0 ──────────────────────────────────────────────────────────────────
    for (int o = 0; o < 12; ++o) la0_rechannel_w[o] = *w++;

    for (int l = 0; l < 7; ++l)
    {
      for (int o = 0; o < 12; ++o)
        for (int i = 0; i < 12; ++i)
          for (int k = 0; k < 3; ++k)
          {
            const float val = *w++;
            if      (k == 2) la0[l].w0T[i][o] = val;
            else if (k == 1) la0[l].w1T[i][o] = val;
            else             la0[l].w2T[i][o] = val;
          }

      for (int o = 0; o < 12; ++o) la0[l].conv_b[o]  = *w++;
      for (int o = 0; o < 12; ++o) la0[l].mixin_w[o] = *w++;
      for (int o = 0; o < 12; ++o)
        for (int i = 0; i < 12; ++i)
          la0[l].w1x1T[i][o] = *w++;
      for (int o = 0; o < 12; ++o) la0[l].b1x1[o] = *w++;
    }

    for (int o = 0; o < 6; ++o)
      for (int i = 0; i < 12; ++i)
        la0_head_wT[i][o] = *w++;

    // ── LA1 ──────────────────────────────────────────────────────────────────
    for (int o = 0; o < 6; ++o)
      for (int i = 0; i < 12; ++i)
        la1_rechannel_wT[i][o] = *w++;

    for (int l = 0; l < 13; ++l)
    {
      for (int o = 0; o < 6; ++o)
        for (int i = 0; i < 6; ++i)
          for (int k = 0; k < 3; ++k)
          {
            const float val = *w++;
            if      (k == 2) la1[l].w0T[i][o] = val;
            else if (k == 1) la1[l].w1T[i][o] = val;
            else             la1[l].w2T[i][o] = val;
          }

      for (int o = 0; o < 6; ++o) la1[l].conv_b[o]  = *w++;
      for (int o = 0; o < 6; ++o) la1[l].mixin_w[o] = *w++;
      for (int o = 0; o < 6; ++o)
        for (int i = 0; i < 6; ++i)
          la1[l].w1x1T[i][o] = *w++;
      for (int o = 0; o < 6; ++o) la1[l].b1x1[o] = *w++;
    }

    for (int o = 0; o < 6; ++o) la1_head_w[o] = *w++;
    la1_head_b = *w++;
    head_scale = *w++;

    assert((int)(w - model_weights) == 6554);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // forward — process exactly NAM_BLOCK samples
  //
  // Structure (layer-outer, sample-inner):
  //
  //   LA0:
  //     rechannel input[n] → la0_buf_a[n][12]
  //     zero la0_head
  //     for each of 7 layers:
  //       Layer12_process_block(ping, pong, input, la0_head)
  //       swap ping/pong
  //     project la0_head[n][12] → la1_head[n][6]  (initial head for LA1)
  //
  //   LA1:
  //     rechannel la0_buf_b[n][12] → la1_buf_a[n][6]  (7 swaps: output in b)
  //     for each of 13 layers:
  //       Layer6_process_block(ping, pong, input, la1_head)
  //       swap ping/pong
  //
  //   Output:
  //     output[n] = head_scale * (la1_head_b + la1_head[n] @ la1_head_w)
  // ─────────────────────────────────────────────────────────────────────────────
  void forward_impl(const float* input, float* output) noexcept
  {
    const int N = (int)NAM_BLOCK;

    // ── LA0 rechannel: 1→12 ───────────────────────────────────────────────
    for (int o = 0; o < 12; ++o)
    {
      const float r = la0_rechannel_w[o];
      for (int n = 0; n < N; ++n)
        la0_buf_a[n][o] = r * input[n];
    }

    // ── Zero LA0 head accumulator ─────────────────────────────────────────
    for (int n = 0; n < N; ++n)
      for (int o = 0; o < 12; ++o)
        la0_head[n][o] = 0.0f;

    // ── LA0 layer loop ────────────────────────────────────────────────────
    // 7 swaps (odd): final output lands in la0_buf_b
    {
      const float (*ping)[12] = la0_buf_a;
      float       (*pong)[12] = la0_buf_b;

      for (int l = 0; l < 7; ++l)
      {
        Layer12_process_block(la0[l], ping, pong, input, la0_head, N);
        float (*tmp)[12] = pong; pong = (float(*)[12])ping; ping = tmp;
      }
      (void)ping;
    }

    // ── Project LA0 head 12→6 → initialise la1_head ──────────────────────
    for (int n = 0; n < N; ++n)
    {
      for (int o = 0; o < 6; ++o)
      {
        float acc = 0.0f;
        for (int i = 0; i < 12; ++i)
          acc += la0_head[n][i] * la0_head_wT[i][o];
        la1_head[n][o] = acc;
      }
    }

    // ── LA1 rechannel: 12→6 (source: la0_buf_b, output of LA0) ──────────
    for (int n = 0; n < N; ++n)
    {
      for (int o = 0; o < 6; ++o)
      {
        float acc = 0.0f;
        for (int i = 0; i < 12; ++i)
          acc += la0_buf_b[n][i] * la1_rechannel_wT[i][o];
        la1_buf_a[n][o] = acc;
      }
    }

    // ── LA1 layer loop ────────────────────────────────────────────────────
    // 13 swaps (odd): final output lands in la1_buf_b (unused — only head needed)
    {
      const float (*ping)[6] = la1_buf_a;
      float       (*pong)[6] = la1_buf_b;

      for (int l = 0; l < 13; ++l)
      {
        Layer6_process_block(la1[l], ping, pong, input, la1_head, N);
        float (*tmp)[6] = pong; pong = (float(*)[6])ping; ping = tmp;
      }
    }

    // ── Output: head → scalar ─────────────────────────────────────────────
    {
      const float b  = la1_head_b;
      const float sc = head_scale;
      for (int n = 0; n < N; ++n)
      {
        float acc = b;
        for (int o = 0; o < 6; ++o)
          acc += la1_head[n][o] * la1_head_w[o];
        output[n] = sc * acc;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Layer6_process_block  (6-channel; used for LA1)
  //
  // For each sample n:
  //   1. Write ins[n] into circular state buffer.
  //   2. Read s1 = state[ptr-D], s2 = state[ptr-2D].
  //   3. Conv: t[o] = conv_b[o] + W0@ins[n] + W1@s1 + W2@s2 + mixin[o]*cond[n]
  //   4. a[o] = tanh(t[o])
  //   5. head[n][o] += a[o]
  //   6. outs[n][o] = ins[n][o] + b1x1[o] + W1x1 @ a
  // ─────────────────────────────────────────────────────────────────────────────
  void Layer6_process_block(Layer6& layer,
                            const float ins[][6], float outs[][6],
                            const float* cond,    float head[][6],
                            int N) noexcept
  {
    const float (*w0T)[6]  = layer.w0T;
    const float (*w1T)[6]  = layer.w1T;
    const float (*w2T)[6]  = layer.w2T;
    const float (*w1x1T)[6] = layer.w1x1T;
    const float* conv_b   = layer.conv_b;
    const float* mixin_w  = layer.mixin_w;
    const float* b1x1     = layer.b1x1;

    const int ss  = layer.state_size;
    const int nd  = layer.neg_dilation;
    const int nd2 = layer.neg_2dilation;
    int ptr = layer.state_ptr;

    for (int n = 0; n < N; ++n)
    {
      // ── Write tap0 into circular state ───────────────────────────────
      float* const cw = layer.state + ptr * 6;
      for (int c = 0; c < 6; ++c) cw[c] = ins[n][c];

      // ── Tap positions ─────────────────────────────────────────────────
      int sp1 = ptr + nd;  if (sp1 >= ss) sp1 -= ss;
      int sp2 = ptr + nd2; if (sp2 >= ss) sp2 -= ss;
      ++ptr; if (ptr == ss) ptr = 0;

      const float* __restrict__ s1 = layer.state + sp1 * 6;
      const float* __restrict__ s2 = layer.state + sp2 * 6;

      // ── Conv: bias + mixin + W0@i + W1@s1 + W2@s2 ────────────────────
      float t[6];
      const float cn = cond[n];
      for (int o = 0; o < 6; ++o)
        t[o] = conv_b[o] + mixin_w[o] * cn;

      for (int i = 0; i < 6; ++i)
      {
        const float iv  = ins[n][i];
        const float s1v = s1[i];
        const float s2v = s2[i];
        for (int o = 0; o < 6; ++o)
          t[o] += iv*w0T[i][o] + s1v*w1T[i][o] + s2v*w2T[i][o];
      }

      // ── Activation + head accumulation ───────────────────────────────
      float a[6];
      for (int o = 0; o < 6; ++o) { a[o] = nam_tanh(t[o]); head[n][o] += a[o]; }

      // ── 1×1 + residual ────────────────────────────────────────────────
      float out[6];
      for (int o = 0; o < 6; ++o) out[o] = ins[n][o] + b1x1[o];
      for (int i = 0; i < 6; ++i)
      {
        const float av = a[i];
        for (int o = 0; o < 6; ++o) out[o] += av * w1x1T[i][o];
      }
      for (int o = 0; o < 6; ++o) outs[n][o] = out[o];
    }

    layer.state_ptr = ptr;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Layer12_process_block  (12-channel; used for LA0)
  //
  // Identical structure to Layer6_process_block, scaled to 12 channels.
  // ─────────────────────────────────────────────────────────────────────────────
  void Layer12_process_block(Layer12& layer,
                             const float ins[][12], float outs[][12],
                             const float* cond,     float head[][12],
                             int N) noexcept
  {
    const float (*w0T)[12]  = layer.w0T;
    const float (*w1T)[12]  = layer.w1T;
    const float (*w2T)[12]  = layer.w2T;
    const float (*w1x1T)[12] = layer.w1x1T;
    const float* conv_b   = layer.conv_b;
    const float* mixin_w  = layer.mixin_w;
    const float* b1x1     = layer.b1x1;

    const int ss  = layer.state_size;
    const int nd  = layer.neg_dilation;
    const int nd2 = layer.neg_2dilation;
    int ptr = layer.state_ptr;

    for (int n = 0; n < N; ++n)
    {
      // ── Write tap0 into circular state ───────────────────────────────
      float* const cw = layer.state + ptr * 12;
      for (int c = 0; c < 12; ++c) cw[c] = ins[n][c];

      // ── Tap positions ─────────────────────────────────────────────────
      int sp1 = ptr + nd;  if (sp1 >= ss) sp1 -= ss;
      int sp2 = ptr + nd2; if (sp2 >= ss) sp2 -= ss;
      ++ptr; if (ptr == ss) ptr = 0;

      const float* __restrict__ s1 = layer.state + sp1 * 12;
      const float* __restrict__ s2 = layer.state + sp2 * 12;

      // ── Conv: bias + mixin + W0@i + W1@s1 + W2@s2 ────────────────────
      float t[12];
      const float cn = cond[n];
      for (int o = 0; o < 12; ++o)
        t[o] = conv_b[o] + mixin_w[o] * cn;

      for (int i = 0; i < 12; ++i)
      {
        const float iv  = ins[n][i];
        const float s1v = s1[i];
        const float s2v = s2[i];
        for (int o = 0; o < 12; ++o)
          t[o] += iv*w0T[i][o] + s1v*w1T[i][o] + s2v*w2T[i][o];
      }

      // ── Activation + head accumulation ───────────────────────────────
      float a[12];
      for (int o = 0; o < 12; ++o) { a[o] = nam_tanh(t[o]); head[n][o] += a[o]; }

      // ── 1×1 + residual ────────────────────────────────────────────────
      float out[12];
      for (int o = 0; o < 12; ++o) out[o] = ins[n][o] + b1x1[o];
      for (int i = 0; i < 12; ++i)
      {
        const float av = a[i];
        for (int o = 0; o < 12; ++o) out[o] += av * w1x1T[i][o];
      }
      for (int o = 0; o < 12; ++o) outs[n][o] = out[o];
    }

    layer.state_ptr = ptr;
  }

};

} // namespace MicroNAM
