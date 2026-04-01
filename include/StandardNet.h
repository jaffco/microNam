#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>

namespace MicroNAM {

// ─────────────────────────────────────────────────────────────────────────────
// Standard WaveNet inference engine
//
// Standard architecture (from neural-amp-modeler/nam/train/core.py):
//   LA0: 10 layers × 16-channel, kernel_size=3,
//        dilations=[1,2,4,8,16,32,64,128,256,512]
//        head_size=8  (16-ch activations projected 16→8 to seed LA1 head)
//   LA1: 10 layers × 8-channel, kernel_size=3,
//        dilations=[1,2,4,8,16,32,64,128,256,512]
//        head_size=1  (8-ch head dot-producted → scalar output)
//   head_scale=0.02
//
// load_weights expects exactly 13802 floats in the NAM file weight order.
// forward processes exactly NAM_BLOCK samples per call.
// ─────────────────────────────────────────────────────────────────────────────

template<size_t NAM_BLOCK>
class StandardNet
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

  // ── 8-channel layer (used for LA1) ───────────────────────────────────────
  struct Layer8
  {
    float w0T[8][8];    // tap0 (current input),  [in][out]
    float w1T[8][8];    // tap1 = state[ptr-D],   [in][out]
    float w2T[8][8];    // tap2 = state[ptr-2D],  [in][out]
    float conv_b[8];
    float mixin_w[8];
    float w1x1T[8][8];  // 1×1 projection, [in][out]
    float b1x1[8];

    float* state;
    int    state_size;
    int    state_ptr;
    int    neg_dilation;
    int    neg_2dilation;
  };

  // ── 16-channel layer (used for LA0) ──────────────────────────────────────
  struct Layer16
  {
    float w0T[16][16];    // tap0 (current input),  [in][out]
    float w1T[16][16];    // tap1 = state[ptr-D],   [in][out]
    float w2T[16][16];    // tap2 = state[ptr-2D],  [in][out]
    float conv_b[16];
    float mixin_w[16];
    float w1x1T[16][16];  // 1×1 projection, [in][out]
    float b1x1[16];

    float* state;
    int    state_size;
    int    state_ptr;
    int    neg_dilation;
    int    neg_2dilation;
  };

  // ── Layer Array 0: 10 × 16-channel ───────────────────────────────────────
  float   la0_rechannel_w[16];    // 1→16 input projection
  float   la0_head_wT[16][8];     // 16→8 head projection, [in][out]
  Layer16 la0[10];
  // State: sum((2d+1)*16) for d in {1,2,4,8,16,32,64,128,256,512}
  // sum(2d+1) = 3+5+9+17+33+65+129+257+513+1025 = 2056
  // 2056 * 16 = 32896
  float   la0_state[32896];

  // ── Layer Array 1: 10 × 8-channel ────────────────────────────────────────
  float  la1_rechannel_wT[16][8]; // 16→8 rechannel, [in][out]
  float  la1_head_w[8];
  float  la1_head_b;
  Layer8 la1[10];
  // State: sum((2d+1)*8) for d in {1,2,4,8,16,32,64,128,256,512}
  // 2056 * 8 = 16448
  float  la1_state[16448];

  float  head_scale;

  // ── Block working buffers ─────────────────────────────────────────────────
  float la0_buf_a[NAM_BLOCK][16];
  float la0_buf_b[NAM_BLOCK][16];
  float la0_head [NAM_BLOCK][16];  // LA0 skip-connection accumulator (16-ch)

  float la1_buf_a[NAM_BLOCK][8];
  float la1_buf_b[NAM_BLOCK][8];
  float la1_head [NAM_BLOCK][8];

  void init_state_ptrs()
  {
    static constexpr int la0_d[10] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    static constexpr int la1_d[10] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

    int off = 0;
    for (int i = 0; i < 10; ++i)
    {
      const int d  = la0_d[i];
      const int ss = (3-1)*d + 1;
      la0[i].state_size    = ss;
      la0[i].state_ptr     = 0;
      la0[i].neg_dilation  = ss - d;
      la0[i].neg_2dilation = ss - 2*d;
      la0[i].state         = la0_state + off;
      off += 16 * ss;
    }

    off = 0;
    for (int i = 0; i < 10; ++i)
    {
      const int d  = la1_d[i];
      const int ss = (3-1)*d + 1;
      la1[i].state_size    = ss;
      la1[i].state_ptr     = 0;
      la1[i].neg_dilation  = ss - d;
      la1[i].neg_2dilation = ss - 2*d;
      la1[i].state         = la1_state + off;
      off += 8 * ss;
    }
  }

  void reset_impl()
  {
    for (int i = 0; i < 32896; ++i) la0_state[i] = 0.f;
    for (int i = 0; i < 16448; ++i) la1_state[i] = 0.f;
    for (int i = 0; i < 10; ++i) la0[i].state_ptr = 0;
    for (int i = 0; i < 10; ++i) la1[i].state_ptr = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // load_weights — float[13802]
  //
  // File weight order: conv[out][in][k], k=0 oldest, k=2 newest.
  // Split at load time:
  //   file k=2 → w0T[in][out]  (tap0 = current input)
  //   file k=1 → w1T[in][out]
  //   file k=0 → w2T[in][out]
  //
  // Weight tally:
  //   la0_rechannel_w:  16
  //   LA0 × 10 layers:  10 × (16*16*3 + 16 + 16 + 16*16 + 16) = 10 × 1072 = 10720
  //   la0_head_wT:      16×8 = 128
  //   la1_rechannel_wT: 16×8 = 128
  //   LA1 × 10 layers:  10 × (8*8*3 + 8 + 8 + 8*8 + 8) = 10 × 280 = 2800
  //   la1_head_w:       8
  //   la1_head_b:       1
  //   head_scale:       1
  //   Total:            13802
  // ─────────────────────────────────────────────────────────────────────────────
  void load_weights_impl(const float* model_weights)
  {
    init_state_ptrs();
    reset_impl();

    const float* w = model_weights;

    // ── LA0 ──────────────────────────────────────────────────────────────────
    for (int o = 0; o < 16; ++o) la0_rechannel_w[o] = *w++;

    for (int l = 0; l < 10; ++l)
    {
      for (int o = 0; o < 16; ++o)
        for (int i = 0; i < 16; ++i)
          for (int k = 0; k < 3; ++k)
          {
            const float val = *w++;
            if      (k == 2) la0[l].w0T[i][o] = val;
            else if (k == 1) la0[l].w1T[i][o] = val;
            else             la0[l].w2T[i][o] = val;
          }

      for (int o = 0; o < 16; ++o) la0[l].conv_b[o]  = *w++;
      for (int o = 0; o < 16; ++o) la0[l].mixin_w[o] = *w++;
      for (int o = 0; o < 16; ++o)
        for (int i = 0; i < 16; ++i)
          la0[l].w1x1T[i][o] = *w++;
      for (int o = 0; o < 16; ++o) la0[l].b1x1[o] = *w++;
    }

    for (int o = 0; o < 8; ++o)
      for (int i = 0; i < 16; ++i)
        la0_head_wT[i][o] = *w++;

    // ── LA1 ──────────────────────────────────────────────────────────────────
    for (int o = 0; o < 8; ++o)
      for (int i = 0; i < 16; ++i)
        la1_rechannel_wT[i][o] = *w++;

    for (int l = 0; l < 10; ++l)
    {
      for (int o = 0; o < 8; ++o)
        for (int i = 0; i < 8; ++i)
          for (int k = 0; k < 3; ++k)
          {
            const float val = *w++;
            if      (k == 2) la1[l].w0T[i][o] = val;
            else if (k == 1) la1[l].w1T[i][o] = val;
            else             la1[l].w2T[i][o] = val;
          }

      for (int o = 0; o < 8; ++o) la1[l].conv_b[o]  = *w++;
      for (int o = 0; o < 8; ++o) la1[l].mixin_w[o] = *w++;
      for (int o = 0; o < 8; ++o)
        for (int i = 0; i < 8; ++i)
          la1[l].w1x1T[i][o] = *w++;
      for (int o = 0; o < 8; ++o) la1[l].b1x1[o] = *w++;
    }

    for (int o = 0; o < 8; ++o) la1_head_w[o] = *w++;
    la1_head_b = *w++;
    head_scale = *w++;

    assert((int)(w - model_weights) == 13802);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // forward — process exactly NAM_BLOCK samples
  //
  // Structure (layer-outer, sample-inner):
  //
  //   LA0:
  //     rechannel input[n] → la0_buf_a[n][16]
  //     zero la0_head
  //     for each of 10 layers:
  //       Layer16_process_block(ping, pong, input, la0_head)
  //       swap ping/pong
  //     project la0_head[n][16] → la1_head[n][8]  (initial head for LA1)
  //
  //   LA1:
  //     rechannel la0_buf_a[n][16] → la1_buf_a[n][8]  (10 swaps: output in a)
  //     for each of 10 layers:
  //       Layer8_process_block(ping, pong, input, la1_head)
  //       swap ping/pong
  //
  //   Output:
  //     output[n] = head_scale * (la1_head_b + la1_head[n] @ la1_head_w)
  //
  // Note on ping/pong: with an even number of layers (10) and starting with
  // ping=buf_a, the final layer writes into buf_a (not buf_b as in the 7- and
  // 13-layer arrays). LA1 rechannel therefore reads from la0_buf_a.
  // ─────────────────────────────────────────────────────────────────────────────
  void forward_impl(const float* input, float* output) noexcept
  {
    const int N = (int)NAM_BLOCK;

    // ── LA0 rechannel: 1→16 ───────────────────────────────────────────────
    for (int o = 0; o < 16; ++o)
    {
      const float r = la0_rechannel_w[o];
      for (int n = 0; n < N; ++n)
        la0_buf_a[n][o] = r * input[n];
    }

    // ── Zero LA0 head accumulator ─────────────────────────────────────────
    for (int n = 0; n < N; ++n)
      for (int o = 0; o < 16; ++o)
        la0_head[n][o] = 0.0f;

    // ── LA0 layer loop ────────────────────────────────────────────────────
    // 10 swaps (even): final output lands in la0_buf_a
    {
      const float (*ping)[16] = la0_buf_a;
      float       (*pong)[16] = la0_buf_b;

      for (int l = 0; l < 10; ++l)
      {
        Layer16_process_block(la0[l], ping, pong, input, la0_head, N);
        float (*tmp)[16] = pong; pong = (float(*)[16])ping; ping = tmp;
      }
      (void)ping;
    }

    // ── Project LA0 head 16→8 → initialise la1_head ──────────────────────
    for (int n = 0; n < N; ++n)
    {
      for (int o = 0; o < 8; ++o)
      {
        float acc = 0.0f;
        for (int i = 0; i < 16; ++i)
          acc += la0_head[n][i] * la0_head_wT[i][o];
        la1_head[n][o] = acc;
      }
    }

    // ── LA1 rechannel: 16→8 (source: la0_buf_a, output of LA0) ──────────
    for (int n = 0; n < N; ++n)
    {
      for (int o = 0; o < 8; ++o)
      {
        float acc = 0.0f;
        for (int i = 0; i < 16; ++i)
          acc += la0_buf_a[n][i] * la1_rechannel_wT[i][o];
        la1_buf_a[n][o] = acc;
      }
    }

    // ── LA1 layer loop ────────────────────────────────────────────────────
    // 10 swaps (even): final output lands in la1_buf_a (unused — only head needed)
    {
      const float (*ping)[8] = la1_buf_a;
      float       (*pong)[8] = la1_buf_b;

      for (int l = 0; l < 10; ++l)
      {
        Layer8_process_block(la1[l], ping, pong, input, la1_head, N);
        float (*tmp)[8] = pong; pong = (float(*)[8])ping; ping = tmp;
      }
    }

    // ── Output: head → scalar ─────────────────────────────────────────────
    {
      const float b  = la1_head_b;
      const float sc = head_scale;
      for (int n = 0; n < N; ++n)
      {
        float acc = b;
        for (int o = 0; o < 8; ++o)
          acc += la1_head[n][o] * la1_head_w[o];
        output[n] = sc * acc;
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Layer8_process_block  (8-channel; used for LA1)
  //
  // For each sample n:
  //   1. Write ins[n] into circular state buffer.
  //   2. Read s1 = state[ptr-D], s2 = state[ptr-2D].
  //   3. Conv: t[o] = conv_b[o] + W0@ins[n] + W1@s1 + W2@s2 + mixin[o]*cond[n]
  //   4. a[o] = tanh(t[o])
  //   5. head[n][o] += a[o]
  //   6. outs[n][o] = ins[n][o] + b1x1[o] + W1x1 @ a
  // ─────────────────────────────────────────────────────────────────────────────
  void Layer8_process_block(Layer8& layer,
                            const float ins[][8], float outs[][8],
                            const float* cond,    float head[][8],
                            int N) noexcept
  {
    const float (*w0T)[8]   = layer.w0T;
    const float (*w1T)[8]   = layer.w1T;
    const float (*w2T)[8]   = layer.w2T;
    const float (*w1x1T)[8] = layer.w1x1T;
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
      float* const cw = layer.state + ptr * 8;
      for (int c = 0; c < 8; ++c) cw[c] = ins[n][c];

      // ── Tap positions ─────────────────────────────────────────────────
      int sp1 = ptr + nd;  if (sp1 >= ss) sp1 -= ss;
      int sp2 = ptr + nd2; if (sp2 >= ss) sp2 -= ss;
      ++ptr; if (ptr == ss) ptr = 0;

      const float* __restrict__ s1 = layer.state + sp1 * 8;
      const float* __restrict__ s2 = layer.state + sp2 * 8;

      // ── Conv: bias + mixin + W0@i + W1@s1 + W2@s2 ────────────────────
      float t[8];
      const float cn = cond[n];
      for (int o = 0; o < 8; ++o)
        t[o] = conv_b[o] + mixin_w[o] * cn;

      for (int i = 0; i < 8; ++i)
      {
        const float iv  = ins[n][i];
        const float s1v = s1[i];
        const float s2v = s2[i];
        for (int o = 0; o < 8; ++o)
          t[o] += iv*w0T[i][o] + s1v*w1T[i][o] + s2v*w2T[i][o];
      }

      // ── Activation + head accumulation ───────────────────────────────
      float a[8];
      for (int o = 0; o < 8; ++o) { a[o] = nam_tanh(t[o]); head[n][o] += a[o]; }

      // ── 1×1 + residual ────────────────────────────────────────────────
      float out[8];
      for (int o = 0; o < 8; ++o) out[o] = ins[n][o] + b1x1[o];
      for (int i = 0; i < 8; ++i)
      {
        const float av = a[i];
        for (int o = 0; o < 8; ++o) out[o] += av * w1x1T[i][o];
      }
      for (int o = 0; o < 8; ++o) outs[n][o] = out[o];
    }

    layer.state_ptr = ptr;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Layer16_process_block  (16-channel; used for LA0)
  //
  // Identical structure to Layer8_process_block, scaled to 16 channels.
  // ─────────────────────────────────────────────────────────────────────────────
  void Layer16_process_block(Layer16& layer,
                             const float ins[][16], float outs[][16],
                             const float* cond,     float head[][16],
                             int N) noexcept
  {
    const float (*w0T)[16]   = layer.w0T;
    const float (*w1T)[16]   = layer.w1T;
    const float (*w2T)[16]   = layer.w2T;
    const float (*w1x1T)[16] = layer.w1x1T;
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
      float* const cw = layer.state + ptr * 16;
      for (int c = 0; c < 16; ++c) cw[c] = ins[n][c];

      // ── Tap positions ─────────────────────────────────────────────────
      int sp1 = ptr + nd;  if (sp1 >= ss) sp1 -= ss;
      int sp2 = ptr + nd2; if (sp2 >= ss) sp2 -= ss;
      ++ptr; if (ptr == ss) ptr = 0;

      const float* __restrict__ s1 = layer.state + sp1 * 16;
      const float* __restrict__ s2 = layer.state + sp2 * 16;

      // ── Conv: bias + mixin + W0@i + W1@s1 + W2@s2 ────────────────────
      float t[16];
      const float cn = cond[n];
      for (int o = 0; o < 16; ++o)
        t[o] = conv_b[o] + mixin_w[o] * cn;

      for (int i = 0; i < 16; ++i)
      {
        const float iv  = ins[n][i];
        const float s1v = s1[i];
        const float s2v = s2[i];
        for (int o = 0; o < 16; ++o)
          t[o] += iv*w0T[i][o] + s1v*w1T[i][o] + s2v*w2T[i][o];
      }

      // ── Activation + head accumulation ───────────────────────────────
      float a[16];
      for (int o = 0; o < 16; ++o) { a[o] = nam_tanh(t[o]); head[n][o] += a[o]; }

      // ── 1×1 + residual ────────────────────────────────────────────────
      float out[16];
      for (int o = 0; o < 16; ++o) out[o] = ins[n][o] + b1x1[o];
      for (int i = 0; i < 16; ++i)
      {
        const float av = a[i];
        for (int o = 0; o < 16; ++o) out[o] += av * w1x1T[i][o];
      }
      for (int o = 0; o < 16; ++o) outs[n][o] = out[o];
    }

    layer.state_ptr = ptr;
  }

};

} // namespace MicroNAM
