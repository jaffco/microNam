#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>

namespace MicroNAM {

// ─────────────────────────────────────────────────────────────────────────────
// Feather WaveNet inference engine
//
// Feather architecture (from neural-amp-modeler/nam/train/core.py):
//   LA0: 7 layers × 8-channel, kernel_size=3, dilations=[1,2,4,8,16,32,64]
//        head_size=4  (8-ch activations projected 8→4 to seed LA1 head)
//   LA1: 13 layers × 4-channel, kernel_size=3,
//        dilations=[128,256,512,1,2,4,8,16,32,64,128,256,512]
//        head_size=1  (4-ch head dot-producted → scalar output)
//   head_scale=0.02
//
// load_weights expects exactly 3026 floats in the NAM file weight order.
// forward processes exactly NAM_BLOCK samples per call.
// ─────────────────────────────────────────────────────────────────────────────

template<size_t NAM_BLOCK>
class FeatherNet
{
public:
  // ── API ───────────────────────────────────────────────────────────────
  void reset() { reset_impl(); }
  void load_weights(const float* model_weights) { load_weights_impl(model_weights); }
  void forward(const float* input, float* output) noexcept { forward_impl(input, output); }
  //   ^ Process exactly NAM_BLOCK samples.
  // ──────────────────────────────────────────────────────────────────────

private:
  // ─────────────────────────────────────────────────────────────────────────────
  // Padé(3,2) tanh — same approximation as NanoNAM.
  // ─────────────────────────────────────────────────────────────────────────────
  static inline float nam_tanh(float x) noexcept
  {
    const float x2  = x * x;
    const float num = x * (27.0f + x2);
    const float den = 27.0f + 9.0f * x2;
    return num / den;
  }

  // ── 4-channel layer (used for LA1) ───────────────────────────────────────
  struct Layer4
  {
    float w0T[4][4];    // tap0 (current input),  [in][out]
    float w1T[4][4];    // tap1 = state[ptr-D],   [in][out]
    float w2T[4][4];    // tap2 = state[ptr-2D],  [in][out]
    float conv_b[4];
    float mixin_w[4];
    float w1x1T[4][4];  // 1×1 projection, [in][out]
    float b1x1[4];

    float* state;
    int    state_size;
    int    state_ptr;
    int    neg_dilation;
    int    neg_2dilation;
  };

  // ── 8-channel layer (used for LA0) ───────────────────────────────────────
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

  // ── Layer Array 0: 7 × 8-channel ─────────────────────────────────────────
  float  la0_rechannel_w[8];      // 1→8 input projection
  float  la0_head_wT[8][4];       // 8→4 head projection, [in][out]
  Layer8 la0[7];
  // State: sum((2d+1)*8) for d in {1,2,4,8,16,32,64}
  // = 24+40+72+136+264+520+1032 = 2088
  float  la0_state[2088];

  // ── Layer Array 1: 13 × 4-channel ────────────────────────────────────────
  float  la1_rechannel_wT[8][4];  // 8→4 rechannel, [in][out]
  float  la1_head_w[4];
  float  la1_head_b;
  Layer4 la1[13];
  // State: sum((2d+1)*4) for d in {128,256,512,1,2,4,8,16,32,64,128,256,512}
  // = 1028+2052+4100+12+20+36+68+132+260+516+1028+2052+4100 = 15404
  float  la1_state[15404];

  float  head_scale;

  // ── Block working buffers ─────────────────────────────────────────────────
  float la0_buf_a[NAM_BLOCK][8];
  float la0_buf_b[NAM_BLOCK][8];
  float la0_head [NAM_BLOCK][8];   // LA0 skip-connection accumulator (8-ch)

  float la1_buf_a[NAM_BLOCK][4];
  float la1_buf_b[NAM_BLOCK][4];
  float la1_head [NAM_BLOCK][4];

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
      off += 8 * ss;
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
      off += 4 * ss;
    }
  }

  void reset_impl()
  {
    for (int i = 0; i < 2088;  ++i) la0_state[i] = 0.f;
    for (int i = 0; i < 15404; ++i) la1_state[i] = 0.f;
    for (int i = 0; i < 7;  ++i) la0[i].state_ptr = 0;
    for (int i = 0; i < 13; ++i) la1[i].state_ptr = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // load_weights — float[3026]
  //
  // File weight order: conv[out][in][k], k=0 oldest, k=2 newest.
  // Split at load time:
  //   file k=2 → w0T[in][out]  (tap0 = current input)
  //   file k=1 → w1T[in][out]
  //   file k=0 → w2T[in][out]
  //
  // Weight tally:
  //   la0_rechannel_w:  8
  //   LA0 × 7 layers:   7 × (8*8*3 + 8 + 8 + 8*8 + 8) = 7 × 280 = 1960
  //   la0_head_wT:      8×4 = 32
  //   la1_rechannel_wT: 8×4 = 32
  //   LA1 × 13 layers:  13 × (4*4*3 + 4 + 4 + 4*4 + 4) = 13 × 76 = 988
  //   la1_head_w:       4
  //   la1_head_b:       1
  //   head_scale:       1
  //   Total:            3026
  // ─────────────────────────────────────────────────────────────────────────────
  void load_weights_impl(const float* model_weights)
  {
    init_state_ptrs();
    reset_impl();

    const float* w = model_weights;

    // ── LA0 ──────────────────────────────────────────────────────────────────
    for (int o = 0; o < 8; ++o) la0_rechannel_w[o] = *w++;

    for (int l = 0; l < 7; ++l)
    {
      for (int o = 0; o < 8; ++o)
        for (int i = 0; i < 8; ++i)
          for (int k = 0; k < 3; ++k)
          {
            const float val = *w++;
            if      (k == 2) la0[l].w0T[i][o] = val;
            else if (k == 1) la0[l].w1T[i][o] = val;
            else             la0[l].w2T[i][o] = val;
          }

      for (int o = 0; o < 8; ++o) la0[l].conv_b[o]  = *w++;
      for (int o = 0; o < 8; ++o) la0[l].mixin_w[o] = *w++;
      for (int o = 0; o < 8; ++o)
        for (int i = 0; i < 8; ++i)
          la0[l].w1x1T[i][o] = *w++;
      for (int o = 0; o < 8; ++o) la0[l].b1x1[o] = *w++;
    }

    for (int o = 0; o < 4; ++o)
      for (int i = 0; i < 8; ++i)
        la0_head_wT[i][o] = *w++;

    // ── LA1 ──────────────────────────────────────────────────────────────────
    for (int o = 0; o < 4; ++o)
      for (int i = 0; i < 8; ++i)
        la1_rechannel_wT[i][o] = *w++;

    for (int l = 0; l < 13; ++l)
    {
      for (int o = 0; o < 4; ++o)
        for (int i = 0; i < 4; ++i)
          for (int k = 0; k < 3; ++k)
          {
            const float val = *w++;
            if      (k == 2) la1[l].w0T[i][o] = val;
            else if (k == 1) la1[l].w1T[i][o] = val;
            else             la1[l].w2T[i][o] = val;
          }

      for (int o = 0; o < 4; ++o) la1[l].conv_b[o]  = *w++;
      for (int o = 0; o < 4; ++o) la1[l].mixin_w[o] = *w++;
      for (int o = 0; o < 4; ++o)
        for (int i = 0; i < 4; ++i)
          la1[l].w1x1T[i][o] = *w++;
      for (int o = 0; o < 4; ++o) la1[l].b1x1[o] = *w++;
    }

    la1_head_w[0] = *w++;
    la1_head_w[1] = *w++;
    la1_head_w[2] = *w++;
    la1_head_w[3] = *w++;
    la1_head_b    = *w++;
    head_scale    = *w++;

    assert((int)(w - model_weights) == 3026);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // forward — process exactly NAM_BLOCK samples
  //
  // Structure (layer-outer, sample-inner):
  //
  //   LA0:
  //     rechannel input[n] → la0_buf_a[n][8]
  //     zero la0_head
  //     for each of 7 layers:
  //       Layer8_process_block(ping, pong, input, la0_head)
  //       swap ping/pong
  //     project la0_head[n][8] → la1_head[n][4]  (initial head for LA1)
  //
  //   LA1:
  //     rechannel la0_buf_b[n][8] → la1_buf_a[n][4]
  //     for each of 13 layers:
  //       Layer4_process_block(ping, pong, input, la1_head)
  //       swap ping/pong
  //
  //   Output:
  //     output[n] = head_scale * (la1_head_b + la1_head[n] @ la1_head_w)
  // ─────────────────────────────────────────────────────────────────────────────
  void forward_impl(const float* input, float* output) noexcept
  {
    const int N = (int)NAM_BLOCK;

    // ── LA0 rechannel: 1→8 ────────────────────────────────────────────────
    {
      const float r0 = la0_rechannel_w[0], r1 = la0_rechannel_w[1],
                  r2 = la0_rechannel_w[2], r3 = la0_rechannel_w[3],
                  r4 = la0_rechannel_w[4], r5 = la0_rechannel_w[5],
                  r6 = la0_rechannel_w[6], r7 = la0_rechannel_w[7];
      for (int n = 0; n < N; ++n)
      {
        const float x = input[n];
        la0_buf_a[n][0] = r0*x; la0_buf_a[n][1] = r1*x;
        la0_buf_a[n][2] = r2*x; la0_buf_a[n][3] = r3*x;
        la0_buf_a[n][4] = r4*x; la0_buf_a[n][5] = r5*x;
        la0_buf_a[n][6] = r6*x; la0_buf_a[n][7] = r7*x;
      }
    }

    // ── Zero LA0 head accumulator ─────────────────────────────────────────
    for (int n = 0; n < N; ++n)
      la0_head[n][0] = la0_head[n][1] = la0_head[n][2] = la0_head[n][3] =
      la0_head[n][4] = la0_head[n][5] = la0_head[n][6] = la0_head[n][7] = 0.0f;

    // ── LA0 layer loop ────────────────────────────────────────────────────
    // 7 swaps (odd): final output lands in la0_buf_b
    {
      const float (*ping)[8] = la0_buf_a;
      float       (*pong)[8] = la0_buf_b;

      for (int l = 0; l < 7; ++l)
      {
        Layer8_process_block(la0[l], ping, pong, input, la0_head, N);
        float (*tmp)[8] = pong; pong = (float(*)[8])ping; ping = tmp;
      }
      (void)ping;
    }

    // ── Project LA0 head 8→4 → initialise la1_head ───────────────────────
    {
      const float h00 = la0_head_wT[0][0], h01 = la0_head_wT[0][1],
                  h02 = la0_head_wT[0][2], h03 = la0_head_wT[0][3];
      const float h10 = la0_head_wT[1][0], h11 = la0_head_wT[1][1],
                  h12 = la0_head_wT[1][2], h13 = la0_head_wT[1][3];
      const float h20 = la0_head_wT[2][0], h21 = la0_head_wT[2][1],
                  h22 = la0_head_wT[2][2], h23 = la0_head_wT[2][3];
      const float h30 = la0_head_wT[3][0], h31 = la0_head_wT[3][1],
                  h32 = la0_head_wT[3][2], h33 = la0_head_wT[3][3];
      const float h40 = la0_head_wT[4][0], h41 = la0_head_wT[4][1],
                  h42 = la0_head_wT[4][2], h43 = la0_head_wT[4][3];
      const float h50 = la0_head_wT[5][0], h51 = la0_head_wT[5][1],
                  h52 = la0_head_wT[5][2], h53 = la0_head_wT[5][3];
      const float h60 = la0_head_wT[6][0], h61 = la0_head_wT[6][1],
                  h62 = la0_head_wT[6][2], h63 = la0_head_wT[6][3];
      const float h70 = la0_head_wT[7][0], h71 = la0_head_wT[7][1],
                  h72 = la0_head_wT[7][2], h73 = la0_head_wT[7][3];
      for (int n = 0; n < N; ++n)
      {
        const float v0 = la0_head[n][0], v1 = la0_head[n][1],
                    v2 = la0_head[n][2], v3 = la0_head[n][3],
                    v4 = la0_head[n][4], v5 = la0_head[n][5],
                    v6 = la0_head[n][6], v7 = la0_head[n][7];
        la1_head[n][0] = v0*h00 + v1*h10 + v2*h20 + v3*h30
                       + v4*h40 + v5*h50 + v6*h60 + v7*h70;
        la1_head[n][1] = v0*h01 + v1*h11 + v2*h21 + v3*h31
                       + v4*h41 + v5*h51 + v6*h61 + v7*h71;
        la1_head[n][2] = v0*h02 + v1*h12 + v2*h22 + v3*h32
                       + v4*h42 + v5*h52 + v6*h62 + v7*h72;
        la1_head[n][3] = v0*h03 + v1*h13 + v2*h23 + v3*h33
                       + v4*h43 + v5*h53 + v6*h63 + v7*h73;
      }
    }

    // ── LA1 rechannel: 8→4 ───────────────────────────────────────────────
    {
      const float r00 = la1_rechannel_wT[0][0], r01 = la1_rechannel_wT[0][1],
                  r02 = la1_rechannel_wT[0][2], r03 = la1_rechannel_wT[0][3];
      const float r10 = la1_rechannel_wT[1][0], r11 = la1_rechannel_wT[1][1],
                  r12 = la1_rechannel_wT[1][2], r13 = la1_rechannel_wT[1][3];
      const float r20 = la1_rechannel_wT[2][0], r21 = la1_rechannel_wT[2][1],
                  r22 = la1_rechannel_wT[2][2], r23 = la1_rechannel_wT[2][3];
      const float r30 = la1_rechannel_wT[3][0], r31 = la1_rechannel_wT[3][1],
                  r32 = la1_rechannel_wT[3][2], r33 = la1_rechannel_wT[3][3];
      const float r40 = la1_rechannel_wT[4][0], r41 = la1_rechannel_wT[4][1],
                  r42 = la1_rechannel_wT[4][2], r43 = la1_rechannel_wT[4][3];
      const float r50 = la1_rechannel_wT[5][0], r51 = la1_rechannel_wT[5][1],
                  r52 = la1_rechannel_wT[5][2], r53 = la1_rechannel_wT[5][3];
      const float r60 = la1_rechannel_wT[6][0], r61 = la1_rechannel_wT[6][1],
                  r62 = la1_rechannel_wT[6][2], r63 = la1_rechannel_wT[6][3];
      const float r70 = la1_rechannel_wT[7][0], r71 = la1_rechannel_wT[7][1],
                  r72 = la1_rechannel_wT[7][2], r73 = la1_rechannel_wT[7][3];
      for (int n = 0; n < N; ++n)
      {
        const float* const src = la0_buf_b[n];
        la1_buf_a[n][0] = src[0]*r00 + src[1]*r10 + src[2]*r20 + src[3]*r30
                        + src[4]*r40 + src[5]*r50 + src[6]*r60 + src[7]*r70;
        la1_buf_a[n][1] = src[0]*r01 + src[1]*r11 + src[2]*r21 + src[3]*r31
                        + src[4]*r41 + src[5]*r51 + src[6]*r61 + src[7]*r71;
        la1_buf_a[n][2] = src[0]*r02 + src[1]*r12 + src[2]*r22 + src[3]*r32
                        + src[4]*r42 + src[5]*r52 + src[6]*r62 + src[7]*r72;
        la1_buf_a[n][3] = src[0]*r03 + src[1]*r13 + src[2]*r23 + src[3]*r33
                        + src[4]*r43 + src[5]*r53 + src[6]*r63 + src[7]*r73;
      }
    }

    // ── LA1 layer loop ────────────────────────────────────────────────────
    // 13 swaps (odd): final output lands in la1_buf_b (unused — only head needed)
    {
      const float (*ping)[4] = la1_buf_a;
      float       (*pong)[4] = la1_buf_b;

      for (int l = 0; l < 13; ++l)
      {
        Layer4_process_block(la1[l], ping, pong, input, la1_head, N);
        float (*tmp)[4] = pong; pong = (float(*)[4])ping; ping = tmp;
      }
    }

    // ── Output: head → scalar ─────────────────────────────────────────────
    {
      const float w0 = la1_head_w[0], w1 = la1_head_w[1],
                  w2 = la1_head_w[2], w3 = la1_head_w[3];
      const float b  = la1_head_b;
      const float sc = head_scale;
      for (int n = 0; n < N; ++n)
        output[n] = sc * (b + la1_head[n][0]*w0 + la1_head[n][1]*w1
                            + la1_head[n][2]*w2 + la1_head[n][3]*w3);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Layer4_process_block  (4-channel; used for LA1)
  // Identical in structure to NanoNAM's Layer4_process_block.
  // ─────────────────────────────────────────────────────────────────────────────
  void Layer4_process_block(Layer4& layer,
                            const float ins[][4], float outs[][4],
                            const float* cond,    float head[][4],
                            int N) noexcept
  {
    const float c0_00 = layer.w0T[0][0], c0_01 = layer.w0T[0][1], c0_02 = layer.w0T[0][2], c0_03 = layer.w0T[0][3];
    const float c0_10 = layer.w0T[1][0], c0_11 = layer.w0T[1][1], c0_12 = layer.w0T[1][2], c0_13 = layer.w0T[1][3];
    const float c0_20 = layer.w0T[2][0], c0_21 = layer.w0T[2][1], c0_22 = layer.w0T[2][2], c0_23 = layer.w0T[2][3];
    const float c0_30 = layer.w0T[3][0], c0_31 = layer.w0T[3][1], c0_32 = layer.w0T[3][2], c0_33 = layer.w0T[3][3];

    const float c1_00 = layer.w1T[0][0], c1_01 = layer.w1T[0][1], c1_02 = layer.w1T[0][2], c1_03 = layer.w1T[0][3];
    const float c1_10 = layer.w1T[1][0], c1_11 = layer.w1T[1][1], c1_12 = layer.w1T[1][2], c1_13 = layer.w1T[1][3];
    const float c1_20 = layer.w1T[2][0], c1_21 = layer.w1T[2][1], c1_22 = layer.w1T[2][2], c1_23 = layer.w1T[2][3];
    const float c1_30 = layer.w1T[3][0], c1_31 = layer.w1T[3][1], c1_32 = layer.w1T[3][2], c1_33 = layer.w1T[3][3];

    const float c2_00 = layer.w2T[0][0], c2_01 = layer.w2T[0][1], c2_02 = layer.w2T[0][2], c2_03 = layer.w2T[0][3];
    const float c2_10 = layer.w2T[1][0], c2_11 = layer.w2T[1][1], c2_12 = layer.w2T[1][2], c2_13 = layer.w2T[1][3];
    const float c2_20 = layer.w2T[2][0], c2_21 = layer.w2T[2][1], c2_22 = layer.w2T[2][2], c2_23 = layer.w2T[2][3];
    const float c2_30 = layer.w2T[3][0], c2_31 = layer.w2T[3][1], c2_32 = layer.w2T[3][2], c2_33 = layer.w2T[3][3];

    const float cx_00 = layer.w1x1T[0][0], cx_01 = layer.w1x1T[0][1], cx_02 = layer.w1x1T[0][2], cx_03 = layer.w1x1T[0][3];
    const float cx_10 = layer.w1x1T[1][0], cx_11 = layer.w1x1T[1][1], cx_12 = layer.w1x1T[1][2], cx_13 = layer.w1x1T[1][3];
    const float cx_20 = layer.w1x1T[2][0], cx_21 = layer.w1x1T[2][1], cx_22 = layer.w1x1T[2][2], cx_23 = layer.w1x1T[2][3];
    const float cx_30 = layer.w1x1T[3][0], cx_31 = layer.w1x1T[3][1], cx_32 = layer.w1x1T[3][2], cx_33 = layer.w1x1T[3][3];

    const float mb0 = layer.conv_b[0], mb1 = layer.conv_b[1],
                mb2 = layer.conv_b[2], mb3 = layer.conv_b[3];
    const float mx0 = layer.mixin_w[0], mx1 = layer.mixin_w[1],
                mx2 = layer.mixin_w[2], mx3 = layer.mixin_w[3];
    const float bx0 = layer.b1x1[0], bx1 = layer.b1x1[1],
                bx2 = layer.b1x1[2], bx3 = layer.b1x1[3];

    const int ss  = layer.state_size;
    const int nd  = layer.neg_dilation;
    const int nd2 = layer.neg_2dilation;
    int ptr = layer.state_ptr;

    for (int n = 0; n < N; ++n)
    {
      float* const cw = layer.state + ptr * 4;
      const float i0 = ins[n][0], i1 = ins[n][1],
                  i2 = ins[n][2], i3 = ins[n][3];
      cw[0] = i0; cw[1] = i1; cw[2] = i2; cw[3] = i3;

      int sp1 = ptr + nd;  if (sp1 >= ss) sp1 -= ss;
      int sp2 = ptr + nd2; if (sp2 >= ss) sp2 -= ss;

      ++ptr; if (ptr == ss) ptr = 0;

      const float* __restrict__ s1 = layer.state + sp1 * 4;
      const float* __restrict__ s2 = layer.state + sp2 * 4;

      float t0 = mb0, t1 = mb1, t2 = mb2, t3 = mb3;

      t0 += i0*c0_00 + i1*c0_10 + i2*c0_20 + i3*c0_30;
      t1 += i0*c0_01 + i1*c0_11 + i2*c0_21 + i3*c0_31;
      t2 += i0*c0_02 + i1*c0_12 + i2*c0_22 + i3*c0_32;
      t3 += i0*c0_03 + i1*c0_13 + i2*c0_23 + i3*c0_33;

      const float s1_0 = s1[0], s1_1 = s1[1], s1_2 = s1[2], s1_3 = s1[3];
      t0 += s1_0*c1_00 + s1_1*c1_10 + s1_2*c1_20 + s1_3*c1_30;
      t1 += s1_0*c1_01 + s1_1*c1_11 + s1_2*c1_21 + s1_3*c1_31;
      t2 += s1_0*c1_02 + s1_1*c1_12 + s1_2*c1_22 + s1_3*c1_32;
      t3 += s1_0*c1_03 + s1_1*c1_13 + s1_2*c1_23 + s1_3*c1_33;

      const float s2_0 = s2[0], s2_1 = s2[1], s2_2 = s2[2], s2_3 = s2[3];
      t0 += s2_0*c2_00 + s2_1*c2_10 + s2_2*c2_20 + s2_3*c2_30;
      t1 += s2_0*c2_01 + s2_1*c2_11 + s2_2*c2_21 + s2_3*c2_31;
      t2 += s2_0*c2_02 + s2_1*c2_12 + s2_2*c2_22 + s2_3*c2_32;
      t3 += s2_0*c2_03 + s2_1*c2_13 + s2_2*c2_23 + s2_3*c2_33;

      const float cn = cond[n];
      t0 += mx0*cn; t1 += mx1*cn; t2 += mx2*cn; t3 += mx3*cn;

      const float a0 = nam_tanh(t0), a1 = nam_tanh(t1),
                  a2 = nam_tanh(t2), a3 = nam_tanh(t3);

      head[n][0] += a0; head[n][1] += a1;
      head[n][2] += a2; head[n][3] += a3;

      float o0 = i0 + bx0, o1 = i1 + bx1, o2 = i2 + bx2, o3 = i3 + bx3;
      o0 += a0*cx_00 + a1*cx_10 + a2*cx_20 + a3*cx_30;
      o1 += a0*cx_01 + a1*cx_11 + a2*cx_21 + a3*cx_31;
      o2 += a0*cx_02 + a1*cx_12 + a2*cx_22 + a3*cx_32;
      o3 += a0*cx_03 + a1*cx_13 + a2*cx_23 + a3*cx_33;

      outs[n][0] = o0; outs[n][1] = o1;
      outs[n][2] = o2; outs[n][3] = o3;
    }

    layer.state_ptr = ptr;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Layer8_process_block  (8-channel; used for LA0)
  //
  // Same structure as Layer4_process_block, scaled to 8×8 weight matrices.
  // Weights hoisted as local const for register allocation over the N loop.
  // ─────────────────────────────────────────────────────────────────────────────
  void Layer8_process_block(Layer8& layer,
                            const float ins[][8], float outs[][8],
                            const float* cond,    float head[][8],
                            int N) noexcept
  {
    // ── Hoist w0T[8][8] ───────────────────────────────────────────────────
    const float c0_00 = layer.w0T[0][0], c0_01 = layer.w0T[0][1], c0_02 = layer.w0T[0][2], c0_03 = layer.w0T[0][3],
                c0_04 = layer.w0T[0][4], c0_05 = layer.w0T[0][5], c0_06 = layer.w0T[0][6], c0_07 = layer.w0T[0][7];
    const float c0_10 = layer.w0T[1][0], c0_11 = layer.w0T[1][1], c0_12 = layer.w0T[1][2], c0_13 = layer.w0T[1][3],
                c0_14 = layer.w0T[1][4], c0_15 = layer.w0T[1][5], c0_16 = layer.w0T[1][6], c0_17 = layer.w0T[1][7];
    const float c0_20 = layer.w0T[2][0], c0_21 = layer.w0T[2][1], c0_22 = layer.w0T[2][2], c0_23 = layer.w0T[2][3],
                c0_24 = layer.w0T[2][4], c0_25 = layer.w0T[2][5], c0_26 = layer.w0T[2][6], c0_27 = layer.w0T[2][7];
    const float c0_30 = layer.w0T[3][0], c0_31 = layer.w0T[3][1], c0_32 = layer.w0T[3][2], c0_33 = layer.w0T[3][3],
                c0_34 = layer.w0T[3][4], c0_35 = layer.w0T[3][5], c0_36 = layer.w0T[3][6], c0_37 = layer.w0T[3][7];
    const float c0_40 = layer.w0T[4][0], c0_41 = layer.w0T[4][1], c0_42 = layer.w0T[4][2], c0_43 = layer.w0T[4][3],
                c0_44 = layer.w0T[4][4], c0_45 = layer.w0T[4][5], c0_46 = layer.w0T[4][6], c0_47 = layer.w0T[4][7];
    const float c0_50 = layer.w0T[5][0], c0_51 = layer.w0T[5][1], c0_52 = layer.w0T[5][2], c0_53 = layer.w0T[5][3],
                c0_54 = layer.w0T[5][4], c0_55 = layer.w0T[5][5], c0_56 = layer.w0T[5][6], c0_57 = layer.w0T[5][7];
    const float c0_60 = layer.w0T[6][0], c0_61 = layer.w0T[6][1], c0_62 = layer.w0T[6][2], c0_63 = layer.w0T[6][3],
                c0_64 = layer.w0T[6][4], c0_65 = layer.w0T[6][5], c0_66 = layer.w0T[6][6], c0_67 = layer.w0T[6][7];
    const float c0_70 = layer.w0T[7][0], c0_71 = layer.w0T[7][1], c0_72 = layer.w0T[7][2], c0_73 = layer.w0T[7][3],
                c0_74 = layer.w0T[7][4], c0_75 = layer.w0T[7][5], c0_76 = layer.w0T[7][6], c0_77 = layer.w0T[7][7];

    // ── Hoist w1T[8][8] ───────────────────────────────────────────────────
    const float c1_00 = layer.w1T[0][0], c1_01 = layer.w1T[0][1], c1_02 = layer.w1T[0][2], c1_03 = layer.w1T[0][3],
                c1_04 = layer.w1T[0][4], c1_05 = layer.w1T[0][5], c1_06 = layer.w1T[0][6], c1_07 = layer.w1T[0][7];
    const float c1_10 = layer.w1T[1][0], c1_11 = layer.w1T[1][1], c1_12 = layer.w1T[1][2], c1_13 = layer.w1T[1][3],
                c1_14 = layer.w1T[1][4], c1_15 = layer.w1T[1][5], c1_16 = layer.w1T[1][6], c1_17 = layer.w1T[1][7];
    const float c1_20 = layer.w1T[2][0], c1_21 = layer.w1T[2][1], c1_22 = layer.w1T[2][2], c1_23 = layer.w1T[2][3],
                c1_24 = layer.w1T[2][4], c1_25 = layer.w1T[2][5], c1_26 = layer.w1T[2][6], c1_27 = layer.w1T[2][7];
    const float c1_30 = layer.w1T[3][0], c1_31 = layer.w1T[3][1], c1_32 = layer.w1T[3][2], c1_33 = layer.w1T[3][3],
                c1_34 = layer.w1T[3][4], c1_35 = layer.w1T[3][5], c1_36 = layer.w1T[3][6], c1_37 = layer.w1T[3][7];
    const float c1_40 = layer.w1T[4][0], c1_41 = layer.w1T[4][1], c1_42 = layer.w1T[4][2], c1_43 = layer.w1T[4][3],
                c1_44 = layer.w1T[4][4], c1_45 = layer.w1T[4][5], c1_46 = layer.w1T[4][6], c1_47 = layer.w1T[4][7];
    const float c1_50 = layer.w1T[5][0], c1_51 = layer.w1T[5][1], c1_52 = layer.w1T[5][2], c1_53 = layer.w1T[5][3],
                c1_54 = layer.w1T[5][4], c1_55 = layer.w1T[5][5], c1_56 = layer.w1T[5][6], c1_57 = layer.w1T[5][7];
    const float c1_60 = layer.w1T[6][0], c1_61 = layer.w1T[6][1], c1_62 = layer.w1T[6][2], c1_63 = layer.w1T[6][3],
                c1_64 = layer.w1T[6][4], c1_65 = layer.w1T[6][5], c1_66 = layer.w1T[6][6], c1_67 = layer.w1T[6][7];
    const float c1_70 = layer.w1T[7][0], c1_71 = layer.w1T[7][1], c1_72 = layer.w1T[7][2], c1_73 = layer.w1T[7][3],
                c1_74 = layer.w1T[7][4], c1_75 = layer.w1T[7][5], c1_76 = layer.w1T[7][6], c1_77 = layer.w1T[7][7];

    // ── Hoist w2T[8][8] ───────────────────────────────────────────────────
    const float c2_00 = layer.w2T[0][0], c2_01 = layer.w2T[0][1], c2_02 = layer.w2T[0][2], c2_03 = layer.w2T[0][3],
                c2_04 = layer.w2T[0][4], c2_05 = layer.w2T[0][5], c2_06 = layer.w2T[0][6], c2_07 = layer.w2T[0][7];
    const float c2_10 = layer.w2T[1][0], c2_11 = layer.w2T[1][1], c2_12 = layer.w2T[1][2], c2_13 = layer.w2T[1][3],
                c2_14 = layer.w2T[1][4], c2_15 = layer.w2T[1][5], c2_16 = layer.w2T[1][6], c2_17 = layer.w2T[1][7];
    const float c2_20 = layer.w2T[2][0], c2_21 = layer.w2T[2][1], c2_22 = layer.w2T[2][2], c2_23 = layer.w2T[2][3],
                c2_24 = layer.w2T[2][4], c2_25 = layer.w2T[2][5], c2_26 = layer.w2T[2][6], c2_27 = layer.w2T[2][7];
    const float c2_30 = layer.w2T[3][0], c2_31 = layer.w2T[3][1], c2_32 = layer.w2T[3][2], c2_33 = layer.w2T[3][3],
                c2_34 = layer.w2T[3][4], c2_35 = layer.w2T[3][5], c2_36 = layer.w2T[3][6], c2_37 = layer.w2T[3][7];
    const float c2_40 = layer.w2T[4][0], c2_41 = layer.w2T[4][1], c2_42 = layer.w2T[4][2], c2_43 = layer.w2T[4][3],
                c2_44 = layer.w2T[4][4], c2_45 = layer.w2T[4][5], c2_46 = layer.w2T[4][6], c2_47 = layer.w2T[4][7];
    const float c2_50 = layer.w2T[5][0], c2_51 = layer.w2T[5][1], c2_52 = layer.w2T[5][2], c2_53 = layer.w2T[5][3],
                c2_54 = layer.w2T[5][4], c2_55 = layer.w2T[5][5], c2_56 = layer.w2T[5][6], c2_57 = layer.w2T[5][7];
    const float c2_60 = layer.w2T[6][0], c2_61 = layer.w2T[6][1], c2_62 = layer.w2T[6][2], c2_63 = layer.w2T[6][3],
                c2_64 = layer.w2T[6][4], c2_65 = layer.w2T[6][5], c2_66 = layer.w2T[6][6], c2_67 = layer.w2T[6][7];
    const float c2_70 = layer.w2T[7][0], c2_71 = layer.w2T[7][1], c2_72 = layer.w2T[7][2], c2_73 = layer.w2T[7][3],
                c2_74 = layer.w2T[7][4], c2_75 = layer.w2T[7][5], c2_76 = layer.w2T[7][6], c2_77 = layer.w2T[7][7];

    // ── Hoist w1x1T[8][8] ─────────────────────────────────────────────────
    const float cx_00 = layer.w1x1T[0][0], cx_01 = layer.w1x1T[0][1], cx_02 = layer.w1x1T[0][2], cx_03 = layer.w1x1T[0][3],
                cx_04 = layer.w1x1T[0][4], cx_05 = layer.w1x1T[0][5], cx_06 = layer.w1x1T[0][6], cx_07 = layer.w1x1T[0][7];
    const float cx_10 = layer.w1x1T[1][0], cx_11 = layer.w1x1T[1][1], cx_12 = layer.w1x1T[1][2], cx_13 = layer.w1x1T[1][3],
                cx_14 = layer.w1x1T[1][4], cx_15 = layer.w1x1T[1][5], cx_16 = layer.w1x1T[1][6], cx_17 = layer.w1x1T[1][7];
    const float cx_20 = layer.w1x1T[2][0], cx_21 = layer.w1x1T[2][1], cx_22 = layer.w1x1T[2][2], cx_23 = layer.w1x1T[2][3],
                cx_24 = layer.w1x1T[2][4], cx_25 = layer.w1x1T[2][5], cx_26 = layer.w1x1T[2][6], cx_27 = layer.w1x1T[2][7];
    const float cx_30 = layer.w1x1T[3][0], cx_31 = layer.w1x1T[3][1], cx_32 = layer.w1x1T[3][2], cx_33 = layer.w1x1T[3][3],
                cx_34 = layer.w1x1T[3][4], cx_35 = layer.w1x1T[3][5], cx_36 = layer.w1x1T[3][6], cx_37 = layer.w1x1T[3][7];
    const float cx_40 = layer.w1x1T[4][0], cx_41 = layer.w1x1T[4][1], cx_42 = layer.w1x1T[4][2], cx_43 = layer.w1x1T[4][3],
                cx_44 = layer.w1x1T[4][4], cx_45 = layer.w1x1T[4][5], cx_46 = layer.w1x1T[4][6], cx_47 = layer.w1x1T[4][7];
    const float cx_50 = layer.w1x1T[5][0], cx_51 = layer.w1x1T[5][1], cx_52 = layer.w1x1T[5][2], cx_53 = layer.w1x1T[5][3],
                cx_54 = layer.w1x1T[5][4], cx_55 = layer.w1x1T[5][5], cx_56 = layer.w1x1T[5][6], cx_57 = layer.w1x1T[5][7];
    const float cx_60 = layer.w1x1T[6][0], cx_61 = layer.w1x1T[6][1], cx_62 = layer.w1x1T[6][2], cx_63 = layer.w1x1T[6][3],
                cx_64 = layer.w1x1T[6][4], cx_65 = layer.w1x1T[6][5], cx_66 = layer.w1x1T[6][6], cx_67 = layer.w1x1T[6][7];
    const float cx_70 = layer.w1x1T[7][0], cx_71 = layer.w1x1T[7][1], cx_72 = layer.w1x1T[7][2], cx_73 = layer.w1x1T[7][3],
                cx_74 = layer.w1x1T[7][4], cx_75 = layer.w1x1T[7][5], cx_76 = layer.w1x1T[7][6], cx_77 = layer.w1x1T[7][7];

    const float mb0 = layer.conv_b[0], mb1 = layer.conv_b[1],
                mb2 = layer.conv_b[2], mb3 = layer.conv_b[3],
                mb4 = layer.conv_b[4], mb5 = layer.conv_b[5],
                mb6 = layer.conv_b[6], mb7 = layer.conv_b[7];
    const float mx0 = layer.mixin_w[0], mx1 = layer.mixin_w[1],
                mx2 = layer.mixin_w[2], mx3 = layer.mixin_w[3],
                mx4 = layer.mixin_w[4], mx5 = layer.mixin_w[5],
                mx6 = layer.mixin_w[6], mx7 = layer.mixin_w[7];
    const float bx0 = layer.b1x1[0], bx1 = layer.b1x1[1],
                bx2 = layer.b1x1[2], bx3 = layer.b1x1[3],
                bx4 = layer.b1x1[4], bx5 = layer.b1x1[5],
                bx6 = layer.b1x1[6], bx7 = layer.b1x1[7];

    const int ss  = layer.state_size;
    const int nd  = layer.neg_dilation;
    const int nd2 = layer.neg_2dilation;
    int ptr = layer.state_ptr;

    for (int n = 0; n < N; ++n)
    {
      // ── Write tap0 into circular state ───────────────────────────────────
      float* const cw = layer.state + ptr * 8;
      const float i0 = ins[n][0], i1 = ins[n][1], i2 = ins[n][2], i3 = ins[n][3],
                  i4 = ins[n][4], i5 = ins[n][5], i6 = ins[n][6], i7 = ins[n][7];
      cw[0]=i0; cw[1]=i1; cw[2]=i2; cw[3]=i3;
      cw[4]=i4; cw[5]=i5; cw[6]=i6; cw[7]=i7;

      int sp1 = ptr + nd;  if (sp1 >= ss) sp1 -= ss;
      int sp2 = ptr + nd2; if (sp2 >= ss) sp2 -= ss;

      ++ptr; if (ptr == ss) ptr = 0;

      const float* __restrict__ s1 = layer.state + sp1 * 8;
      const float* __restrict__ s2 = layer.state + sp2 * 8;

      // ── Conv: bias + W0@i + W1@s1 + W2@s2 + mixin*cond ───────────────────
      float t0 = mb0, t1 = mb1, t2 = mb2, t3 = mb3,
            t4 = mb4, t5 = mb5, t6 = mb6, t7 = mb7;

      // Tap 0
      t0 += i0*c0_00 + i1*c0_10 + i2*c0_20 + i3*c0_30 + i4*c0_40 + i5*c0_50 + i6*c0_60 + i7*c0_70;
      t1 += i0*c0_01 + i1*c0_11 + i2*c0_21 + i3*c0_31 + i4*c0_41 + i5*c0_51 + i6*c0_61 + i7*c0_71;
      t2 += i0*c0_02 + i1*c0_12 + i2*c0_22 + i3*c0_32 + i4*c0_42 + i5*c0_52 + i6*c0_62 + i7*c0_72;
      t3 += i0*c0_03 + i1*c0_13 + i2*c0_23 + i3*c0_33 + i4*c0_43 + i5*c0_53 + i6*c0_63 + i7*c0_73;
      t4 += i0*c0_04 + i1*c0_14 + i2*c0_24 + i3*c0_34 + i4*c0_44 + i5*c0_54 + i6*c0_64 + i7*c0_74;
      t5 += i0*c0_05 + i1*c0_15 + i2*c0_25 + i3*c0_35 + i4*c0_45 + i5*c0_55 + i6*c0_65 + i7*c0_75;
      t6 += i0*c0_06 + i1*c0_16 + i2*c0_26 + i3*c0_36 + i4*c0_46 + i5*c0_56 + i6*c0_66 + i7*c0_76;
      t7 += i0*c0_07 + i1*c0_17 + i2*c0_27 + i3*c0_37 + i4*c0_47 + i5*c0_57 + i6*c0_67 + i7*c0_77;

      // Tap 1
      const float s1_0=s1[0], s1_1=s1[1], s1_2=s1[2], s1_3=s1[3],
                  s1_4=s1[4], s1_5=s1[5], s1_6=s1[6], s1_7=s1[7];
      t0 += s1_0*c1_00 + s1_1*c1_10 + s1_2*c1_20 + s1_3*c1_30 + s1_4*c1_40 + s1_5*c1_50 + s1_6*c1_60 + s1_7*c1_70;
      t1 += s1_0*c1_01 + s1_1*c1_11 + s1_2*c1_21 + s1_3*c1_31 + s1_4*c1_41 + s1_5*c1_51 + s1_6*c1_61 + s1_7*c1_71;
      t2 += s1_0*c1_02 + s1_1*c1_12 + s1_2*c1_22 + s1_3*c1_32 + s1_4*c1_42 + s1_5*c1_52 + s1_6*c1_62 + s1_7*c1_72;
      t3 += s1_0*c1_03 + s1_1*c1_13 + s1_2*c1_23 + s1_3*c1_33 + s1_4*c1_43 + s1_5*c1_53 + s1_6*c1_63 + s1_7*c1_73;
      t4 += s1_0*c1_04 + s1_1*c1_14 + s1_2*c1_24 + s1_3*c1_34 + s1_4*c1_44 + s1_5*c1_54 + s1_6*c1_64 + s1_7*c1_74;
      t5 += s1_0*c1_05 + s1_1*c1_15 + s1_2*c1_25 + s1_3*c1_35 + s1_4*c1_45 + s1_5*c1_55 + s1_6*c1_65 + s1_7*c1_75;
      t6 += s1_0*c1_06 + s1_1*c1_16 + s1_2*c1_26 + s1_3*c1_36 + s1_4*c1_46 + s1_5*c1_56 + s1_6*c1_66 + s1_7*c1_76;
      t7 += s1_0*c1_07 + s1_1*c1_17 + s1_2*c1_27 + s1_3*c1_37 + s1_4*c1_47 + s1_5*c1_57 + s1_6*c1_67 + s1_7*c1_77;

      // Tap 2
      const float s2_0=s2[0], s2_1=s2[1], s2_2=s2[2], s2_3=s2[3],
                  s2_4=s2[4], s2_5=s2[5], s2_6=s2[6], s2_7=s2[7];
      t0 += s2_0*c2_00 + s2_1*c2_10 + s2_2*c2_20 + s2_3*c2_30 + s2_4*c2_40 + s2_5*c2_50 + s2_6*c2_60 + s2_7*c2_70;
      t1 += s2_0*c2_01 + s2_1*c2_11 + s2_2*c2_21 + s2_3*c2_31 + s2_4*c2_41 + s2_5*c2_51 + s2_6*c2_61 + s2_7*c2_71;
      t2 += s2_0*c2_02 + s2_1*c2_12 + s2_2*c2_22 + s2_3*c2_32 + s2_4*c2_42 + s2_5*c2_52 + s2_6*c2_62 + s2_7*c2_72;
      t3 += s2_0*c2_03 + s2_1*c2_13 + s2_2*c2_23 + s2_3*c2_33 + s2_4*c2_43 + s2_5*c2_53 + s2_6*c2_63 + s2_7*c2_73;
      t4 += s2_0*c2_04 + s2_1*c2_14 + s2_2*c2_24 + s2_3*c2_34 + s2_4*c2_44 + s2_5*c2_54 + s2_6*c2_64 + s2_7*c2_74;
      t5 += s2_0*c2_05 + s2_1*c2_15 + s2_2*c2_25 + s2_3*c2_35 + s2_4*c2_45 + s2_5*c2_55 + s2_6*c2_65 + s2_7*c2_75;
      t6 += s2_0*c2_06 + s2_1*c2_16 + s2_2*c2_26 + s2_3*c2_36 + s2_4*c2_46 + s2_5*c2_56 + s2_6*c2_66 + s2_7*c2_76;
      t7 += s2_0*c2_07 + s2_1*c2_17 + s2_2*c2_27 + s2_3*c2_37 + s2_4*c2_47 + s2_5*c2_57 + s2_6*c2_67 + s2_7*c2_77;

      // Mixin conditioning
      const float cn = cond[n];
      t0+=mx0*cn; t1+=mx1*cn; t2+=mx2*cn; t3+=mx3*cn;
      t4+=mx4*cn; t5+=mx5*cn; t6+=mx6*cn; t7+=mx7*cn;

      // ── Activation ────────────────────────────────────────────────────────
      const float a0=nam_tanh(t0), a1=nam_tanh(t1), a2=nam_tanh(t2), a3=nam_tanh(t3),
                  a4=nam_tanh(t4), a5=nam_tanh(t5), a6=nam_tanh(t6), a7=nam_tanh(t7);

      // ── Accumulate head ───────────────────────────────────────────────────
      head[n][0]+=a0; head[n][1]+=a1; head[n][2]+=a2; head[n][3]+=a3;
      head[n][4]+=a4; head[n][5]+=a5; head[n][6]+=a6; head[n][7]+=a7;

      // ── 1×1 + residual ────────────────────────────────────────────────────
      float o0=i0+bx0, o1=i1+bx1, o2=i2+bx2, o3=i3+bx3,
            o4=i4+bx4, o5=i5+bx5, o6=i6+bx6, o7=i7+bx7;
      o0 += a0*cx_00 + a1*cx_10 + a2*cx_20 + a3*cx_30 + a4*cx_40 + a5*cx_50 + a6*cx_60 + a7*cx_70;
      o1 += a0*cx_01 + a1*cx_11 + a2*cx_21 + a3*cx_31 + a4*cx_41 + a5*cx_51 + a6*cx_61 + a7*cx_71;
      o2 += a0*cx_02 + a1*cx_12 + a2*cx_22 + a3*cx_32 + a4*cx_42 + a5*cx_52 + a6*cx_62 + a7*cx_72;
      o3 += a0*cx_03 + a1*cx_13 + a2*cx_23 + a3*cx_33 + a4*cx_43 + a5*cx_53 + a6*cx_63 + a7*cx_73;
      o4 += a0*cx_04 + a1*cx_14 + a2*cx_24 + a3*cx_34 + a4*cx_44 + a5*cx_54 + a6*cx_64 + a7*cx_74;
      o5 += a0*cx_05 + a1*cx_15 + a2*cx_25 + a3*cx_35 + a4*cx_45 + a5*cx_55 + a6*cx_65 + a7*cx_75;
      o6 += a0*cx_06 + a1*cx_16 + a2*cx_26 + a3*cx_36 + a4*cx_46 + a5*cx_56 + a6*cx_66 + a7*cx_76;
      o7 += a0*cx_07 + a1*cx_17 + a2*cx_27 + a3*cx_37 + a4*cx_47 + a5*cx_57 + a6*cx_67 + a7*cx_77;

      outs[n][0]=o0; outs[n][1]=o1; outs[n][2]=o2; outs[n][3]=o3;
      outs[n][4]=o4; outs[n][5]=o5; outs[n][6]=o6; outs[n][7]=o7;
    }

    layer.state_ptr = ptr;
  }

};

} // namespace MicroNAM