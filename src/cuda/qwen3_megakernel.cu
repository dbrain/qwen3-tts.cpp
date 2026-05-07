// Qwen3-TTS megakernel — Phase A skeleton.
// See HANDOFF-megakernel-v0.md for the full plan.
//
// Status: SKELETON ONLY. The K=1024 N=2048 mmvq below is illustrative —
// it shows the template structure and dispatch but is NOT TUNED. The
// next agent should:
//   1. Microbench it vs ggml's mul_mat_vec_q to establish a baseline ratio.
//   2. Tune the BLOCK_N tile size and K-loop unroll factor for each shape.
//   3. Add the remaining 6 specializations.
//   4. Wire dispatch from tts_transformer.cpp behind QWEN3_TTS_SPECIALIZED_MMVQ.

#include "qwen3_megakernel.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace qwen3_megakernel {

// ----------------------------------------------------------------------------
// Phase A — illustrative specialization for (K=1024, N=2048)
// ----------------------------------------------------------------------------

template <int K, int N>
__global__ void mmvq_q8_0_f32_specialized(
    float * __restrict__ y,
    const block_q8_0 * __restrict__ w,
    const float * __restrict__ x) {

    constexpr int K_BLOCKS = K / 32;       // Q8_0 blocks per K row
    static_assert(K % 32 == 0, "K must be multiple of Q8_0 block size");

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Each thread accumulates one output column.
    // K-loop is fully unrolled by nvcc when K is a compile-time constant.
    // TODO(phaseA): the per-thread-per-output pattern is naive. Real wins
    // come from cooperative loading of x[] into smem (shared across threads)
    // and warp-level dot-product reduction. See ggml-cuda/mmvq.cu for the
    // existing pattern and aim to match its register tile size.
    float acc = 0.0f;
    #pragma unroll
    for (int kb = 0; kb < K_BLOCKS; ++kb) {
        const block_q8_0 & block = w[kb * N + n];
        const float scale = __half2float(block.d);
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int k_idx = kb * 32 + j;
            acc += x[k_idx] * (float)block.qs[j] * scale;
        }
    }
    y[n] = acc;
}

// Explicit instantiations for the shapes we ship in Phase A.
template __global__ void mmvq_q8_0_f32_specialized<1024, 2048>(
    float *, const block_q8_0 *, const float *);
template __global__ void mmvq_q8_0_f32_specialized<1024, 1024>(
    float *, const block_q8_0 *, const float *);
template __global__ void mmvq_q8_0_f32_specialized<2048, 1024>(
    float *, const block_q8_0 *, const float *);
template __global__ void mmvq_q8_0_f32_specialized<1024, 3072>(
    float *, const block_q8_0 *, const float *);
template __global__ void mmvq_q8_0_f32_specialized<3072, 1024>(
    float *, const block_q8_0 *, const float *);

// ----------------------------------------------------------------------------
// Launch wrappers
// ----------------------------------------------------------------------------

void launch_mmvq_q8_0_K1024_N2048(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream) {
    constexpr int N = 2048;
    constexpr int BLOCK = 256;
    const int grid = (N + BLOCK - 1) / BLOCK;
    mmvq_q8_0_f32_specialized<1024, N><<<grid, BLOCK, 0, stream>>>(y, w, x);
}

void launch_mmvq_q8_0_K1024_N1024(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream) {
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    const int grid = (N + BLOCK - 1) / BLOCK;
    mmvq_q8_0_f32_specialized<1024, N><<<grid, BLOCK, 0, stream>>>(y, w, x);
}

void launch_mmvq_q8_0_K2048_N1024(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream) {
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    const int grid = (N + BLOCK - 1) / BLOCK;
    mmvq_q8_0_f32_specialized<2048, N><<<grid, BLOCK, 0, stream>>>(y, w, x);
}

void launch_mmvq_q8_0_K1024_N3072(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream) {
    constexpr int N = 3072;
    constexpr int BLOCK = 256;
    const int grid = (N + BLOCK - 1) / BLOCK;
    mmvq_q8_0_f32_specialized<1024, N><<<grid, BLOCK, 0, stream>>>(y, w, x);
}

void launch_mmvq_q8_0_K3072_N1024(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream) {
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    const int grid = (N + BLOCK - 1) / BLOCK;
    mmvq_q8_0_f32_specialized<3072, N><<<grid, BLOCK, 0, stream>>>(y, w, x);
}

void launch_mmvq_q8_0_K1024_codec_head(float * y, const block_q8_0 * w,
                                        const float * x, int N,
                                        cudaStream_t stream) {
    // N is runtime here (codec_vocab_size varies per model variant).
    // For the common N values, dispatch to a specialized instantiation.
    // For others, the next agent should either add the instantiation or
    // fall back to ggml's MMVQ — caller decides.
    (void) y; (void) w; (void) x; (void) N; (void) stream;
    // TODO(phaseA): implement runtime-N dispatch or document the fallback.
}

// ----------------------------------------------------------------------------
// Phase B — Per-layer megakernel (NOT IMPLEMENTED)
// ----------------------------------------------------------------------------

void launch_talker_layer_q8_0(
    float * residual_out,
    const float * residual_in,
    const __half * attn_norm_w,
    const __half * ffn_norm_w,
    const __half * q_norm_w,
    const __half * k_norm_w,
    const block_q8_0 * wq,
    const block_q8_0 * wk,
    const block_q8_0 * wv,
    const block_q8_0 * wo,
    const block_q8_0 * wgate,
    const block_q8_0 * wup,
    const block_q8_0 * wdown,
    __half * k_cache,
    __half * v_cache,
    int kv_n_past,
    int kv_n_eff,
    float rope_theta,
    float kq_scale,
    cudaStream_t stream) {

    // TODO(phaseB): see HANDOFF-megakernel-v0.md "Phase B" section for the
    // full design (cooperative groups vs split sub-megakernels, smem layout,
    // grid sync points). DO NOT START until Phase A is shipped and validated.
    (void) residual_out; (void) residual_in;
    (void) attn_norm_w; (void) ffn_norm_w; (void) q_norm_w; (void) k_norm_w;
    (void) wq; (void) wk; (void) wv; (void) wo;
    (void) wgate; (void) wup; (void) wdown;
    (void) k_cache; (void) v_cache;
    (void) kv_n_past; (void) kv_n_eff;
    (void) rope_theta; (void) kq_scale; (void) stream;
}

}  // namespace qwen3_megakernel
