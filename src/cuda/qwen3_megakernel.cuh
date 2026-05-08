// Qwen3-TTS megakernel — Phase A specialized MMVQ + Phase B layer megakernel.
// See kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md for the full plan.
//
// Phase A: 7 shape-specialized Q8_0 MMVQ kernels for known (K, N) pairs.
// Phase B: per-layer megakernel that calls Phase A's matmul as device functions.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace qwen3_megakernel {

// ----------------------------------------------------------------------------
// Model shape constants (Qwen3-TTS 0.6B, talker)
// ----------------------------------------------------------------------------

constexpr int HIDDEN              = 1024;
constexpr int INTERMEDIATE        = 3072;
constexpr int N_HEADS             = 16;
constexpr int N_KV_HEADS          = 8;
constexpr int HEAD_DIM            = 128;
constexpr int N_ATTN_OUT          = N_HEADS * HEAD_DIM;       // 2048
constexpr int N_KV_OUT            = N_KV_HEADS * HEAD_DIM;    // 1024
constexpr int N_TALKER_LAYERS     = 28;
constexpr int N_CODE_PRED_LAYERS  = 5;
constexpr int CODE_PRED_VOCAB     = 2048;

// Q8_0 block layout (matches ggml's block_q8_0):
// 32 quantized int8 values + 1 fp16 scale per block.
struct block_q8_0 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 layout mismatch with ggml");

// ----------------------------------------------------------------------------
// Phase A — Shape-specialized Q8_0 MMVQ
// ----------------------------------------------------------------------------
//
// Computes y = W * x for M=1, where W is Q8_0 [K, N] and x is F16/F32 [K].
// K and N are compile-time constants per instantiation. nvcc fully unrolls
// the K-loop and chooses register tile size based on the known shape.
//
// Caller convention: launch with grid configured to cover N output columns,
// pass device pointers, run on the supplied stream. Falls back to ggml
// MMVQ if the shape isn't one we instantiated.

template <int K, int N>
__global__ void mmvq_q8_0_f32_specialized(
    float * __restrict__ y,            // [N]
    const block_q8_0 * __restrict__ w, // [K/32, N] (column-major Q8_0 blocks)
    const float * __restrict__ x);     // [K]

// Launch wrappers — one per (K, N) pair we ship.
// Each wrapper picks block/grid config tuned for the shape, then launches
// the templated kernel above.

void launch_mmvq_q8_0_K1024_N2048(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream);
void launch_mmvq_q8_0_K1024_N1024(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream);
void launch_mmvq_q8_0_K2048_N1024(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream);
void launch_mmvq_q8_0_K1024_N3072(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream);
void launch_mmvq_q8_0_K3072_N1024(float * y, const block_q8_0 * w,
                                   const float * x, cudaStream_t stream);

// Codec / code-pred head shapes — N is data-dependent so dispatch picks the
// nearest specialized variant or falls back to ggml at runtime.
void launch_mmvq_q8_0_K1024_codec_head(float * y, const block_q8_0 * w,
                                        const float * x, int N,
                                        cudaStream_t stream);

// ----------------------------------------------------------------------------
// Phase B — Per-layer megakernel for talker AR step
// ----------------------------------------------------------------------------
//
// Computes one full transformer layer at M=1 in a single CUDA kernel.
// All sub-ops (norm / qkv / rope / FA / out_proj / ffn / residual) live in
// one kernel; cooperative_groups::grid_group::sync between phases.
//
// NOT IMPLEMENTED in v0 — skeleton only. The first agent to take this on
// should answer the open questions in HANDOFF-megakernel-v0.md before
// writing the body.

void launch_talker_layer_q8_0(
    float * residual_out,              // [HIDDEN]
    const float * residual_in,         // [HIDDEN]
    const __half * attn_norm_w,        // [HIDDEN]
    const __half * ffn_norm_w,         // [HIDDEN]
    const __half * q_norm_w,           // [HEAD_DIM] or nullptr
    const __half * k_norm_w,           // [HEAD_DIM] or nullptr
    const block_q8_0 * wq,             // [HIDDEN, N_ATTN_OUT]
    const block_q8_0 * wk,             // [HIDDEN, N_KV_OUT]
    const block_q8_0 * wv,             // [HIDDEN, N_KV_OUT]
    const block_q8_0 * wo,             // [N_ATTN_OUT, HIDDEN]
    const block_q8_0 * wgate,          // [HIDDEN, INTERMEDIATE]
    const block_q8_0 * wup,            // [HIDDEN, INTERMEDIATE]
    const block_q8_0 * wdown,          // [INTERMEDIATE, HIDDEN]
    __half * k_cache,                  // [n_ctx, N_KV_HEADS, HEAD_DIM]
    __half * v_cache,                  // [n_ctx, N_KV_HEADS, HEAD_DIM]
    int kv_n_past,
    int kv_n_eff,
    float rope_theta,
    float kq_scale,
    cudaStream_t stream);

}  // namespace qwen3_megakernel
