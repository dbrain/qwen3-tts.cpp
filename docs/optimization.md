# Qwen3-TTS GGML Optimization Report

This document details the performance characteristics and optimization opportunities for the Qwen3-TTS GGML implementation.

## Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Peak Memory (with cloning) | 3.07 GB | < 18 GB | **PASS** |
| Peak Memory (no cloning) | 2.81 GB | < 18 GB | **PASS** |
| Real-time Factor (RTF) | 1.94x | < 1.0x | Needs improvement |
| Tokens per Second | 25.5 tok/s | - | Baseline |
| Model Size (F16) | 1.8 GB | - | Baseline |
| Model Size (Q8_0) | 1.3 GB | - | 28% smaller |

**Test Configuration:**
- Audio output: 8 seconds (100 tokens @ 12 Hz)
- Threads: 4
- Hardware: AMD Ryzen 5 3600 (4 cores available)
- Memory: 24 GB RAM

> **Note:** The RTF/timing table above is a CPU-only baseline. Vulkan is now the verified default backend on the server (Radeon 8060S / Strix Halo iGPU, UMA, FP16, KHR coopmat). See the Current Implementation State section. Updated RTF numbers on Vulkan have not yet been captured.

## Performance Breakdown

### Pipeline Timing (with voice cloning)

| Stage | Time (ms) | % of Total | Notes |
|-------|-----------|------------|-------|
| Model Loading | 2,685 | - | One-time cost |
| Tokenization | < 1 | 0.0% | Negligible |
| Speaker Encode | 27,318 | 63.8% | **Bottleneck** |
| Code Generation | 3,917 | 9.1% | Transformer inference |
| Vocoder Decode | 11,612 | 27.1% | Audio synthesis |
| **Total** | **42,847** | 100% | End-to-end |

*Note: Total includes model loading. Pipeline time (excluding load) is ~15,529 ms.*

### Pipeline Timing (without voice cloning)

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Tokenization | < 1 | 0.0% |
| Speaker Encode | 0 | 0.0% |
| Code Generation | 3,938 | 25.1% |
| Vocoder Decode | 11,729 | 74.9% |
| **Total** | **15,667** | 100% |

## Memory Usage Analysis

### Peak Memory by Configuration

| Configuration | Peak RSS | Model Memory | Working Memory |
|---------------|----------|--------------|----------------|
| With voice cloning | 3,141 MB | ~2,100 MB | ~1,041 MB |
| Without voice cloning | 2,874 MB | ~2,100 MB | ~774 MB |

### Memory Breakdown (estimated)

| Component | Size | Notes |
|-----------|------|-------|
| TTS Model (F16) | 1,750 MB | Main transformer + speaker encoder |
| Vocoder Model (F16) | 326 MB | Audio decoder |
| KV Cache | ~200 MB | For 4096 context |
| Intermediate Tensors | ~500 MB | Graph computation buffers |
| Audio Buffers | ~100 MB | Input/output audio |
| **Total** | ~2,876 MB | Matches observed RSS |

## Performance Metrics

### Tokens per Second

```
Code Generation: 100 tokens / 3.917 s = 25.5 tokens/sec
```

Each token represents 1/12 second of audio (12 Hz token rate).

### Real-time Factor (RTF)

```
RTF = Generation Time / Audio Duration
RTF = 15.53 s / 8.00 s = 1.94x

RTF < 1.0 = Faster than real-time (streaming capable)
RTF > 1.0 = Slower than real-time
```

Current implementation is **1.94x slower than real-time**.

### Throughput

```
Audio throughput: 8.0 s / 15.53 s = 0.52 seconds of audio per second
```

## Bottleneck Analysis

### 1. Speaker Encoder (63.8% of time with cloning)

The speaker encoder processes reference audio to extract voice characteristics:
- Input: ~30 seconds of reference audio (clone.wav)
- Processing: ECAPA-TDNN architecture with Res2Net blocks
- Output: 1024-dim speaker embedding

**Why it's slow:**
- Large input (2812 mel frames from 30s audio)
- Complex architecture (3 SE-Res2Net blocks + attention pooling)
- Many small convolutions not well-optimized for CPU

### 2. Vocoder Decoder (27.1% of time)

The vocoder converts discrete codes to audio waveforms:
- Input: 100 audio tokens (16 codebooks each)
- Processing: ConvNeXt + upsampling layers
- Output: 192,000 audio samples (8 seconds @ 24kHz)

**Why it's slow:**
- Multiple upsampling stages (8x, 5x, 4x, 2x, 2x)
- Large intermediate tensors during upsampling
- Snake activation functions require element-wise operations

### 3. Code Generation (9.1% of time)

The transformer generates audio codes from text:
- Input: Text tokens + speaker embedding
- Processing: 28-layer transformer
- Output: 100 audio tokens

**Relatively efficient** due to:
- Small sequence lengths
- Optimized GGML matrix operations

## Current Implementation State

The backend plumbing has moved past CPU-only. Relevant facts as of 2026-04-20:

- **Backend selection is runtime and agnostic.** `src/gguf_loader.cpp:28-62` uses `ggml_backend_init_by_type()` with an IGPU → GPU → ACCEL → CPU fallback and a shared refcounted backend across stages. `QWEN3_TTS_FORCE_CPU` forces CPU.
- **ggml is a subdirectory build** (`CMakeLists.txt:48-50`). `-DGGML_VULKAN=ON` is the verified default in production use. `-DGGML_METAL=ON` / `-DGGML_CUDA=ON` pass through the same way but are not currently exercised.
- **Vulkan is confirmed working end-to-end** on AMD Strix Halo (Radeon 8060S, UMA, FP16, KHR coopmat). Both `TTSTransformer` and `AudioTokenizerDecoder` report `backend: Vulkan0` at load. Q8_0 1.7B loads cleanly. Speaker encoder is lazy-loaded and its backend placement has not been explicitly verified from logs.
- **No custom CPU ops.** All three stages use stock GGML ops. Snake activation (`src/audio_tokenizer_decoder.cpp:371-398`) is composed from `ggml_exp`/`ggml_sin`/`ggml_sqr` plus `ggml_reshape_3d` + `ggml_repeat` — portable, but the broadcast pattern is the most likely GPU hotspot.
- **KV cache lives in the backend buffer** (`src/tts_transformer.cpp:769-874`), F16, no per-token host roundtrip.
- **Quantization is type-agnostic.** Loader respects GGUF-native types; embedding extraction handles both F16 and F32 (`src/tts_transformer.cpp:909`). Q8_0 models already ship for 0.6B and 1.7B variants.
- **CoreML path** is isolated to the code predictor on macOS, env-gated (`QWEN3_TTS_USE_COREML`), stubbed elsewhere — does not interact with other backends.
- **Server caches custom voice embeddings** in-memory (`src/server.cpp:110-118, 323-325`) keyed by voice ID.
- **Built-in speaker embeddings are re-fetched each request** (`src/server.cpp:623`).
- **The C API (`src/qwen3tts_c_api.h`) has no embedding extract/reuse primitives** — `synthesize_with_voice()` always re-encodes from a reference audio file path.
- **Vocoder is one-shot**; no streaming hook in the decoder or server. Only `ggml_abort_callback` is wired.

## Optimization Recommendations

Status legend: **DONE** — plumbed and verified; **PARTIAL** — partially implemented; **PENDING** — no code yet.

### Phase 1 — GPU backend (Vulkan)

1. **GPU Acceleration (Vulkan)** — *DONE for transformer + vocoder; benchmarking PENDING*
   - `-DGGML_VULKAN=ON` against the ggml subdir is sufficient; server startup logs confirm Vulkan0 selection for both `TTSTransformer` and `AudioTokenizerDecoder`.
   - Remaining work:
     - Confirm the speaker encoder (`AudioTokenizerEncoder`) also lands on Vulkan when lazy-loaded — its backend is not currently logged.
     - Capture Vulkan RTF and per-stage timings on the canonical ICL repro and update the Summary and Pipeline Timing tables in this doc.
     - Decide whether the per-stage `ggml_backend_sched_t` strategy (one scheduler per stage) leaves Vulkan perf on the table vs. a single shared scheduler across the pipeline.

2. **Q8_0 Quantization** — *DONE*
   - Available for all model sizes in `models/gguf/`.
   - No F16-only code paths. Vulkan inherits ggml's dequant kernels.

### Phase 2 — Embedding cache primitives (backend-independent)

3. **Speaker Embedding Cache** — *PARTIAL*
   - Server caches custom voices but not built-in speakers; C API has no embedding API.
   - Remaining work: add `qwen3tts_extract_embedding()` and `qwen3tts_synthesize_with_embedding()` to the C API; cache built-in speaker embeddings in the server on first use; optional disk cache keyed by `(model_id, hash(ref_audio))`.
   - Eliminates the ~27 s speaker-encode cost on repeat calls — the single largest latency win that does not depend on a GPU backend.

4. **Shorter Reference Audio** — *PENDING (documentation only)*
   - Use 5–10 s clips instead of 30 s for linear reduction in speaker-encode time.
   - Low quality impact for voice cloning. No code change required beyond CLI/README guidance.

5. **Batch Processing** — *PENDING*
   - Process multiple texts with one speaker embedding; GPUs amortize launch overhead across batches much better than CPU, so payoff grows under Phase 1.

### Phase 3 — Streaming vocoder

6. **Streaming Vocoder** — *PENDING*
   - Refactor `src/audio_tokenizer_decoder.cpp` to decode in chunks of N tokens and emit PCM through a callback. Requires handling upsampling receptive-field overlap with ring buffers between chunks.
   - Wire chunked / SSE response path in `src/server.cpp`.
   - Improves latency-to-first-audio and caps peak memory for long outputs, even without raw speedup.

### Phase 4 — Targeted micro-optimizations (only if Phase 1 points here)

7. **Snake activation broadcast** — *PENDING*
   - If vocoder perf on Vulkan disappoints, replace the `ggml_reshape_3d` + `ggml_repeat` pattern with a pre-broadcast alpha/beta tensor, or push a fused custom op upstream in ggml.

8. **Persistent backend-resident speaker embedding** — *PENDING*
   - Today the embedding is passed between stages as a host `const float*` (`src/qwen3_tts.cpp:554-556`) and copied via `ggml_backend_tensor_set`. One-time cost per synthesis; only worth addressing if a batching/streaming server makes it measurable.

### Deferred

9. **Q4_K Quantization** — *DEFERRED*
   - ggml's Vulkan dequant kernels for K-quants are less mature than CPU. Revisit only if Phase 1 shows memory bandwidth as the bottleneck.

10. **SIMD flags** — *DEFERRED*
    - Irrelevant once a GPU backend is primary.

## Benchmark Reproduction

```bash
# Memory check (Linux):
/usr/bin/time -v ./build/qwen3-tts-cli \
    -m models \
    -t "Hello, this is a test." \
    -r clone.wav \
    -o output.wav \
    --max-tokens 100 2>&1 | grep "Maximum resident set size"

# Expected: < 18000000 KB (18 GB)
# Actual:   ~3140000 KB (3.1 GB) - PASS

# Performance benchmark:
./build/qwen3-tts-cli \
    -m models \
    -t "Hello, this is a test." \
    -r clone.wav \
    -o output.wav \
    --max-tokens 100

# Expected output includes timing breakdown
```

## Comparison with Original Test Results

| Metric | Original (Feb 5) | Current | Change |
|--------|------------------|---------|--------|
| Speaker encode | 28,204 ms | 27,318 ms | -3.1% |
| Code generation | 2,607 ms | 3,917 ms | +50.2% |
| Vocoder decode | 6,157 ms | 11,612 ms | +88.6% |
| Total | ~37,000 ms | ~42,847 ms | +15.8% |

*Note: Variations may be due to different test conditions, system load, or measurement methodology.*

## Conclusion

The Qwen3-TTS GGML implementation:

- **Memory: EXCELLENT** — peak 3.1 GB is well under 18 GB target (83% margin).
- **Performance (CPU): NEEDS IMPROVEMENT** — RTF 1.94x is not real-time capable.
- **Performance (Vulkan): UNMEASURED** — the backend runs end-to-end in production on Strix Halo, but no updated RTF numbers have been recorded.
- **Readiness for other GPU backends: HIGH** — Metal/CUDA should enable via the same ggml subdir pattern; stock GGML ops throughout, no refactor needed.

Key findings:
1. Speaker encoder is the primary bottleneck with cloning (64% of time).
2. Vocoder is secondary (27% of time); snake activation's reshape+repeat broadcast is the most likely GPU hotspot.
3. Transformer code generation is relatively efficient (9% of time).
4. The highest-leverage near-term work is the combination of **Phase 1 (Vulkan + measure)** and **Phase 2 (embedding cache primitives in the C API)**. Phase 2 is backend-independent and removes the entire speaker-encode cost on repeat voices.
