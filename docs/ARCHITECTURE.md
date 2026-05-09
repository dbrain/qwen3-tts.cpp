# Architecture & lever stack

The fork-arc journey from upstream's RTF ~0.1 to this fork's RTF 4.29 (Q8) / 4.84 (Q4) on the same RTX 3060. Two phases, ~25 commits.

The talker dominates compute (~14 ms/frame at Q8); the code predictor's 14 sequential AR steps dominate within the talker (~9 ms/frame); the vocoder's cascade convs dominate the remaining ~5 ms.

```
prompt ──► tokenizer ──► talker (28 layers, autoregressive at 12 Hz)
                              │  every frame produces one cb0 token + 14 code-pred steps for cb1..cb15
                              ▼
                          codec frame [16 codebooks @ 12 Hz]
                              │
                              ▼
                          vocoder (cascade of conv_1d_direct + conv_transpose_1d + snake)
                              │
                              ▼
                          PCM @ 24 kHz (V1) or 48 kHz (V2)
```

---

## Phase 1 — vocoder (early forks)

The vocoder was the dominant cost in upstream because the cascade ran F32 with `im2col → cuBLAS gemm` for every conv. Fix: tensor-core kernels + F16 cascade + a fused activation op.

### `wmma` `conv_1d_direct` and `conv_transpose_1d` kernels

Single biggest fork-arc jump (RTF 1.87 → 2.61 on conv_1d_direct, then 2.65 → 3.37 once smem-tiled F16 `conv_transpose_1d` landed). Tensor-core `nvcuda::wmma` GEMM with on-the-fly im2col index calculation, replacing the im2col-temp + cuBLAS path.

**F16 weights only** — do not quantise the vocoder beyond F16, that drops the kernel.

Lives in `dbrain/ggml`:
- `feat(cuda): conv_1d_direct wmma kernel`
- `feat(cuda): conv_transpose_1d smem-tiled wmma kernel`
- `feat(cuda): F16 in/out paths through conv_1d_direct, conv_transpose_1d, snake, acc + ggml_*_to dst-type API`

### F16 cascade activations

Every cascade intermediate (4–5 dec_blocks × `conv_transpose_1d` + 3 residuals each) runs F16 instead of F32. The `wmma` accumulator and snake math stay F32 internally, only dst writes truncate. Halves the scheduler arena: vocoder `sched_cu` 144 → 94 MiB at chunk=30.

Opt out: `QWEN3_TTS_VOCODER_FP16_CASCADE=0`.

### Streaming batch=30, first_batch=1

Cuts cascade buffer size linearly vs the prior batch=60 default (each cascade intermediate is `batch × stride^cascade × channels`), saving another 144 MiB sched_cu at noise-level RTF cost. TTFA unchanged because `first_batch_size=1` is independent of steady-state batch.

### Chunked ICL warmup

Voice-clone warmup decode used to feed the entire `n_ref_frames`-wide ref clip through `stream_decode` in one call, building a single oversized cascade graph that pinned the scheduler arena (e.g. 96-frame ref → 478 MiB sched_cu, frozen for the lifetime of the process). Now chunked at the steady-state batch size, keeping the arena flat at 144 MiB on cloned-voice paths.

### Fused `GGML_OP_SNAKE` op

CPU + CUDA, F32/F16 paths. Replaces the `pow(sin(αx), 2) / α + β·x` broadcast chain used at every vocoder residual. Removes a ~10× tensor-broadcast overhead and a `powf` NaN edge case.

### Q8_0 vocoder mat-mul weights

Load-time quant of pre_tfm attn/FFN/proj + upsample pwconv1/2 + VQ projections. Conv weights stay F16 (wmma kernels demand it). Saves 42 MiB on weights at every budget. Opt out: `QWEN3_TTS_VOCODER_NO_Q8_LOAD=1`.

---

## Phase 2 — talker (megakernel-v0)

Upstream talker was launch-overhead-bound: ggml's generic `mul_mat` dispatcher fires hundreds of tiny kernels per frame on small `(M=1, K∈{1024,2048,3072}, N∈{...})` shapes. Each launch costs ~0.5 µs of fixed overhead — at ~150 launches/frame × 12 fps audio = ~900 µs/sec of pure overhead. Fixing this is what `dbrain/megakernel-v0` does, in 9 sub-versions.

### v1–v3: Phase A specialized MMVQ

Shape-specialized `Q8_0 × Q8_1` mul_mat_vec_q for the 8 unique `(K, N)` pairs in this model — the talker's `(2048, {2048,1024,3072})` and code-pred's `(1024, {1024,2048,3072})` plus `(3072, {1024,2048})`. Kernel structure mirrors ggml's: F32 → Q8_1 row quantize, Q8_0 × Q8_1 dot via `__dp4a` with paired-u16 loads (since `block_q8_0` is 34 B and qs is only 2-byte aligned across blocks), one warp per output row, K compile-time so nvcc fully unrolls.

Hooked in via `ggml_cuda_set_mul_mat_hook` (added to dbrain/ggml). On miss, falls through to ggml's generic path.

Plus a quantize-x cache keyed on the `ggml_tensor*` pointer (not `src1->data`, which ggml's allocator reuses for unrelated logical tensors mid-graph).

### v4: topo fix unblocks QKV fusion

The naive QKV fusion (single launch for Q + K + V mat-muls sharing post-attn-norm `cur`) crashed because ggml's allocator gives V the same slot K just freed (K's rms_norm consumes K before V's mul_mat in topo order, so K's `n_children + n_views` hits zero and the slot recycles into V).

Fix: 3-line reorder in `tts_transformer.cpp` — explicitly expand Q/K/V mul_mats into adjacent topo positions BEFORE any consumer is built. Allocator now gives all three distinct dst slots. Same trick unblocks gate-up fusion. **+0.18 RTF**.

### v5–v6.2: norm-quant fusion, INT8 mma — both ripped

- `fused_rmsnorm_mul_quantize_q8_1<1024>`: rms_norm + mul + quantize-x in one launch. Compiled and wired, but ggml-cuda's `try_fuse({RMS_NORM, MUL})` eats the chain pre-dispatch — op_hook anchor never fires. Counter dump showed `groups=22260 anchor=0 fol=0` over ~600 frames. v5's "+0.024 RTF med" was sampling noise. Ripped.
- INT8 tensor-core `mma m16n8k32` on the qwen3-tts hot wide-N shapes: 0.31–0.70× dp4a on RTX 3060. Reason: GA106's INT8 mma peak ≈ dp4a peak (the famous "16×" was A100); M=1 matvec utilization is 1/8; dp4a is already at 75–100% HBM peak so the bottleneck is bandwidth not compute. Ripped. Microbench preserved at `tools/mmvq_int8_bench.cu`.

### v7: GPU-side sampling + full-AR cgraph

Phase 1 — replace host softmax + `std::discrete_distribution` with an in-cgraph Gumbel-max chain (no new CUDA kernels — all existing ggml-cuda primitives: `argmax`/`top_k`/`get_rows`/`scale`/`add`). For T<=0 collapses to a single `ggml_argmax(logits)`.

Phase 2 — build prefill + 14 AR steps as **one cgraph**. Each step's sampled-token tensor feeds the next step's embed lookup as a graph-level data dep, so ggml-cuda's per-compute graph capture covers the entire AR loop in one capture instead of 15 separate captures + host roundtrips. **`cudaGraphExecUpdate` reuses ~150 launches** per frame.

**+0.247 RTF** measured. 2.5× the forecast — the win wasn't host-overhead recovery, it was the CUDA graph capture amortization.

### v8: defaults flip

Megakernel install gate flipped from opt-IN to opt-OUT. Combined with already-default `QWEN3_TTS_KV_Q8` + ggml-cuda's default-on graphs, a bare `qwen3-tts-server` ships the full v7 fast path with no env vars. Bench: RTF avg 3.822 / med 3.859 / max 3.947.

### v9.1–v9.2: async vocoder

Vocoder dispatched on a worker thread on a **dedicated** ggml-cuda backend (own context + streams) so its ~150 ms/batch work pipelines behind the talker AR loop's host work instead of blocking on the same scheduler.

Two crashes had to be fixed first:
1. `g_fusion_plan` (process-global `unordered_map`) was written by `graph_begin_hook` and read by `mul_mat`/`op` hooks — async vocoder's hook calls raced the talker's writes. Fix: gate hooks on `g_talker_ctx` (atomic latch on the talker backend's CUDA context, identified by its hidden-shape catalog in graph_begin).
2. `g_staging` (process-global GPU buffer) was written/read across two CUDA streams asynchronously. Cross-stream aliasing. Same `g_talker_ctx` gate fixes it.

Per-frame "Other/overhead" (sync vocoder wait, 4.7 ms/frame on v8) drops to 0.04 ms/frame. **+0.119 RTF avg** at default-on.

Plus a `SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL` crash handler that emits a backtrace before re-raising — caught the original SIGSEGV stack that pinned bug #1. Unbuffered stderr/stdout for the same reason.

### v9.3: talker = HIGH stream priority

Pin the talker backend's CUDA streams to HIGH priority (greatest negative in `cudaDeviceGetStreamPriorityRange`) so the talker preempts the dedicated vocoder backend (default-priority streams) for SM time when both compute concurrently.

**HIGH on the talker, not LOW on the vocoder**: `cudaDeviceGetStreamPriorityRange` returns `[greatest=-5, least=0]` on RTX 3060 — the "least" priority value (0) is identical to default-create priority. There is no priority slot below default to demote the vocoder into. The only differentiation on this device is to bump the talker UP. A first lap that set vocoder→LOW measured +0.0016 RTF (literal no-op). **+0.046 RTF paired**, breaks RTF avg 4.0.

### v9.4: full-AR cgraph cache

Cache the v7 full-AR cgraph + alloc state across frames so it doesn't rebuild every frame. Naive `ggml_backend_sched`-based caching crashed on frame 2: `sched_split_graph` frees + re-inits `sched->ctx` every `alloc_graph`, recreating split-copy tensors at recycled offsets that alias the cached graph's already-allocated tensors.

Fix avoids forking ggml-alloc entirely. The full-AR cgraph runs exclusively on `state_.backend` (no CPU fallback nodes), so the cached path is driven directly:
- `state_.cp_galloc = ggml_gallocr_new(buft)` — one-backend gallocr, allocated once at init, kept across frames.
- `ggml_gallocr_alloc_graph(cp_galloc, gf)` + `ggml_backend_graph_compute(state_.backend, gf)`.

Bypassing sched skips the ctx-free-and-re-init that breaks naive caching. **+0.20–0.25 RTF paired.** Default-on; opt out via `QWEN3_TTS_NO_FULL_AR_CACHE=1`.

### v9.5: GPU-side talker sampling

Per-frame talker cb0 sampling (suppress mask + repetition penalty + top-k + softmax + `discrete_distribution`) moves from host into the cgraph as a sign-aware HF rep penalty + suppress mask + top-K + Gumbel-max chain. Eliminates the per-frame logits roundtrip; only `last_hidden_` (~8 KB) crosses the host boundary.

Default-on; opt out via `QWEN3_TTS_NO_TALKER_GPU_SAMPLE=1`.

### v9.6: ggml Q4_K `get_rows` kernel

dbrain/ggml carries the missing `Q4_K` `get_rows` CUDA kernel (was F16-only upstream). Unblocks Q4_K_M models on the v9.4 full-AR cgraph cache path — codec embed lookup is `get_rows(codec_embd, sampled_cb0)` and Q4 storage means without this kernel, the cached path falls back to a slower CPU fetch.

### v9.7: talker step cgraph cache

Same bypass-sched pattern as v9.4, applied to the talker step graph. Single-entry cgraph cache keyed by `kv_n_eff` bucket, dropped+rebuilt on bucket transition. **Single-entry rather than per-bucket map** because `ggml_gallocr` can't safely cycle distinct cgraphs on one gallocr — `init_tensor` short-circuits when `tensor->data != NULL` (`ggml-alloc.c:983-993`), so a later cgraph's buffer-grow leaves prior cgraphs with stale device pointers. Per-bucket gallocrs would solve it but bloat activation arena 250–400 MiB. Within `generate()` `kv_n_eff` is monotonic so a single entry visits each bucket exactly once; rebuild cost amortizes to ~4 µs/frame.

**+0.061 RTF paired** (n=74×2, CI [+0.054, +0.068]). Default-on; opt out via `QWEN3_TTS_NO_TALKER_STEP_CACHE=1`.

### Streaming KV slab + `flash_attn_ext`

Replaces the prior per-call `ggml_concat(past, current)` rebuild with one F16 slab written via `ggml_set_rows`, lazily grown in doubling steps to the synth's `max_audio_tokens` budget. Default-budget paragraphs cap at ~64 MiB slab, max-budget at 256 MiB, vs the prior unbounded growth that pushed the CUDA pool past 4 GB on long inputs. `QWEN3_TTS_STREAM_KV_KEEP=1` forces "always keep at high-water"; default shrinks between synths when next budget ≤ ½ current.

### Talker Q8_0 KV cache

Default-on with the FA prefill path. Saves ~64 MiB peak VRAM at no measurable RTFA cost. Opt out: `QWEN3_TTS_KV_Q8=0`.

### Lazy talker KV growth

Initial alloc = `prefill + 1024 + 8`, doubles in 2048-frame steps capped at the synth's `max_audio_tokens + 8`. Default-budget paragraphs sit at ~60 MiB talker KV vs the prior 120 MiB always-alloc-to-cap. `QWEN3_TTS_TALKER_INITIAL_AUDIO=N` overrides.

### Voice-keyed prefill ICL cache + per-voice prefill K/V state cache

Cold-from-disk TTFA <300 ms on the second synth of the same voice. Combined with persistent `voice.warmup` snapshots: cached cold-start synths land in 35–70 ms TTFA from a fresh process boot (the snapshot restores the vocoder ICL state in 5–7 ms instead of running a 700–1200 ms warmup decode).

---

## Worker isolation

`/v1/admin/unload` in single-process mode calls `Qwen3TTS::unload_model()` which frees weight buffers, KV slabs, gallocrs, and the ggml-cuda backend. The VMM pool destructor calls `cuMemUnmap` / `cuMemAddressFree` so physical VRAM is released. But the CUDA primary context itself — cuBLAS / cuBLASLt workspaces, compiled cubin/PTX cache (one slab per built arch), driver runtime state — only goes away when the process exits. That's the ~370 MiB residual that survives `/unload`.

`QWEN3_TTS_WORKER_ISOLATION=1` opts into a parent/child split:

```
parent (qwen3-tts-server)              worker (qwen3-tts-server --worker N)
─────────────────────────              ────────────────────────────────────
HTTP server (httplib)
voice archive on disk
worker manager
  ├─ socketpair() ─◄────── socket ─────► reads SYNTH_REQ etc.
  ├─ fork() ───────────────────────────► owns Qwen3TTS + ggml-cuda
  └─ /unload → SIGKILL + waitpid         on EOF or SHUTDOWN: exits
                                         PR_SET_PDEATHSIG: kernel kills
                                         the worker if parent dies
```

The parent never touches CUDA, so it never appears in `nvidia-smi`. `/v1/admin/unload` and the idle-unload watchdog both SIGKILL the worker; kernel exit destroys the CUDA primary context; **all VRAM is reclaimed**.

### IPC protocol

Length-prefixed frames over a Unix-domain `socketpair`:

```
[u32 frame_type][u32 payload_len][u32 req_id][u8 payload[payload_len]]
```

| frame | dir | payload |
|---|---|---|
| `HELLO` | W→P | `{pid, role}` |
| `LOAD_REQ` / `LOAD_RESP` | both | `{model, vocoder, speaker_encoder, voice_archive_dir, model_id, lazy_load}` → `{ok, error, sample_rate, loaded_warmups}` |
| `SYNTH_REQ` | P→W | json `{text, params, embedding_size, n_ref_codes, n_ref_frames, stream_batch_size}` + raw f32 embedding + raw i32 ref_codes |
| `SYNTH_RESP` | W→P | non-streaming: full audio in one frame (json meta + raw f32) |
| `AUDIO_FRAME` | W→P | streaming: per-chunk PCM (json header + raw f32) |
| `SYNTH_DONE` | W→P | streaming: end-of-stream metadata (cache keys, sample counts) |
| `EXTRACT_EMBED_REQ` / `RESP` | both | voice rego: filepath → 1024-float embedding |
| `ENCODE_CODES_REQ` / `RESP` | both | voice rego: f32 samples → i32 ref_codes + n_frames |
| `SAVE_WARMUP_REQ` / `RESP` | both | per-synth warmup-blob persistence (worker writes to shared volume) |

`send_frame` coalesces header + payload via `writev` so a single small frame is one kernel send. The standalone POC (in `src/test_worker_ipc.cpp`) measured p99 RTT of 7–24 µs depending on payload (16 KB AUDIO_FRAME = 12 µs p99). At ~75 frames per 6 s synth, IPC overhead amortises to ~0.84 ms vs ~1500 ms wall — sub-noise vs single-process.

### What the worker owns vs what the parent owns

- Parent owns the **voice archive** (filesystem). On voice register the parent decodes the upload, writes the tmpfile, and forwards the path / samples to the worker via `EXTRACT_EMBED_REQ` / `ENCODE_CODES_REQ`. Resulting embedding + ref_codes get persisted to `voice.bundle` by the parent for cross-restart durability.
- Worker owns the **prefill + ICL warmup caches** in process-local memory. After every successful synth with a non-zero `prefill_cache_key` the parent fires `SAVE_WARMUP_REQ` so the worker can dump `voice.warmup` to the archive volume. On every fresh worker spawn the worker scans `voice_archive_dir` and reloads any `*.warmup` blobs — that's how warmup hits survive `/unload` and idle-unload across worker lifetimes.

### Failure semantics

| event | result |
|---|---|
| Worker crashes mid-synth | EOF on socket → parent marks dead, returns 500, next request respawns |
| HTTP client disconnects mid-stream | parent's `on_pcm` returns false; parent keeps draining frames so worker doesn't back-pressure on a half-consumed stream |
| Parent SIGKILL'd | kernel sends `SIGTERM` to the worker via `PR_SET_PDEATHSIG`; container restart policy revives the parent |
| OOM in worker | OOM-kill same as crash |

### Cycle-test

Verified on RTX 3060 / Q8 talker / V1 24 kHz vocoder:

| operation | parent VRAM | worker VRAM | total qwen3-tts.cpp share |
|---|---:|---:|---:|
| startup (lazy mode) | 0 | – not spawned | **0 MiB** |
| first synth | 0 | 2600 MiB | 2600 MiB |
| 30 s idle → idle-unload | 0 | killed | **0 MiB** |
| `/v1/admin/unload` | 0 | killed | **0 MiB** (synchronous; ~80–100 ms wall) |
| respawn after `/unload` | 0 | 2600 MiB | first synth: `voice.warmup` cache HIT from disk |
| `kill -9` parent | 0 | killed via PDEATHSIG | container restarts cleanly |

Warm-state RTF is 4.25–4.41×, matching the in-process baseline of 4.17× within sampling noise (40-call concurrent stress run had no zombies or leaked VRAM). The IPC POC is in `src/test_worker_ipc.cpp`; the parent-side handle in `src/worker_session.{h,cpp}`; the wire format in `src/worker_ipc.{h,cpp}`.

---

## Dead ends

Things that were tried, measured, and ripped out — kept here so future agents don't re-derive.

### Stage 1 + 3 of `#3b` — N-step talker window

Attempted to merge N talker steps + N code-pred lanes (in shared KV cache, lane m at slots `[m*16, m*16+15]`) + N-1 embed-folds all in one cgraph. ~1100 LoC, compiled clean, audio held at N=2.

**Perf regressed monotonically with N.** The merge adds a new cost: shared code-pred KV cache `n_ctx` grows to `N*16`, and FA scans the FULL cache per AR step (linear in N). v9.7's step cgraph cache and v9.4's full-AR cgraph cache already extract the launch-overhead win on talker AND code-pred independently, so the merge doesn't add a fundamentally new optimization on top — it just trades two cached compute calls for one bigger one with worse FA scan economics.

| config | RTF med (Q8, RTX 3060) | vs v9.7 |
|---|---:|---:|
| v9.7 baseline | 4.29 | — |
| Stage 3 N=2 | 4.10 | -0.19 |
| Stage 3 N=4 | 3.79 | -0.50 |

Salvage path that *might* work but hasn't been measured: per-lane independent code-pred KV caches instead of the shared `N*16` + cross-lane mask. Each lane's FA scans 16 slots like v9.4. Eliminates the t/s tax. Bigger refactor. Untried.

### Cooperative-launch fused per-layer megakernel

Sub-noise on RTX 3060. Launch-overhead ceiling at this point is ~0.06 RTF total cap; per-layer collapse extracts a fraction of that.

### Norm-quant + A2 fusion

`try_fuse({RMS_NORM, MUL})` eats the chain in ggml-cuda before any op_hook fires. Both fusions deleted. Even with a `pre_fuse_hook`, the launch-overhead ceiling is sub-noise.

### Vocoder `batch_size` 60

Tested earlier (perf-15-era), regressed. 30 stays.

### Skip `sched_reset` on cached path

v8 hypothesis for what was breaking the code-pred cgraph cache. Wrong — root cause is gallocr offset reuse, not sched-reset state loss. v9.4 (gallocr direct path, bypass sched) is the real fix.

### CUDA graphs at batch=1

`GGML_CUDA_DISABLE_GRAPHS=1` is the **safe default** for naive workloads — at batch=1 autoregressive decode, capture costs +~700 MiB peak with no measurable gain because per-token cgraph rebuilds prevent cudaGraphExecUpdate from ever converging on a stable key. v7's full-AR cgraph wraps 14 of those steps into one re-usable graph specifically to recover the capture win, so within v7+ workloads `GGML_CUDA_DISABLE_GRAPHS=0` is correct; outside that path, leave it disabled.

### Q5_K_M

Slower than Q4_K_M on Ampere via ggml MMQ. Practical quant choice is Q4_K_M or Q8_0 only.

### F16 src1 for Q8 dot products (parakeet/siglip class lever)

ggml-cuda MMQ/MMVQ assert F32 src1; `ggml_cast(activations, F16)` on Q4_K weights forces dequant→cuBLAS-half fallback, net regression. Don't propose F16 src1 against quant src0 until ggml ships an F16-consuming K-quant kernel.

---

## Required ggml kernels

The submodule pointer in `.gitmodules` references [`dbrain/ggml`](https://github.com/dbrain/ggml). 13 commits ahead of `ggml-org/ggml@master`. Most are Qwen3-TTS-specific layouts that don't merge upstream as-is, plus general infrastructure for multi-backend latency-sensitive workloads.

| Op / area | What |
|---|---|
| `GGML_OP_SNAKE` (CPU + CUDA) | fused `α·sin(βx)² + γx` for the vocoder activation |
| `GGML_OP_CONV_1D_DIRECT` | smem-tiled CUDA kernel + tensor-core wmma variant (F16 weights) |
| `conv_transpose_1d` wmma | smem-tiled F16 kernel (RTF 2.65 → 3.37 alone) |
| F16 cascade plumbing | F16 in/out paths + `ggml_*_to` dst-type API for `conv_1d_direct`, `conv_transpose_1d`, `snake`, `acc` |
| `concat` F16 + I32 paths | (was F32-only via assert) |
| Q4_K `get_rows` CUDA | unblocks Q4_K_M codec embed lookup |
| `mul_mat` dispatcher hook | lets the megakernel install shape-specialized Q8 MMVQ at runtime |
| `graph_compute_begin` hook | per-cgraph plan rebuild trigger; receives the cgraph |
| Generic per-op hook | norm-quant / fusion routing (currently unused — the fusions that needed it didn't pay) |
| Stream priority knob | `ggml_backend_cuda_init_with_priority` + `params="priority=low\|high"` parsing |
| I32 row_indices for ROPE+VIEW+SET_ROWS fusion | unblocks talker streaming KV writes |
| Sched fixes | `hash_set` 2× graph_size; `reserve_n` failure check; null-check tensor-copy malloc |
| Kernel launch error checks | `conv_1d_direct`, `snake`, `acc` |
| `supports_op` tightening | for `CONV_1D_DIRECT`, `SNAKE`, `ACC` |

## Further reading

- [docs/optimization.md](optimization.md) — earlier optimization analysis, predates megakernel-v0.
- [docs/streaming_design.md](streaming_design.md) — vocoder streaming design notes.
- [docs/tensor_mapping.md](tensor_mapping.md) — HuggingFace ↔ GGUF tensor name mapping.
- [docs/BENCHMARKS.md](BENCHMARKS.md) — measured numbers + methodology.
- [docs/API.md](API.md) — HTTP API reference.
