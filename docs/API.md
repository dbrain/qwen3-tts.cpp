# HTTP API reference

OpenAI-compatible endpoints with this fork's extras layered on top. Per-request body always overrides defaults; defaults are tuned for single-stream decode on a 12 GB Ampere.

## `POST /v1/audio/speech`

Main TTS endpoint. Accepts JSON body, returns audio (one-shot) or a streaming response.

### Body

| Parameter | Origin | Notes |
|---|---|---|
| `input` | OpenAI | required; default cap 6144 chars (env `QWEN3_TTS_MAX_INPUT_CHARS`) |
| `voice` | OpenAI | `"default"`, a built-in speaker, or a registered clone id; empty = `"default"` |
| `instructions` | OpenAI | VoiceDesign description; works alongside `voice="default"` (no clone needed) |
| `response_format` | OpenAI (subset) | `wav`, `pcm`, `mp3`, `ogg` (Opus-in-Ogg), `aac` (ADTS). FLAC is not implemented. Compressed formats (mp3/ogg/aac) honour `bitrate_kbps` (32–192, default 64). |
| `stream_format` | OpenAI | `audio` (chunked binary) or `sse` (`speech.audio.delta` events with base64 audio); empty = one-shot |
| `model`, `speed` | OpenAI | accepted in the body but currently ignored |
| `language` | this fork | language code (`en`, `zh`, ...); see `GET /v1/audio/languages` |
| `temperature`, `top_k`, `top_p`, `seed`, `repetition_penalty` | this fork | sampling controls |
| `max_audio_tokens` | this fork | 1..8192, default 2048; KV `n_ctx` scales linearly (linear t/s tax) |
| `stream_batch_size` | this fork | vocoder chunk size, default 30; smaller = lower per-chunk sched_cu, similar TTFA |
| `stream_first_batch_size` | this fork | first chunk only, default 1 = lowest TTFA |
| `align` | this fork | `true` enables forced alignment. Requires `stream_format="sse"` + worker isolation + `--hf-repo-fa` at server start. Delivery mode set by `align_stream`. |
| `align_stream` | this fork | `"final-only"` (default) emits one `speech.audio.alignment.final` event after the audio stream completes. `"partial"` additionally emits `speech.audio.alignment.partial` events *interleaved* with `speech.audio.delta` as audio is produced — each partial re-emits the full word list with timings refreshed against the audio seen so far; the final event has locked timings. Partial mode runs the aligner in a sibling subprocess concurrent with synth — adds ~150 ms end-to-end vs `align=off` (vs ~700 ms for the legacy `align=true`+`final-only` path). Ignored when `align=false`. |

### Streaming-mode matrix (`response_format` × `stream_format`)

| `response_format` | `stream_format` | Wire format |
|---|---|---|
| `wav` | `audio` | `audio/wav`, chunked: 44-byte placeholder header (size fields = `0xFFFFFFFF`) emitted with first PCM batch, then 16-bit-LE PCM chunks. Most players accept the placeholder; `wave.open()` rejects it — fix by rewriting bytes 4 and 40 once the stream lands. |
| `pcm` | `audio` | `audio/pcm`, chunked: raw 16-bit-LE PCM, no header. Caller must know the sample rate (24 kHz V1, 48 kHz V2). |
| `mp3` | `audio` | `audio/mpeg`, chunked: self-syncing MP3 frames; no leading header needed. |
| `ogg` | `audio` | `audio/ogg`, chunked: standalone Ogg Opus pages. |
| `aac` | `audio` | `audio/aac`, chunked: ADTS-framed AAC. Default `stream_first_batch_size` for AAC is 4 (encoder buffers ≥1 frame internally; smaller values gain little TTFA). |
| `wav` \| `pcm` | `sse` | `text/event-stream`: `speech.audio.delta` events with `{type, audio: <base64>}` data, then a closing `speech.audio.done` event carrying usage/timing. |
| any | empty | one-shot: full body in a single response, no chunked transfer encoding. |

### TTFA-accurate Python clients

Python's `urllib` `HTTPResponse` uses an internal buffered reader. `read()` and `readline()` block waiting for the buffer to fill, which inflates TTFA by hundreds of ms over a streaming SSE response. **Use `read1(N)`** — returns whatever bytes are currently available without trying to top up the buffer — and parse SSE events from the raw stream by hand:

```python
read_chunk = getattr(resp, "read1", None) or (lambda n: resp.read(n))
buf = b""
while chunk := read_chunk(4096):
    buf += chunk
    while b"\n\n" in buf:
        evt_bytes, buf = buf.split(b"\n\n", 1)
        # parse "event: ..." and "data: ..." lines, decode base64 PCM
```

Reference implementation: [`docker/tts-qwen3-dev/breakit_seeds.py`](#) `post_speech_pcm()`. `requests` and `aiohttp` have analogous gotchas — verify your client doesn't line-buffer the SSE stream before trusting any TTFA measurement.

### Hot-path TTFA characteristics

With `stream_first_batch_size=1` the first emit is a single 12 Hz codec frame, so once prefill + 1 talker step + 1 vocoder decode complete, PCM is on the wire — independent of total synth length.

| Path | Audio length | TTFA |
|---|---:|---:|
| Default voice / VoiceDesign (no ICL) | 0.6–21 s | **35–47 ms** |
| ICL clone, `voice.warmup` cache HIT | 5–21 s | **51–70 ms** |
| ICL clone, cache MISS (first synth/voice) | 5–6 s | 384–641 ms |

Hot-path TTFA stays flat across audio length — neither doubles for a 17 s synth nor halves for a 0.6 s one — because only the first codec frame has to be ready before bytes flow.

The cache-MISS path pays the warmup decode once per voice on the first synth in a fresh process, then persists the snapshot to `voice.warmup` on disk so subsequent boots restore from disk in single-digit-ms.

### Forced alignment

`qwen3-forced-aligner-0.6b` — Qwen3-ASR architecture with a 5000-class lm_head outputting 80 ms timestamp resolution. The server splits `input` on whitespace; the order in `words[]` matches that split. Word timings are deterministic — argmax over the timestamp logits, no sampling — so repeated runs with the same audio + word list produce bit-identical timings.

#### Setup (one-time, per server)

- Start with `--hf-repo-fa cstr/qwen3-forced-aligner-0.6b-GGUF:Q4_K` (or set `QWEN3_TTS_FA_REPO=cstr/qwen3-forced-aligner-0.6b-GGUF` + `QWEN3_TTS_FA_QUANT=Q4_K` in Docker).
- Worker isolation (`QWEN3_TTS_WORKER_ISOLATION=1`) is mandatory — the aligner runs in a dedicated subprocess.
- `stream_format="sse"` is mandatory — non-SSE callers get a `400`. Alignment is delivered as SSE events, not a separate response field.

#### Two delivery modes

| `align_stream` | When you get timings | Sibling subprocess | Audio-pipeline impact |
|---|---|---|---|
| `"final-only"` (default) | One `speech.audio.alignment.final` event after `speech.audio.delta` stream ends, before `done` | Reuses the synth worker | Adds ~700 ms tail latency before `done` |
| `"partial"` | `speech.audio.alignment.partial` events interleaved with audio.delta as audio is produced, plus a final event at the end | Spawns a separate aligner-only worker (Q4_K aligner) on first request | Adds ~150 ms end-to-end (aligner runs concurrent with synth; the final event short-circuits when audio_seen hasn't grown since the last partial) |

#### Cold load

First request that needs alignment pays:

- Aligner sibling subprocess spawn: ~160 ms
- Q4_K aligner GGUF load + GPU buffer alloc: ~500-700 ms
- First-pass audio encode + body forward: ~500-700 ms

Total first-request alignment overhead: **~1.2 s**. Subsequent partial requests reuse the loaded aligner; per-partial steady-state cost is ~150-500 ms (linear in accumulated PCM length), running concurrent with synth so most of it hides behind audio.delta emissions.

The aligner subprocess unloads alongside the synth worker on `/v1/admin/unload` and the idle-unload watchdog — no separate lifecycle.

#### Aligner VRAM cost

Steady-state GPU resident in the aligner sibling subprocess: **~556 MiB** (RTX 3060, Q4_K aligner, 30 s paragraph). Breakdown:

| region | MiB | notes |
|---|---:|---|
| non-layered LLM weights (embed, lm_head, norms) | 86 | Q4_K |
| audio tower weights | 177 | Q4_K |
| KV cache | 30-70 | grows linearly with audio length |
| audio encoder gallocr + body sched | 35 | fixed-shape |
| LLM body sched (multi-backend) | 47 | growing with prompt length |
| CUDA primary context + cuBLAS workspace | 143 | irreducible per-process |

Plus **~237 MiB of host RAM** for the 28 LLM blk layers — the aligner-only subprocess routes them to CPU by default (`CRISPASR_N_GPU_LAYERS=0` baked) to keep GPU footprint small. CPU body forward costs ~+5 % wallclock end-to-end vs running the LLM body on GPU; the trade saves ~210 MiB of GPU. Set `CRISPASR_N_GPU_LAYERS=28` to revert to GPU body if you have the VRAM and want the time back.

#### One-shot `final-only` mode

```
event: speech.audio.alignment.final
data: {"type":"speech.audio.alignment.final",
       "audio_total_ms": 17280,
       "words": [{"word_index":0,"text":"In","t0_ms":120,"t1_ms":280}, ...],
       "timings": {"align_load_ms": 0,
                   "align_resample_ms": 4,
                   "align_forward_ms": 340,
                   "align_total_ms": 345,
                   "align_n_words": 73,
                   "n_partial_events": 0}}
```

`align_load_ms` is non-zero only on the very first request (lazy load on first use). On error the server emits `speech.audio.alignment.error` carrying `{"error": "..."}` instead, then continues to `done`.

#### Streaming `partial` mode

`align_stream:"partial"` opts into interleaved partial events. Each partial re-emits ALL words; clients should key by `word_index` and overwrite timings on each event. The `final` event carries locked timings.

```
event: speech.audio.delta
data: {"type":"speech.audio.delta","audio":"<base64-pcm>"}

event: speech.audio.alignment.partial
data: {"type":"speech.audio.alignment.partial",
       "audio_seen_ms": 2480,
       "words": [{"word_index":0,"text":"In","t0_ms":120,"t1_ms":280}, ...]}

event: speech.audio.delta
data: {"type":"speech.audio.delta","audio":"<base64-pcm>"}

... more delta + partial events interleaved ...

event: speech.audio.alignment.final
data: {"type":"speech.audio.alignment.final","audio_total_ms":17280,"words":[...],"timings":{...}}

event: speech.audio.done
data: {...}
```

A 17 s paragraph at default streaming defaults typically produces ~9 audio.delta events and ~7 partial events plus the final. First partial arrives within ~600 ms of the first audio.delta (warmed) / ~1.2 s (cold). The final event carries identical timings to the last partial in steady state (the server short-circuits a redundant final pass when no new audio arrived since the last partial — same audio + same word list = same alignment by construction).

## `GET /v1/audio/voices`

List available voices. Returns built-in speakers + every registered clone.

## `POST /v1/audio/voices`

Register a voice clone. Multipart:

| Field | Required | Notes |
|---|---|---|
| `name` | yes | the voice id; must pass `is_safe_voice_name` (alnum, dash, underscore) |
| `audio_sample` | yes | reference WAV (mono int16, any rate the speaker encoder accepts) |
| `ref_text` | optional | ICL transcript matching the audio; enables prompt-conditioned voice cloning |

On success: `voice.bundle` written to `$QWEN3_TTS_VOICE_ARCHIVE_DIR/<name>/`. Re-registering the same name updates the bundle in place. Speaker + codec encoders auto-unload after register (one-shot pieces, no need to keep VRAM resident).

If a `voice.bundle` already exists for this name + current model, the encoder forward passes are skipped — you can re-register a clone in <50 ms.

## `GET /v1/audio/voices/{name}/sample.wav`

Returns the original `ref.wav` written when the clone was registered (mono PCM WAV, the bytes the caller submitted as `audio_sample`). For replay / re-upload / debug only — synthesis runs from the bundle's embedding + ref_codes, not this file. `404` when `QWEN3_TTS_VOICE_ARCHIVE_DIR` is unset or the file isn't on disk (e.g. clone was registered before the archive dir existed, or the file was hand-deleted).

## `GET /v1/audio/voices/{name}/ref_text`

Returns the `ref_text.txt` content. `text/plain; charset=utf-8` by default, or `application/json` (`{"ref_text": "..."}`) when the request carries `Accept: application/json`. `404` when no `ref_text` was provided at register time.

## `DELETE /v1/audio/voices/{name}`

Remove the voice from the in-memory map. **Disk artifacts persist** — `voice.bundle` and `voice.warmup` stay in the archive volume so a re-register or a process restart picks them back up. To truly purge, also `rm -rf` the directory under `$QWEN3_TTS_VOICE_ARCHIVE_DIR/<name>/`.

## Lifecycle endpoints

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","model_loaded":bool}` |
| `GET` | `/v1/models` | OpenAI-compat models list |
| `GET` | `/v1/audio/languages` | language codes the model supports |
| `POST` | `/v1/admin/load` | re-materialize the model from captured paths (no-op if already loaded). In worker-isolation mode this respawns the worker subprocess. |
| `POST` | `/v1/admin/unload` | release all GPU/CPU buffers. **Single-process mode**: frees model weights + KV slabs but the ~370 MiB CUDA-context floor stays resident until process exit. **Worker-isolation mode** (`QWEN3_TTS_WORKER_ISOLATION=1`): SIGKILL + waitpid the worker subprocess; the response only returns once the kernel has reaped it, so VRAM is fully reclaimed by the time the HTTP response sends (~80–100 ms wall). The aligner sibling subprocess (if loaded) is killed in the same call — response carries `aligner_unloaded: true` when the aligner had been resident. Idle-unload (`QWEN3_TTS_IDLE_UNLOAD_SECONDS`) does the same teardown. |

`POST /v1/admin/{load,unload}` give you explicit control over GPU lifecycle, useful for sharing the GPU with intermittent peers. Otherwise idle-unload via `QWEN3_TTS_IDLE_UNLOAD_SECONDS` does it automatically.

## Voice archive layout

`$QWEN3_TTS_VOICE_ARCHIVE_DIR/<name>/` per voice:

| File | Required | Notes |
|---|---|---|
| `voice.bundle` | required | binary `QTVB` magic; speaker embedding + ref_codes + model_id stamp. Atomically written on register; re-validated against the loaded model on startup. |
| `ref_text.txt` | optional | ICL transcript |
| `description.txt` | optional | VoiceDesign description |
| `ref.wav` | optional | kept only for human replay; never read at synth time |
| `voice.warmup` | optional | persistent vocoder ICL warm-up snapshot (causal-conv tails, conv_t overlap buffers, per-layer KV cache) keyed by ref_codes hash. Written after the first synth per voice; restored on subsequent cold-from-disk synths in 5–7 ms instead of running the warmup decode (which would be 700–1200 ms). |

Voice survives across model swaps as long as `model_id` matches. Mismatched bundles are deferred (not deleted) so a model swap doesn't lose voices — when the original model returns, the bundle is picked back up.

## Environment variables

### Lifecycle

| Var | Default | Notes |
|---|---|---|
| `QWEN3_TTS_LAZY_LOAD` | unset | `=1` skip startup model load, defer to first request. Voice-archive scan still runs at startup. |
| `QWEN3_TTS_IDLE_UNLOAD_SECONDS` | unset (off in prod compose; 30 in dev) | release GPU after N seconds of inactivity, lazy-reload on next request |
| `QWEN3_TTS_WORKER_ISOLATION` | unset | `=1` runs the GPU work in a forked subprocess so `/v1/admin/unload` and the idle-unload watchdog can SIGKILL it and reclaim **all** VRAM (no CUDA-context floor residue). Per-call IPC overhead is sub-noise (~16 µs p99 per AUDIO_FRAME). See [ARCHITECTURE.md#worker-isolation](ARCHITECTURE.md#worker-isolation) for the protocol + cycle-test. |
| `QWEN3_TTS_VOICE_ARCHIVE_DIR` | `./voice-archive` | where bundles + warmup snapshots persist |

### Quality / sizing

| Var | Default | Notes |
|---|---|---|
| `QWEN3_TTS_KV_Q8` | `1` | Q8_0 KV cache for the talker, ~64 MiB peak savings, no measurable RTFA cost. `=0` reverts to F16 KV. |
| `QWEN3_TTS_TALKER_INITIAL_AUDIO` | `1024` | initial talker KV alloc (frames) |
| `QWEN3_TTS_STREAM_KV_KEEP` | unset | `=1` keep streaming KV slab at high-water across synths; default shrinks if next budget ≤ ½ current |
| `QWEN3_TTS_STREAM_KV_MAX_NPAST` | (model max) | cap on streaming KV slab growth |
| `QWEN3_TTS_MAX_INPUT_CHARS` | `6144` | input text cap |
| `QWEN3_TTS_DEFAULT_STREAM_BATCH_SIZE` | `30` | default vocoder chunk size |
| `QWEN3_TTS_DEFAULT_STREAM_FIRST_BATCH_SIZE` | `1` | first chunk is one codec frame (lowest TTFA) |
| `QWEN3_TTS_VOCODER_FP16_CASCADE` | `1` | F16 cascade activations; `=0` reverts to F32 (heavier sched_cu) |
| `QWEN3_TTS_VOCODER_NO_Q8_LOAD` | unset | `=1` keeps vocoder mat-mul weights F16 instead of Q8_0 |

### Forced alignment

| Var | Default | Notes |
|---|---|---|
| `QWEN3_TTS_FA_REPO` | unset (alignment disabled) | HF repo of the forced-aligner GGUF. Set to `cstr/qwen3-forced-aligner-0.6b-GGUF` to enable. Empty = `align=true` requests return 400. Resolves to a `--hf-repo-fa` CLI flag in the Docker entrypoint. |
| `QWEN3_TTS_FA_QUANT` | `Q4_K` | quant suffix for the FA repo. Q4_K is the user-supported floor; smaller quants are not validated. |
| `CRISPASR_N_GPU_LAYERS` | `0` (aligner subprocess only — all 28 LLM blk layers on CPU) | aligner-only baked default. Set to `28` to keep the LLM body on GPU (+210 MiB GPU, −5 % wallclock). The synth worker is unaffected by this env. |
| `CRISPASR_QWEN3_ASR_FUSED_QKV` | `0` (aligner-only baked default) | drops the 63 MiB fused-QKV duplicate buffer. Set to `1` to re-enable fusion (slightly faster body forward on the aligner; not worth the VRAM under C1). |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` (aligner-only baked default) | aligner subprocess can't capture useful cuda graphs under the streaming-partial topology-churn workload — disabling reclaims ~48 MiB of context state with no perf loss. Set to `0` to re-enable (no-op on the cache count). |

### Performance opt-outs (debug only — defaults are correct)

| Var | Effect |
|---|---|
| `QWEN3_TTS_NO_FULL_AR_CACHE=1` | disable v9.4 full-AR cgraph cache |
| `QWEN3_TTS_NO_TALKER_GPU_SAMPLE=1` | disable v9.5 GPU-side talker sampling |
| `QWEN3_TTS_NO_TALKER_STEP_CACHE=1` | disable v9.7 talker step cgraph cache |
| `QWEN3_TTS_NO_FULL_AR_GRAPH=1` | disable v7-phase2 full-AR cgraph (per-step fallback) |
| `QWEN3_TTS_NO_ASYNC_VOCODER=1` | disable v9.2 async vocoder dispatch |
| `QWEN3_TTS_TALKER_PRIORITY=default` | disable v9.3 talker = HIGH stream priority (also `low` / `high` overrides) |
| `QWEN3_TTS_VOCODER_PRIORITY=default` | disable v9.3 vocoder = LOW (no-op on RTX 3060) |
| `QWEN3_TTS_SPECIALIZED_MMVQ=0` | disable shape-specialized Q8 MMVQ kernels (fall through to ggml stock) |
| `GGML_CUDA_DISABLE_GRAPHS=1` | disable CUDA-graph capture |

Set to default values in production — the opt-outs exist for diagnostic comparison, not steady-state operation.

### Boot-time CUDA arch trim

Build-arg `QWEN3_TTS_CUDA_ARCHS` controls which arches are baked into the image. Default `75;80;86;89;90` (Turing → Hopper); set to `86` (or whichever arch you target) to drop the per-arch ~15–20 MiB CUDA-context floor on a single-arch host.

## Companion tools

`docker/tts-qwen3-dev/cold-start.py` exercises every path above end-to-end (model reload, voice register HIT/MISS, post-reload TTFA, VoiceDesign, all four streaming-mode combinations) and reports TTFA / wall / framing checks. Run it after a deploy to validate.
