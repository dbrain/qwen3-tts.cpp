# Benchmarks

Single-stream decode on RTX 3060 12 GB / Ampere (sm_86). Linux, CUDA 12.x, calm GPU (single tts-qwen3-server container, no concurrent compute). Streaming SSE+PCM via `POST /v1/audio/speech`.

## Test suite

74-synth `breakit_seeds.py` battery × 2 seeds = 148 synths per config:

- **Voice clone** (3 voices × 4 prompts × 2 seeds = 24): ICL clones from real reference clips.
- **Voice design** (3 designs × 4 prompts × 2 seeds = 24): description-driven via `instructions`.
- **Edge cases** (12 prompts × 2 seeds + 2 cache-reuse = 26): single word, all-caps, numerals-heavy, code, URLs, ellipses, diacritics, rapid Q/A, mixed-case acronyms, very long run-ons, markdown.

Same battery across all 6 configs, so comparisons are apples-to-apples.

## V1 vocoder (24 kHz)

| talker | RTF avg | RTF med | RTF min | RTF max | TTFA avg | TTFA med | VRAM post-boot | VRAM idle (warm) | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Q4_K_M** | 4.841× | 4.923× | 3.374× | **5.031×** | 70.4 ms | 43.4 ms | 1652 MiB | 1932 MiB | 2494 MiB |
| **Q8_0**   | 4.289× | 4.310× | 3.897× | 4.405× | 54.7 ms | 47.4 ms | 2880 MiB | 3156 MiB | 3482 MiB |
| **F16**    | 2.855× | 2.901× | 2.102× | 2.947× | 91.4 ms | 66.4 ms | 4220 MiB | 4492 MiB | 5086 MiB |

## V2 vocoder (48 kHz)

| talker | RTF avg | RTF med | RTF min | RTF max | TTFA avg | TTFA med | VRAM post-boot | VRAM idle (warm) | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Q4_K_M** | 4.728× | 4.814× | 3.264× | 4.923× | 72.5 ms | 43.5 ms | 1654 MiB | 1976 MiB | 2538 MiB |
| **Q8_0**   | 4.178× | 4.250× | 2.898× | 4.336× | 78.2 ms | 50.7 ms | 2882 MiB | 3198 MiB | 3792 MiB |
| **F16**    | 2.819× | 2.864× | 2.065× | 2.906× | 93.7 ms | 66.3 ms | 4222 MiB | 4536 MiB | 4864 MiB |

V2 ≈ V1 minus ~0.1 RTF — V2 produces 2× the audio samples per chunk, vocoder does proportionally more work.

## Take-aways

- **Q4_K_M is the speed leader on Ampere**, at the cost of some quality (subjective; subtle on this model). RTF avg 4.84× / max 5.03× / VRAM peak 2.5 GiB.
- **Q8_0 is the quality default**: RTF avg 4.29×, VRAM peak 3.5 GiB. Pick this unless 2.5 GiB peak is the bar that matters.
- **F16 is the quality reference** but materially slower (2.86×) — F16 weights are 2× the bandwidth per AR step on a memory-bound kernel. Use only if you need the unquantized weights for parity testing.
- **TTFA stays ~40–95 ms across configs** — independent of audio length once `voice.warmup` is hot, because only the first codec frame has to be ready before bytes flow.
- **VRAM scales cleanly with talker quant**: Q4 ~2.5 GiB, Q8 ~3.5 GiB, F16 ~5 GiB peak. V1 vs V2 vocoder is a +50–300 MiB peak delta.

## RTF convention

This fork (and `qwen-research/faster-qwen3-tts`) reports RTF as **audio-seconds-per-wall-second** — higher = faster. RTF 4.29× means 4.29 seconds of 24 kHz audio per wall-second of compute. The strict definition (wall / audio, lower better) is the inverse.

## TTFA methodology

Time-to-first-audio is wall-clock from request fire to receipt of the first `speech.audio.delta` SSE event. Implementation in [`docker/tts-qwen3-dev/breakit_seeds.py`](https://github.com/dbrain/qwen3-tts.cpp/blob/main/docs/) — but the critical detail for any custom Python client:

> **Use `HTTPResponse.read1()`, not `read()` or `for line in resp:`.** Python's urllib uses an internal buffered reader; `read()` and `readline()` block waiting for the buffer to fill, which inflates TTFA by hundreds of ms. `read1()` returns whatever bytes are currently available without trying to top up the buffer. SSE events are then parsed from the raw byte stream by hand (split on `\n\n`, look for `event:` and `data:` prefixes).

Equivalent gotcha for non-Python clients: avoid any HTTP layer that does line-buffered `readline()` over a chunked-transfer-encoded SSE stream.

## VRAM measurement

- **VRAM post-boot**: `nvidia-smi --query-gpu=memory.used` 3 s after `/health` returns 200, before any synth. Reflects the talker + vocoder + speaker-encoder weights resident on GPU.
- **VRAM idle (warm)**: same query 2 s after a single warmup synth completes. Reflects post-warmup steady state — caches primed, no compute. This is what an idle-but-loaded server holds.
- **VRAM peak**: max of `nvidia-smi --query-gpu=memory.used` sampled at 0.3 s intervals across the entire 148-synth bench. Captures transient spikes during long-prompt generation.

## Reproduce

```bash
# Build
./docker/tts-qwen3-dev/iter.sh build

# For each config you want to bench:
MODEL_QUANT=q8     VOCODER=v1 ./docker/tts-qwen3-dev/iter.sh restart
python3 ./docker/tts-qwen3-dev/breakit_seeds.py \
  --service-url http://127.0.0.1:8810 \
  --label q8_v1 --seeds 1,2 --sample-rate 24000

MODEL_QUANT=q4_k_m VOCODER=v1 ./docker/tts-qwen3-dev/iter.sh restart
python3 ./docker/tts-qwen3-dev/breakit_seeds.py \
  --service-url http://127.0.0.1:8810 \
  --label q4_k_m_v1 --seeds 1,2 --sample-rate 24000

# ... etc for q8_v2 (sr=48000), q4_k_m_v2, f16_v1, f16_v2.
```

The script writes audio under `tts-samples/breakit-<ts>-<label>/` (organized by category/voice/test) plus a `metadata.json` with per-prompt RTF/TTFA/audio-bytes for paired diffs.

`MODEL_QUANT=q8` requires no setup (auto-downloads from HF). `q4_k_m` uses `dbrains/Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M-GGUF`. `f16` requires local conversion:

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir /tmp/voicedesign
python scripts/convert_tts_to_gguf.py \
  -i /tmp/voicedesign \
  -o ~/.cache/huggingface/Qwen3-TTS-12Hz-1.7B-VoiceDesign-F16.gguf \
  -t f16
```
