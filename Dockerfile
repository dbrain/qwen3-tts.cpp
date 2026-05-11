# syntax=docker/dockerfile:1.7
#
# Multi-stage CUDA build for qwen3-tts-server. Compiles the binary
# in stage 1, ships a slim runtime in stage 2.
#
# Build:
#   docker build --build-arg QWEN3_TTS_CUDA_ARCHS=86 -t qwen3-tts.cpp:local .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#     -v qwen3-tts-models:/root/.cache/huggingface \
#     -v qwen3-tts-voices:/app/voice-archive \
#     -e HF_TOKEN="$HF_TOKEN" \
#     qwen3-tts.cpp:local
#
# See README.md and docker-compose.yml in the repo for the recommended
# operational setup (lazy-load + idle-unload + worker-isolation).

# ─── build stage ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# libav*-dev = ffmpeg 6.1 dev headers for in-process mp3 (libmp3lame) +
# ogg-opus (libopus + ogg muxer) encoding behind /v1/audio/speech?
# response_format=. Encoder uses AVChannelLayout (libavutil 5.1+); 6.1 is
# well above the floor.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl ccache \
        pkg-config \
        libavcodec-dev libavformat-dev libavutil-dev

# Source: by default this Dockerfile builds the source tree it sits in
# (COPY of the build context). Override to clone a pinned ref instead by
# setting QWEN3_TTS_REPO + QWEN3_TTS_REF. (Useful for CI building from a
# pinned commit without checking out the source.)
ARG QWEN3_TTS_REPO=
ARG QWEN3_TTS_REF=

# CUDA archs to compile kernels for. Each arch baked in adds ~15-20 MiB
# of resident PTX/cubin to every CUDA process's primary context — i.e.
# ~80 MiB of "stuck at idle" VRAM even after the model unloads, unless
# you're running with QWEN3_TTS_WORKER_ISOLATION=1 (which kills the
# whole subprocess on /unload, dropping all that to zero).
#
# Default covers Turing through Hopper. Override for a single-arch host:
#   75;80;86;89;90  default  — Turing/Ampere/Ada/Hopper
#   86              Ampere   — RTX 3000-series, A-series datacenter
#   89              Ada      — RTX 4000-series, L40
#   90              Hopper   — H100 / H200
#
# Pre-Volta excluded unconditionally: the wmma kernels need
# `__CUDA_ARCH__ >= 700`. sm_100 (Blackwell) needs CUDA 12.8+ — bump
# the base image first if you add it.
ARG QWEN3_TTS_CUDA_ARCHS="75;80;86;89;90"

# Either COPY the build context (default — local source) or git clone
# at the pinned ref. The shell trampoline picks based on QWEN3_TTS_REPO.
WORKDIR /src
COPY . /src/qwen3-tts.cpp-local/
RUN if [ -n "${QWEN3_TTS_REPO}" ]; then \
        rm -rf qwen3-tts.cpp-local && \
        git clone --recurse-submodules "${QWEN3_TTS_REPO}" qwen3-tts.cpp && \
        cd qwen3-tts.cpp && \
        git checkout "${QWEN3_TTS_REF:-main}" && \
        git submodule sync --recursive && \
        git submodule update --init --recursive; \
    else \
        mv qwen3-tts.cpp-local qwen3-tts.cpp && \
        cd qwen3-tts.cpp && \
        git submodule update --init --recursive 2>/dev/null || true; \
    fi

# Wire ccache in front of g++/nvcc. cmake prepends ccache to compile
# invocations; ccache hashes (compiler, flags, preprocessed source) so
# flag-only changes that don't affect a TU give cache hits.
ENV CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=20G

# GGML_CUDA_FA=ON       — CUDA flash-attention kernels.
# GGML_CUDA_GRAPHS=ON   — CUDA graphs collapse per-token kernel launches into
#     a single replayable submission. The talker AR loop relies on this for
#     the v9.4 full-AR cgraph cache; vanilla naive workloads get less win,
#     but the binary picks the right path based on graph topology.
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S /src/qwen3-tts.cpp -B /src/qwen3-tts.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DQWEN3_TTS_SERVER=ON \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${QWEN3_TTS_CUDA_ARCHS}" \
    && cmake --build /src/qwen3-tts.cpp/build --target qwen3-tts-server -j"$(nproc)" \
    && ccache -s


# ─── runtime stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# nvidia/cuda:runtime carries libcudart + libcublas + libcurand. Extras:
#   libgomp1        — ggml uses OpenMP; libgomp.so.1 isn't in cuda:runtime
#   libsndfile1     — qwen3-tts-server's WAV I/O
#   ffmpeg          — runtime libav* libs the binary links against
#   curl            — healthcheck (`curl -sf /health`)
#   ca-certificates — HF auto-download (HTTPS to huggingface.co)
#   python3 + pip + huggingface_hub
#       qwen3-tts-server's hf_resolve() shells out to the `hf` CLI for
#       first-run GGUF downloads. After cache hit, it just resolves the
#       local path; matters only on cold start / new-quant pull.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        curl libsndfile1 libgomp1 ca-certificates \
        ffmpeg \
        python3 python3-pip
# --break-system-packages: ubuntu 24.04 ships PEP 668 ("externally-managed-
# environment") which blocks system-pip writes by default. We're in a
# disposable container with no system Python apps to break.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir --break-system-packages 'huggingface_hub>=0.30'

COPY --from=builder /src/qwen3-tts.cpp/build/qwen3-tts-server /usr/local/bin/qwen3-tts-server

ENV HF_HOME=/root/.cache/huggingface \
    QWEN3_TTS_VOICE_ARCHIVE_DIR=/app/voice-archive

EXPOSE 8000

# Defaults: Q8_0 talker (best quality/VRAM tradeoff on 8-12 GiB GPUs),
# V1 24 kHz vocoder (cleaner upper-octave than V2 48 kHz), 24 MB
# speaker-encoder sidecar (saves the 2.4 GB Base download).
#
# Override at run time:
#   -e QWEN3_TTS_TALKER_REPO=dbrains/Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M-GGUF
#   -e QWEN3_TTS_TALKER_QUANT=Q4_K_M
#   -e QWEN3_TTS_VOCODER_REPO=dbrains/Qwen3-TTS-Tokenizer-12Hz-48kHz-GGUF  # 48 kHz
#   -e QWEN3_TTS_SE_REPO=khimaros/Qwen3-TTS-12Hz-1.7B-Base-GGUF            # full Base
ENV QWEN3_TTS_TALKER_REPO=khimaros/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF \
    QWEN3_TTS_TALKER_QUANT=Q8_0 \
    QWEN3_TTS_VOCODER_REPO=khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF \
    QWEN3_TTS_VOCODER_QUANT=F16 \
    QWEN3_TTS_SE_REPO=dbrains/Qwen3-TTS-12Hz-Speaker-Encoder-GGUF \
    QWEN3_TTS_SE_QUANT=F16

# Forced aligner (opt-in). Empty FA_REPO disables `align=true` requests
# with a 400. When set, the worker lazy-loads ~530 MB (Q4_K) on first
# align request and keeps the model resident until idle-unload kills the
# worker. The aligner is a Qwen3 0.6B variant with a 5000-class lm_head
# predicting 80 ms-resolution timestamp classes — same architecture as
# qwen3-asr, different head.
ENV QWEN3_TTS_FA_REPO= \
    QWEN3_TTS_FA_QUANT=Q4_K

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -sf http://localhost:8000/health || exit 1

# Aligner CLI flag only added when QWEN3_TTS_FA_REPO is non-empty — leave
# it off the command line entirely in the default-off case so the server
# doesn't try to resolve an empty repo spec.
ENTRYPOINT ["/bin/sh", "-c", "exec /usr/local/bin/qwen3-tts-server \
    --hf-repo    \"${QWEN3_TTS_TALKER_REPO}:${QWEN3_TTS_TALKER_QUANT}\" \
    --hf-repo-v  \"${QWEN3_TTS_VOCODER_REPO}:${QWEN3_TTS_VOCODER_QUANT}\" \
    --hf-repo-se \"${QWEN3_TTS_SE_REPO}:${QWEN3_TTS_SE_QUANT}\" \
    ${QWEN3_TTS_FA_REPO:+--hf-repo-fa \"${QWEN3_TTS_FA_REPO}:${QWEN3_TTS_FA_QUANT}\"} \
    -H 0.0.0.0 -p 8000 -j \"$(nproc)\" ${QWEN3_TTS_EXTRA_ARGS:-}"]
