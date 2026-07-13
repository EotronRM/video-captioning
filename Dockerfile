# syntax=docker/dockerfile:1
# check=skip=SecretsUsedInArgOrEnv
#
# The graders run this image with no environment of their own, so the API key has
# to be baked in. BuildKit rightly warns about that; the check is skipped above
# because the contest leaves no alternative. Use a THROWAWAY key (see below).
#
#   docker buildx build --platform linux/amd64 --build-arg VLM_API_KEY=fw_xxx \
#       -t docker.io/<you>/amd-track2:latest --push .
#
# Expects the quantized model at ./models/gemma4-style-Q4_K_M.gguf (~5.3 GB).
# .gitignore keeps *.gguf out of git; .dockerignore keeps the rest out of the
# build context.
#
# We COPY a prebuilt llama-server instead of compiling one. Two reasons:
#  1. `COPY --from=<image>` moves FILES; it never executes them. So this builds on
#     an arm64 Mac without QEMU touching a compiler. Cross-compiling llama.cpp
#     under emulation segfaults cc1plus nondeterministically ("internal compiler
#     error" is GCC's way of reporting that its child died from a signal).
#  2. The official image ships libggml-cpu-{sse42,ivybridge,haswell,skylakex,
#     icelake,sapphirerapids,alderlake,piledriver,...}.so and dlopens whichever
#     matches the host CPU. That is strictly better than a single -DGGML_NATIVE=OFF
#     binary: an old grader CPU still runs, a modern one still gets AVX-512.
#
# Pinned by digest — `:server` is a moving tag. This one is b9935 (f2d1c2f39) and
# carries both flags the design needs: `--swa-full` and `-np`.
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server@sha256:295dc9897fa8a643e4a513fbcaada51d3b8db4b0afa4fda7aeae2386757de58b

FROM ${LLAMA_IMAGE} AS llama

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim

# ffmpeg: remote keyframe seeks. libgomp1: ggml's OpenMP pool.
# libcurl4: llama-server links it (model download support we never use).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 libcurl4 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# The binary plus its ggml backends must stay together — ggml dlopens the CPU
# variant from beside libggml-base.so at model-load time.
COPY --from=llama /app /opt/llama

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY agent ./agent
# Last layer, and by far the biggest: everything above stays cached when it changes.
COPY models/gemma4-style-Q4_K_M.gguf /models/gemma4-style-Q4_K_M.gguf

# The graders run this image with no environment of their own — every setting the
# agent needs must live here. Without these it would default to OpenRouter +
# Qwen3-VL and fail on the first clip.
ENV VLM_BASE_URL=https://api.fireworks.ai/inference/v1 \
    VLM_MODEL=accounts/fireworks/models/kimi-k2p6 \
    REASONING_EFFORT=none

# Same reasoning applies to the key, so it must be baked in:
#   docker buildx build --build-arg VLM_API_KEY=fw_xxx ...
# A KEY IN A PUBLIC IMAGE IS A PUBLIC KEY. It is also recoverable from
# `docker history` even if a later layer unsets it. Use a THROWAWAY key, cap its
# spend, and rotate it the moment judging ends.
ARG VLM_API_KEY=""
ENV VLM_API_KEY=${VLM_API_KEY}

# LOCAL_ROLE=fallback: the API goes first (it wins on quality 10-3 and is far
# faster on 2 CPU cores); the local model only rescues clips the API loses.
# LOCAL_ROLE=primary flips it — measure decode tok/s on the target CPU first.
# LOCAL_THREADS overrides the cgroup-derived thread count.
ENV PATH="/app/.venv/bin:$PATH" \
    LD_LIBRARY_PATH=/opt/llama \
    LOCAL_ROLE=fallback \
    LOCAL_STYLE=1 \
    LOCAL_MODEL_PATH=/models/gemma4-style-Q4_K_M.gguf \
    LOCAL_SERVER_BIN=/opt/llama/llama-server \
    LOCAL_PORT=8080 \
    LOCAL_CTX=4096 \
    LOCAL_WARMUP_TIMEOUT=120 \
    LOCAL_REQUEST_TIMEOUT=120 \
    LOCAL_MIN_REMAINING=120 \
    LOCAL_MAX_FAILURES=2

# LOCAL_STYLE=0 at run time reverts to the pure-API path with zero code changes.
ENTRYPOINT ["python", "-m", "agent.main"]
