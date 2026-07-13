# magic-video — Track 2: Video Captioning Agent

Reads `/input/tasks.json`, watches each clip, writes 4 styled captions per clip to
`/output/results.json`.

```
tasks.json ─▶ ffmpeg: 8 keyframe-seeks per clip (remote-first, download fallback)
          ─▶ vision stage: VLM → dense factual description   (caption-accuracy score)
          ─▶ style stage:  LLM → {formal, sarcastic, humorous_tech, humorous_non_tech}
          ─▶ results.json (atomic, always complete)          (style-match score)
```

Splitting vision from style means a caption can only be as accurate as what was
actually observed — the pressure to be funny never contaminates the grounding.

## Run it

The image needs **no configuration**. Everything it requires — model, API endpoint,
credentials — is baked in. Mount an input directory and an output directory:

```bash
docker pull docker.io/rmenacho/magic-video:latest

docker run --rm \
  -v "$PWD/sample_input:/input:ro" \
  -v "$PWD/out:/output" \
  docker.io/rmenacho/magic-video:latest

cat out/results.json
```

Requirements: `linux/amd64`, outbound network (clip download + inference API).
**No GPU.** Three clips take ~55 s on two CPU cores.

## Reliability

Failure must degrade the score, never zero it — a missing style or malformed JSON
scores nothing at all.

- A **complete, valid `results.json` is written before the first network call**, then
  atomically replaced per finished clip (`os.replace`). A crash, hang, or outage still
  leaves a scoreable file.
- **Hard self-deadline** at `TIME_BUDGET_SECONDS` (540 s): the container exits 0 with
  whatever it has rather than being killed at 600 s.
- Every stage walks a ladder: **API → local model → template.** Any local failure
  condemns that path for the rest of the run instead of blocking it.

## The local model

The image ships a **LoRA-distilled Gemma 4, quantized to Q4_K_M** (5.3 GB), served by
`llama-server` on CPU inside the container. It is the **fallback** for the style stage:
if the API dies, the container keeps producing real captions with no GPU. It styles a
clip in ~30 s on two cores.

Watch it take over by pointing the style stage at a model that doesn't exist:

```bash
docker run --rm \
  -v "$PWD/sample_input:/input:ro" -v "$PWD/out-fallback:/output" \
  -e STYLE_MODEL=accounts/fireworks/models/does-not-exist \
  docker.io/rmenacho/magic-video:latest
```

It ships as the fallback rather than the primary because a blind, order-swapped LLM
judge (n=19, minimum detectable effect ±0.44) scored the teacher above the distilled
student 10–3. The same judge found Q4_K_M statistically indistinguishable from bf16 —
4-bit is free.

## Environment (all optional — the defaults are correct)

| Variable | Default | Effect |
|---|---|---|
| `LOCAL_ROLE` | `fallback` | `primary` runs the local model first, API as the net. |
| `LOCAL_STYLE` | `1` | `0` disables the local model entirely (pure API). |
| `LOCAL_THREADS` | cgroup quota | Overrides the derived thread count. |
| `STYLE_MODEL` | = `VLM_MODEL` | Point the style stage at a different model. |
| `TIME_BUDGET_SECONDS` | `540` | Self-imposed deadline. |

`--cpus N` is a CFS *quota*, not a core count — `nproc` inside the container still
reports every host core. The agent reads `/sys/fs/cgroup/cpu.max` and tells llama.cpp
the truth, so its OpenMP threads don't spin-wait against a budget they can't use.

## Build

```bash
docker buildx build --platform linux/amd64 \
  --build-arg VLM_API_KEY=fw_YOUR_KEY \
  -t docker.io/<you>/magic-video:latest --load .
```

Expects the quantized model at `models/gemma4-style-Q4_K_M.gguf`.

The judging VM pulls `linux/amd64` only; an arm64-only image fails with `PULL_ERROR`
before it ever runs. The Dockerfile does **not compile llama.cpp** — it copies a
prebuilt `llama-server` with `COPY --from=<image>`, which moves files and never
executes them, so the build works on any host architecture without emulating a
compiler. The copied image also ships per-microarchitecture ggml backends
(`sse42`, `haswell`, `icelake`, `zen4`, …) and dlopens whichever matches the host CPU.

> **The API key is baked into the image, and a key in a public image is a public key** —
> it is recoverable from `docker history`. Use a throwaway key, cap its spend, and
> delete it once judging ends.

## Local development

```bash
uv sync                                  # create .venv from uv.lock
cp .env.example .env                     # fill in your API key
uv run python -m agent.main              # needs ffmpeg on the host

# Offline plumbing test — no ffmpeg, no network, no API key:
VLM_MOCK=1 INPUT_PATH=sample_input/tasks.json OUTPUT_PATH=out/results.json \
  uv run python -m agent.main
```

## Swapping the backend

The agent speaks the OpenAI-compatible API, so moving from a hosted endpoint to a
self-hosted model on an AMD MI300X is configuration, not code:

```bash
# on the GPU box
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 8000

# in .env
VLM_BASE_URL=http://<gpu-box>:8000/v1
VLM_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct
```
