# Utopic Package Manager

Python package management for the Utopic native runtime.

This repository is intentionally thin from the user's perspective. The wheel
installs Python launchers, the package-owned native C++ runtime source, and a
pinned vendored snapshot of Utopic's Python control plane and built chat UI.
Utopic owns product UX, catalog, MCP, OpenAI gateway, and TypeScript chat source;
this repository owns native C++ source packaging, setup, dependency checkout,
build configuration, and binary installation through `utopic setup`.

## Install

```sh
uv tool install utopic
utopic chat
```

Use `uv tool install` for the global `utopic` command. `uv pip install utopic`
installs into the current Python environment instead; use that only inside an
activated project or virtual environment. If you installed into an environment
and the `utopic` script is not on your shell `PATH`, run the same CLI as a
module:

```sh
python -m utopic --help
python -m utopic chat
```

For a reproducible install of this release:

```sh
uv tool install utopic==0.1.8
```

Utopic currently targets Python 3.10 through 3.12. Python is the debuggable
control plane for setup, launchers, catalog checks, and diagnostics; production
generation goes through the packaged native runtime.
If your system `python3` is newer, let uv manage the tool Python or create a
3.12 virtual environment for project-local installs.

If you already installed an older Utopic package:

```sh
uv tool upgrade utopic
```

If you previously installed an exact pinned version such as
`uv tool install utopic==0.1.5`, uv keeps that pin during upgrades. Reinstall
the tool without the pin:

```sh
uv tool install --force utopic
```

`utopic chat` is the easiest first-run path. It checks whether native binaries
exist, runs `utopic setup` once if they do not, shows the curated model list,
pulls the selected GGUF into `~/.cache/utopic/models`, starts the local server,
and drops you into an Ollama-style terminal chat.

When Node.js 18 or newer is on `PATH`, `utopic chat` uses the bundled TypeScript/Node TUI with a `>>>` prompt and streaming output.
If Node is missing or older than 18, `utopic chat` falls back to a minimal built-in Python chat loop so first run still works; install Node.js 18 or newer for the richer TUI.
The rest of the launchers do not require Node.

If you want to run setup separately:

```sh
utopic setup
```

To inspect the local setup state without cloning, building, downloading, or
starting a server:

```sh
utopic doctor
```

If a local build cache gets wedged after an interrupted or older setup run:

```sh
utopic setup --force
```

`utopic setup` detects the best local backend and builds the matching native
runtime:

- macOS with a usable Metal device: `metal`
- NVIDIA host with a usable CUDA compiler: `cuda`
- everything else: `cpu`

The setup command prints the selected backend, detected device, and reason before
building. It installs runtime binaries under `~/.cache/utopic/bin`.

If you prefer a project-local environment:

```sh
uv venv --python 3.12 ~/.venvs/utopic
~/.venvs/utopic/bin/pip install utopic
~/.venvs/utopic/bin/utopic setup
```

For local development from this checkout:

```sh
git clone https://github.com/adavyas/utopic-package-manager.git
cd utopic-package-manager
pip install .
utopic setup
```

## Backend Overrides

Most users should run plain `utopic setup`. To force a backend:

On NVIDIA hosts, build the CUDA backend:

```sh
utopic setup --backend cuda
```

The CUDA setup path detects the local GPU architecture and selects a suitable
CUDA compiler when possible, including CUDA 13 on GB10/DGX Spark hosts. On
hosts with multiple CUDA toolkits, setup pins CMake's CUDA toolkit lookup to
the selected `nvcc` and resets stale toolkit cache entries so the compiler and
runtime libraries do not silently mix versions. On
constrained hosts, limit build parallelism:

```sh
utopic setup --backend cuda --jobs 2
```

On GB10/DGX Spark, setup disables ggml CUDA graphs by default. To override this
for diagnosis:

```sh
UTOPIC_CUDA_GRAPHS=ON utopic setup --backend cuda --force
```

If a Mac cannot initialize Metal, or you want a portable CPU-only build:

```sh
utopic setup --backend cpu
```

To force Metal on macOS:

```sh
utopic setup --backend metal
```

## Models

`utopic chat`, `utopic run`, and `utopic models pull` accept either a local
GGUF path or a curated model alias. The catalog is modality-aware: each entry
declares its runtime (`native`, `planned_native`, or explicit experimental
`bridge`), engine, hardware fit, OpenAI-style endpoints, and output artifact
type.

| Alias | Model | Modality | Runtime | Notes |
|---|---|---|---|---|
| `diffusiongemma-26b-a4b-q4` | DiffusionGemma 26B-A4B IT Q4_K_M | text | native | Default local text model and safest DiffusionGemma memory fit. |
| `diffusiongemma-26b-a4b-q5` | DiffusionGemma 26B-A4B IT Q5_K_M | text | native | Higher-quality DiffusionGemma quant for 48 GB Apple Silicon, GB10, and CUDA hosts. |
| `diffusiongemma-26b-a4b-q6` | DiffusionGemma 26B-A4B IT Q6_K | text | native | High-quality DiffusionGemma quant for larger local memory budgets. |
| `diffusiongemma-26b-a4b-q8` | DiffusionGemma 26B-A4B IT Q8_0 | text | native | Near-lossless 8-bit DiffusionGemma weights for GB10 and high-memory CUDA hosts. |
| `qwen-image` | Qwen-Image | image | planned_native | Open-weight image generation model with strong prompt following and text rendering. |
| `flux-1-schnell` | FLUX.1-schnell | image | planned_native | Fast Apache-licensed image generation model for local 1-4 step generation. |
| `krea-2-raw` | Krea 2 Raw | image | planned_native | High-quality Krea text-to-image model; GB10 or high-memory CUDA recommended for the future native implementation. |
| `cosmos3-super` | Cosmos3 Super Text2Image | image | planned_native | Agentic high-memory NVIDIA Cosmos3 image model; preflights GPU memory before loading. |
| `kokoro-82m` | Kokoro 82M | tts | planned_native | Tiny, fast open-weight TTS model for local speech synthesis. |
| `chatterbox` | Chatterbox | tts | planned_native | Higher-quality open-weight TTS and voice cloning model. |
| `dia-1.6b` | Dia 1.6B | tts | planned_native | Open-weight dialogue TTS model for expressive multi-speaker speech. |
| `ace-step-3.5b` | ACE-Step 3.5B | music | planned_native | Open-weight music generation model for local text-to-music experiments. |
| `wan2.1-t2v-1.3b` | Wan2.1 T2V 1.3B | video | planned_native | Small open-weight text-to-video model; laptop-plausible at modest resolution once native support lands. |
| `wan2.1-t2v-14b` | Wan2.1 T2V 14B | video | planned_native | Higher-quality open-weight text-to-video model; GB10 or high-memory CUDA recommended. |
| `ltx-video` | LTX-Video | video | planned_native | Optional LTX video target; license and runtime differ from Wan. |
| `zuna` | ZUNA | misc | planned_native | Open-weight EEG and signal foundation model exposed as a generic file-in/file-out artifact workflow. |

The native text path is centered on DiffusionGemma canvas / entropy-bound GGUF
models. Other modalities are cataloged as planned native implementations and share the
same model cache, OpenAI-compatible gateway, and MCP tool contract that the
future C++ engines will use.

Large planned native models can declare runtime requirements in the catalog. For
example, `cosmos3-super` is discoverable through `utopic models list`,
`/v1/models`, and MCP, but requests fail fast with
`native_runner_oom_preflight` on hosts below its GPU-memory requirement instead
of trying to load the model and crashing inside CUDA, Metal, or Python runtime
code.

DiffusionGemma is exposed as curated aliases for the practical quantization
ladder. The BF16 DiffusionGemma file is intentionally not in the default catalog
because it is about 47 GiB before runtime buffers, which is too close to the
usable memory ceiling on 48 GB Apple Silicon. Use a local GGUF path for that file
on larger hosts if needed. The package CUDA setup path has been validated on
GB10/DGX Spark, a 6x RTX 4090 host, and a 4x A100 host. DiffusionGemma Q4_K_M,
Q5_K_M, Q6_K, and Q8_0 all pull, size-check, load, fully offload, and complete a
native C++ smoke prompt on the 4x A100 host. Q4_K_M also completes native C++
smoke tests on GB10/DGX Spark and 6x RTX 4090 CUDA. The previous GB10 SOFT_MAX
failure was caused by a CUDA compiler/toolkit mismatch during setup.

FP8 file names are recognized through common GGUF markers such as `FP8`,
`F8_E4M3`, `F8_E5M2`, `E4M3`, and `E5M2`. Quantized GGUF weights include common
`Q8_0`, `Q6_K`, `Q5_*`, `Q4_*`, `Q3_*`, `Q2_K`, `IQ*`, and `NVFP4` markers,
subject to what the package-managed llama.cpp build can actually load on the
target backend.

## Commands

The package installs these launchers:

- `utopic`
- `utopic-runtime`
- `utopic-bridge`
- `utopic-server`
- `utopic-mcp`
- `utopic-acp`

Show help:

```sh
utopic --help
utopic --version
utopic chat --help
utopic run --help
utopic generate --help
utopic gateway --help
utopic-runtime --help
utopic-bridge --help
utopic doctor
utopic models --help
```

Start the terminal chat UI:

```sh
utopic chat
```

Use a model alias or local GGUF path:

```sh
utopic chat diffusiongemma-26b-a4b-q4
utopic chat -m /path/to/model.gguf
```

Inside chat:

```text
>>> hi
Thinking...
Hi there. How can I help?

/help
/clear
/system You are concise.
/exit
```

List or pull curated models:

```sh
utopic models list
utopic models pull diffusiongemma-26b-a4b-q4
utopic models path diffusiongemma-26b-a4b-q4
utopic models check --all
utopic models check qwen-image
```

To prepare every catalog entry explicitly:

```sh
utopic models pull --all
```

This downloads every native GGUF catalog file and prepares metadata directories
for planned image, TTS, music, video, and misc artifact models. It is
intentionally opt-in because the DiffusionGemma quantization ladder is large.
For most users, pull the single native text model they plan to run first.

`utopic models check --all` prints one JSON readiness report for every catalog
model and exits nonzero if any model is not ready. `utopic models check <alias>`
prints the same readiness shape for one selected model. For native GGUF models
it checks the local file and expected byte size when known. For planned non-text
models it reports `native_runner_not_ready` until native support lands behind
`utopic-runner`, plus the
declared endpoints, model metadata, and hardware requirements.

`utopic doctor` prints package version, cache/bin paths, native runtime cache
state, and Node.js status without probing package-manager build internals.
If native binaries are missing or stale, run `utopic setup`; that command owns
backend detection, CMake/git checks, and native builds.

Python bridge adapters have been retired from the package-manager production
surface. The `utopic-bridge` command remains as a compatibility shim for older
scripts, but `utopic-bridge <engine> --check` reports the adapter as retired and
regular invocations return structured `native_runner_required` JSON that points
callers back to `utopic setup`. A normal Utopic install no longer advertises
Diffusers, Torch, TTS, music, or video bridge extras.

Call the same catalog and gateway contract for planned artifact modalities:

```sh
utopic generate image qwen-image -p "A crisp product photo of a titanium robot assistant" --size 1024x1024 --steps 30 -o image.png
utopic generate image krea-2-raw -p "A crisp editorial poster of a glass coastal city" --size 1024x1024 -o krea.png
utopic generate speech dia-1.6b --input "Utopic is running locally." -o speech.wav
utopic generate music ace-step-3.5b -p "bright synthwave with warm analog drums" --duration 30 --lyrics "" -o music.wav
utopic generate video --quality high -p "A cinematic sunrise over a glass coastal city, slow aerial camera move" --size 832x480 --frames 49 --steps 20 --fps 16 -o video.mp4
utopic generate misc zuna --artifact /path/to/input.bin -o output.bin
```

`utopic generate video --quality high` selects the higher-quality
`wan2.1-t2v-14b` catalog entry when no explicit video model is provided.
`--quality fast` selects the smaller `wan2.1-t2v-1.3b` model. The `speech`
subcommand also accepts the `tts` alias. Until native implementations land
behind `utopic-runner` for these non-text modalities, the default response is a
structured native-readiness error. Use `--param KEY=JSON` to pass request
options through the gateway contract.

Start the local runtime and print the live OpenAI-compatible and MCP URLs:

```sh
utopic run diffusiongemma-26b-a4b-q4 --port 8910 -ngl 99
```

The public runtime endpoints are:

```text
http://127.0.0.1:8910/v1/chat/completions
http://127.0.0.1:8910/v1/responses
http://127.0.0.1:8910/v1/images/generations
http://127.0.0.1:8910/v1/audio/speech
http://127.0.0.1:8910/v1/audio/generations
http://127.0.0.1:8910/v1/videos/generations
http://127.0.0.1:8910/v1/utopic/misc/generations
http://127.0.0.1:8910/v1/models
http://127.0.0.1:8910/mcp
```

`utopic run` is the server process, not an interactive prompt. It starts the
native C++ DiffusionGemma text server behind the unified Utopic gateway, so the
same public port also exposes image, TTS, music, video, model catalog, and MCP
routes. To chat with a server that is already running:

```sh
utopic chat --server http://127.0.0.1:8910
```

If you need to choose the private native backend port explicitly:

```sh
utopic run diffusiongemma-26b-a4b-q4 --port 8910 --native-port 9910 -ngl 99
```

Planned non-text models start the gateway without starting a native text server:

```sh
utopic run qwen-image --port 8910
```

For planned non-text models, the command prints the OpenAI-compatible
endpoint(s) declared by that model, such as `/v1/images/generations` and
`/v1/responses`, plus `/v1/models` and `/mcp`. Requests return
native-readiness errors until the matching native implementation is available
behind `utopic-runner`.

`utopic gateway` remains available for advanced setups where you already have a
native text server running separately:

```sh
utopic gateway --port 8911 --native-base-url http://127.0.0.1:8910
```

The gateway exposes catalog and native-readiness contracts even before every
modality has a native C++ engine:

```text
GET  http://127.0.0.1:8911/v1/models
POST http://127.0.0.1:8911/v1/chat/completions
POST http://127.0.0.1:8911/v1/responses
POST http://127.0.0.1:8911/v1/images/generations
POST http://127.0.0.1:8911/v1/audio/speech
POST http://127.0.0.1:8911/v1/audio/generations
POST http://127.0.0.1:8911/v1/videos/generations
POST http://127.0.0.1:8911/v1/utopic/misc/generations
POST http://127.0.0.1:8911/mcp
```

The `/mcp` route speaks JSON-RPC over HTTP for MCP `initialize`, `ping`, `tools/list`, and `tools/call`. The listed tools cover text chat, image
generation, speech, music, video, misc artifacts, and model cache operations, all routed through
the same runtime boundary as the OpenAI-compatible endpoints.

Native runner artifact engines use the same artifact/progress shape for planned
image, speech, music, video, and misc models. A runner request includes the
endpoint, model metadata, normalized input, model cache path, output directory,
progress event path, and hardware/output metadata:

```json
{
  "schema_version": "utopic-runner/v1",
  "run_id": "run_...",
  "endpoint": "/v1/images/generations",
  "model": "qwen-image",
  "repo": "Qwen/Qwen-Image",
  "modality": "image",
  "engine": "native-image",
  "input": { "prompt": "..." },
  "parameters": { "size": "1024x1024" },
  "model_cache_path": "/Users/me/.cache/utopic/models/qwen-image",
  "output_dir": "/Users/me/.cache/utopic/runs/run_.../outputs",
  "progress_path": "/Users/me/.cache/utopic/runs/run_.../progress.jsonl"
}
```

Progress is newline-delimited JSON written to `progress_path`:

```json
{"event":"generating","progress":0.5,"message":"halfway"}
{"event":"completed","progress":1.0,"message":"done"}
```

The native runner response returns artifacts:

```json
{
  "artifacts": [
    { "type": "image/png", "path": "/absolute/path/to/image.png", "metadata": {} }
  ],
  "metadata": { "engine_version": "local" }
}
```

Utopic normalizes this into an artifact response with `file://` URLs, a
`progress_url`, and cached progress events available at:

```text
GET /v1/utopic/runs/{run_id}/events
```

`/v1/responses` is normalized at the gateway boundary. Text requests are
translated to native chat-completions input and wrapped back into a
Responses-style object. Image, TTS, music, video, and misc requests translate
Responses `input` text into the planned modality request shape. By default they
return native-readiness errors until native support lands behind `utopic-runner`;
artifact-producing runs return a Responses-style artifact message, while the
modality-specific endpoints return the richer `utopic.artifact.response` object.

The gateway routes planned image, TTS, music, video, and misc models through the
native runner contract and returns native runner readiness errors until their
native implementations land behind `utopic-runner`. This keeps production
generation local-native and prevents a normal install from silently falling back
to Torch or Diffusers.

The retired `utopic-bridge` command is intentionally small. It preserves old
engine names for compatibility and returns JSON diagnostics instead of importing
or running Python model stacks:

```sh
utopic-bridge diffusers --check
utopic-bridge kokoro --check
utopic-bridge wan --check
utopic-bridge artifact --check
```

The check prints JSON with `ready: false`, `status: "retired"`, and a message
that points callers back to `utopic setup`, the local native runner, the gateway,
or MCP surfaces.

Current release smoke coverage:

| Area | Coverage |
|---|---|
| Package install | Fresh wheel install, `utopic-runtime --help`, and `utopic-bridge --help` |
| Runtime gateway | `/v1/models`, `/v1/responses`, modality-specific OpenAI-style routes, `/mcp`, and MCP `tools/list` / `tools/call` |
| Hardware surface | Apple Silicon, GB10/DGX Spark, RTX 4090 CUDA, and 4x A100 CUDA smoke tests for installed wheel, catalog, MCP tools, native-readiness diagnostics, and artifact contract |
| Native text generation | DiffusionGemma Q4_K_M native C++ smoke tests on GB10/DGX Spark, 6x RTX 4090 CUDA, and 4x A100 CUDA; Q5_K_M, Q6_K, and Q8_0 native C++ smoke tests on 4x A100 CUDA, using the package-managed llama.cpp build |
| Planned artifact modalities | Qwen-Image, FLUX, Krea, Cosmos3, Kokoro, Chatterbox, Dia, ACE-Step, Wan, LTX, and ZUNA expose stable routes, MCP tools, hardware preflight metadata, and native-readiness errors until native implementations land behind `utopic-runner` |
| Retired bridge shim | `utopic-bridge --help` and `utopic-bridge <engine> --check` stay available for compatibility, but generation requests return structured `native_runner_required` JSON |

The MCP endpoint exposes the same runtime through tools:

- `utopic_chat`
- `utopic_generate_image`
- `utopic_speak`
- `utopic_generate_music`
- `utopic_generate_video`
- `utopic_generate_misc`
- `utopic_models_list`
- `utopic_models_check`
- `utopic_models_pull`

`utopic_models_pull` accepts either `{ "model": "qwen-image" }` for one
catalog entry or `{ "all": true }` to prepare the whole catalog.

Run a one-shot prompt:

```sh
utopic run -m /path/to/model.gguf -p "Answer with one word: 2+2?" -n 16
```

For DiffusionGemma-style canvas models, use the entropy-bound path:

```sh
utopic run -m /path/to/diffusiongemma.gguf -p "Answer with one word: 2+2?" -n 16 --eb-steps 48
```

Low-level native launchers are also available after `utopic setup` has installed
the cached binaries:

```sh
utopic-server -m /path/to/model.gguf --host 127.0.0.1 --port 8910 -ngl 99
```

Health and model list:

```sh
curl http://127.0.0.1:8910/health
curl http://127.0.0.1:8910/v1/models
```

## What Setup Owns

The package manager owns the user-facing setup path:

- own the package-managed CMake project used for local native builds
- use the vendored Utopic core source snapshot
- fetch the pinned compatible public llama.cpp dependency source
- configure the native build for Metal, CUDA, or CPU, including CUDA compiler and architecture detection
- build the dependency layer and Utopic
- copy the final binaries into the Utopic cache

The published wheel stays pure Python and does not fetch or compile native code
during `pip install`. Users should not need to clone dependency repositories or
run build-system commands directly for normal setup. Models are downloaded by
`utopic chat`, `utopic run`, or `utopic models pull` when you choose a curated
model alias.

## Development

Native C++ and CMake changes land directly in this repository. Python control
plane, model catalog, MCP/OpenAI routing, and chat UI source changes land in the
main Utopic repository first; then vendor a pinned control-plane snapshot here:

```sh
python scripts/vendor_core.py --ref <utopic-commit>
```

Validate the vendored chat UI:

```sh
npm install
npm run check:chat
```

Build the same release distributions that CI validates:

```sh
python -m build
python -m twine check dist/*
```

The `Upload Python Package` GitHub Actions workflow has two modes. Manual `workflow_dispatch` runs validate release artifacts only.
They run the Python compatibility matrix, build and audit the source
distribution and wheel, smoke test installed artifacts, and upload the built
distributions as workflow artifacts, but they do not publish to PyPI.

Only a published GitHub Release can run the PyPI publish job. That job uses the
protected `pypi` environment and PyPI trusted publishing, so do a manual
workflow run first and inspect the uploaded artifacts before creating the
GitHub Release.
