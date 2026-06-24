# Utopic Package Manager

Python package management for the Utopic native runtime.

This repository is intentionally thin. The wheel installs Python launchers plus
a pinned vendored snapshot of the shippable Utopic runtime source. Product
runtime code, native C++ code, and the TypeScript chat UI are developed in the
main Utopic repository; this repository packages that snapshot and owns setup,
dependency checkout, build configuration, and binary installation through
`utopic setup`.

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

Utopic currently targets Python 3.10 through 3.12. That range matches the
native launcher and the optional image, speech, music, video, and misc bridge engines.
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
declares its runtime (`native` or `bridge`), engine, hardware fit, OpenAI-style
endpoints, and output artifact type.

| Alias | Model | Modality | Runtime | Notes |
|---|---|---|---|---|
| `diffusiongemma-26b-a4b-q4` | DiffusionGemma 26B-A4B IT Q4_K_M | text | native | Default local text model and safest DiffusionGemma memory fit. |
| `diffusiongemma-26b-a4b-q5` | DiffusionGemma 26B-A4B IT Q5_K_M | text | native | Higher-quality DiffusionGemma quant for 48 GB Apple Silicon, GB10, and CUDA hosts. |
| `diffusiongemma-26b-a4b-q6` | DiffusionGemma 26B-A4B IT Q6_K | text | native | High-quality DiffusionGemma quant for larger local memory budgets. |
| `diffusiongemma-26b-a4b-q8` | DiffusionGemma 26B-A4B IT Q8_0 | text | native | Near-lossless 8-bit DiffusionGemma weights for GB10 and high-memory CUDA hosts. |
| `qwen-image` | Qwen-Image | image | bridge | Open-weight image generation model with strong prompt following and text rendering. |
| `flux-1-schnell` | FLUX.1-schnell | image | bridge | Fast Apache-licensed image generation model for local 1-4 step generation. |
| `krea-2-raw` | Krea 2 Raw | image | bridge | High-quality Krea text-to-image model through Diffusers Krea2Pipeline; GB10 or high-memory CUDA recommended until Mac generation is validated. |
| `cosmos3-super` | Cosmos3 Super Text2Image | image | bridge | Agentic high-memory NVIDIA Cosmos3 image model; preflights GPU memory before loading. |
| `kokoro-82m` | Kokoro 82M | tts | bridge | Tiny, fast open-weight TTS model for local speech synthesis. |
| `chatterbox` | Chatterbox | tts | bridge | Higher-quality open-weight TTS and voice cloning model. |
| `dia-1.6b` | Dia 1.6B | tts | bridge | Open-weight dialogue TTS model for expressive multi-speaker speech. |
| `ace-step-3.5b` | ACE-Step 3.5B | music | bridge | Open-weight music generation model for local text-to-music experiments. |
| `wan2.1-t2v-1.3b` | Wan2.1 T2V 1.3B | video | bridge | Small open-weight text-to-video model; laptop-plausible at modest resolution. |
| `wan2.1-t2v-14b` | Wan2.1 T2V 14B | video | bridge | Higher-quality open-weight text-to-video model; GB10 or high-memory CUDA recommended. |
| `ltx-video` | LTX-Video | video | bridge | Optional LTX video bridge; license and runtime differ from Wan. |
| `zuna` | ZUNA | misc | bridge | Open-weight EEG and signal foundation model exposed as a generic file-in/file-out artifact workflow. |

The native text path is centered on DiffusionGemma canvas / entropy-bound GGUF
models. Other modalities use bridge engines today but share the same catalog,
model cache, OpenAI-compatible gateway, and MCP tool contract that future native
C++ engines will use.

Large bridge models can declare runtime requirements in the catalog. For
example, `cosmos3-super` is discoverable through `utopic models list`,
`/v1/models`, and MCP, but requests fail fast with
`bridge_model_oom_preflight` on hosts below its GPU-memory requirement instead
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

This downloads every native GGUF catalog file and prepares bridge metadata
directories for image, TTS, music, video, and misc artifact models. It is intentionally
opt-in because the DiffusionGemma quantization ladder is large. For most users,
pull the single text model or bridge model they plan to run first.

`utopic models check --all` prints one JSON readiness report for every catalog
model and exits nonzero if any model is not ready. `utopic models check <alias>`
prints the same readiness shape for one selected model. For native GGUF models
it checks the local file and expected byte size when known. For bridge models it
checks both the prepared model cache metadata and the optional Python engine
dependencies, then prints concrete next steps such as `utopic models pull
qwen-image` or the relevant `pip install ...` command.

`utopic doctor` prints the detected backend, native binary cache state, required
setup tools, Node.js status, and one compact bridge-engine readiness line for
each image, TTS, music, video, and misc adapter. For detailed bridge import errors,
run the matching `utopic-bridge <engine> --check` command.

Install bridge dependencies by modality when you want non-text generation:

```sh
uv pip install --python ~/.venvs/utopic/bin/python "utopic[image]"   # Qwen-Image and FLUX through Diffusers
uv pip install --python ~/.venvs/utopic/bin/python "utopic[tts]"     # Kokoro and Dia
uv pip install --python ~/.venvs/utopic/bin/python "utopic[chatterbox]" # Chatterbox, isolated because it pins Diffusers
uv pip install --python ~/.venvs/utopic/bin/python "utopic[music]"   # Shared music audio stack, including TorchCodec
uv pip install --python ~/.venvs/utopic/bin/python "utopic[video]"   # Wan and LTX video through Diffusers
uv pip install --python ~/.venvs/utopic/bin/python "utopic[bridge]"  # Image, Kokoro/Dia, music, and video bridge groups; misc artifact bridge has no extra dependencies
```

Kokoro also needs the spaCy English model in the same Python environment:

```sh
~/.venvs/utopic/bin/python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Inside an already activated Python 3.10-3.12 environment, the same commands can
be shorter:

```sh
uv pip install "utopic[image]"
uv pip install "utopic[tts]"
uv pip install "utopic[chatterbox]"
uv pip install "utopic[music]"
uv pip install "utopic[video]"
uv pip install "utopic[bridge]"
```

Some research engines still require their upstream source package in addition
to the PyPI-safe extra. `utopic models check <alias>` and `utopic-bridge
<engine> --check` print the exact follow-up command when that applies.
Chatterbox is a separate extra because the current upstream package pins an
older Diffusers version that conflicts with the image and video bridge engines.
ACE-Step currently works best in a Python 3.10 bridge environment, then:

```sh
uv pip install "utopic[music]"
uv pip install git+https://github.com/ace-step/ACE-Step.git
utopic-bridge ace-step --check
```

Generate local artifacts directly from the same catalog and gateway contract:

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
subcommand also accepts the `tts` alias. Use `--param KEY=JSON` to pass an
engine-specific bridge option that is not exposed as a first-class CLI flag yet.

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

Bridge-only models start the gateway without starting a native text server:

```sh
utopic run qwen-image --port 8910
```

For bridge-only models, the command prints the OpenAI-compatible endpoint(s)
declared by that model, such as `/v1/images/generations` and `/v1/responses`,
plus `/v1/models` and `/mcp`.

`utopic gateway` remains available for advanced setups where you already have a
native text server running separately:

```sh
utopic gateway --port 8911 --native-base-url http://127.0.0.1:8910
```

The gateway exposes catalog and bridge contracts even before every modality has
a native C++ engine:

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

Bridge engines use the same `utopic-bridge/v1` contract that future native C++
engines will replace. A bridge command receives one JSON request on stdin and
returns one JSON response on stdout. The request includes the endpoint, model
metadata, normalized input, model cache path, output directory, progress event
path, and hardware/output metadata:

```json
{
  "schema_version": "utopic-bridge/v1",
  "run_id": "run_...",
  "endpoint": "/v1/images/generations",
  "model": "qwen-image",
  "repo": "Qwen/Qwen-Image",
  "modality": "image",
  "engine": "diffusers",
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

The bridge response returns artifacts:

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
Responses `input` text into the bridge `prompt`, `input`, or `artifact` field and return a
Responses-style artifact message, while the modality-specific endpoints return
the richer `utopic.artifact.response` object.

By default, the gateway runs the packaged bridge as
`python -m utopic.bridge <engine>`, so a normal install can dispatch to image,
TTS, music, video, and misc bridge adapters without extra shell wiring. Optional
engine packages still need to be installed for real generation.

Engine-specific environment variables override the packaged bridge when you
want to point a model family at a custom bridge binary, script, or future native
engine. For example:

```sh
export UTOPIC_BRIDGE_DIFFUSERS_COMMAND="utopic-bridge diffusers"
export UTOPIC_BRIDGE_KOKORO_COMMAND="utopic-bridge kokoro"
export UTOPIC_BRIDGE_CHATTERBOX_COMMAND="utopic-bridge chatterbox"
export UTOPIC_BRIDGE_DIA_COMMAND="utopic-bridge dia"
export UTOPIC_BRIDGE_ACE_STEP_COMMAND="utopic-bridge ace-step"
export UTOPIC_BRIDGE_WAN_COMMAND="utopic-bridge wan"
export UTOPIC_BRIDGE_LTX_COMMAND="utopic-bridge ltx"
export UTOPIC_BRIDGE_ARTIFACT_COMMAND="utopic-bridge artifact"
```

`GET /v1/models` also exposes the bridge activation metadata for every bridge
model so clients and MCP hosts can discover the exact `utopic-bridge <engine>`
command, environment variable, install hint, input key, output types, and
progress event names without first attempting a generation request.

Check an optional bridge engine before downloading or running heavyweight
models:

```sh
utopic-bridge diffusers --check
utopic-bridge kokoro --check
utopic-bridge wan --check
utopic-bridge artifact --check
```

The check prints JSON with `ready`, `status`, `missing`, `install_hint`, and any
API mismatch message from the installed Python stack.
If the message says `torch/torchvision versions are incompatible`, install a
matching `torch` and `torchvision` pair in that same Python environment before
retrying Diffusers, Wan, LTX, or Dia bridge generation.

Current release smoke coverage:

| Area | Coverage |
|---|---|
| Package install | Fresh wheel install, `utopic-runtime --help`, and `utopic-bridge --help` |
| Runtime gateway | `/v1/models`, `/v1/responses`, modality-specific OpenAI-style routes, `/mcp`, and MCP `tools/list` / `tools/call` |
| Hardware surface | Apple Silicon, GB10/DGX Spark, RTX 4090 CUDA, and 4x A100 CUDA smoke tests for installed wheel, catalog, MCP tools, bridge diagnostics, and artifact contract |
| Native text generation | DiffusionGemma Q4_K_M native C++ smoke tests on GB10/DGX Spark, 6x RTX 4090 CUDA, and 4x A100 CUDA; Q5_K_M, Q6_K, and Q8_0 native C++ smoke tests on 4x A100 CUDA, using the package-managed llama.cpp build |
| Real bridge generation | Qwen-Image PNG on CUDA, Kokoro WAV, Chatterbox WAV, Dia WAV, ACE-Step WAV through the MCP gateway on CUDA, and Wan2.1 1.3B MP4 on CUDA |
| Heavy bridge models | FLUX.1-schnell, Krea 2 Raw, Cosmos3 Super, LTX, and ZUNA expose stable routes and diagnostics, but can still require Hugging Face access, a complete local model download, sufficient GPU memory, and matching local Python engine dependencies before real generation |

The packaged `utopic-bridge` command provides stable adapter entrypoints and
dependency diagnostics for `diffusers`, `cosmos`, `kokoro`, `chatterbox`, `dia`,
`ace-step`, `wan`, `ltx`, and `artifact`. The adapters translate the shared Utopic request into
their local Python engines:

- `diffusers`: Qwen-Image, FLUX, and Krea image generation.
- `cosmos`: Cosmos3 Super diagnostics and external bridge-command handoff.
- `kokoro`: Kokoro speech synthesis.
- `chatterbox`: Chatterbox speech synthesis.
- `dia`: Dia speech synthesis through the Transformers implementation.
- `ace-step`: ACE-Step music generation through the installed ACE-Step Python
  package.
- `wan`: Wan text-to-video generation through Diffusers.
- `ltx`: LTX-Video generation through Diffusers.
- `artifact`: generic misc file-in/file-out artifact bridge, including the ZUNA catalog entry.

If an optional engine package is missing, the gateway returns a structured
`bridge_dependency_missing` response with an install hint. If a fast-moving
upstream package changes its Python API, the gateway returns
`bridge_adapter_api_mismatch` instead of crashing. Bridge failures include
`run_id`, `progress_url`, and any progress events already emitted by the
adapter, so OpenAI clients and MCP hosts can still show useful diagnostics for
long image, speech, music, video, or misc jobs. These bridge commands can be replaced
by future native C++ engines without changing the OpenAI endpoint, MCP tool,
cache, progress, or artifact contract.

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

Diffusers bridge requests also accept `"disable_safety_checker": true` in
`parameters` for nonstandard local pipelines whose built-in safety checker is
incompatible with the generated image dimensions. It is opt-in and is passed only
to the local bridge process.

For bridge models, `repo` is the upstream model source and `model_cache_path` is
the Utopic-managed local cache directory. If the cache directory already contains
real model files, adapters load from it directly; if it only contains Utopic
metadata, adapters load from `repo` while using `model_cache_path` as the cache
location.

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

Runtime, native, model catalog, and chat UI source changes should land in the
main Utopic repository first. Then vendor a pinned core snapshot here:

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
