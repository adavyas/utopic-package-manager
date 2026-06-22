# Utopic Package Manager

Python package management for the Utopic native runtime.

This repository is intentionally thin. The wheel installs Python launchers plus
the small Utopic native source tree. Dependency checkout, build configuration,
and binary installation all happen later through `utopic setup`.

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

The chat UI is a bundled Node app, so Node.js 18 or newer must be on `PATH` for
`utopic chat`. The rest of the launchers do not require Node.

If you want to run setup separately:

```sh
utopic setup
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
python3 -m venv ~/.venvs/utopic
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
GGUF path or a curated model alias. The current curated aliases are:

| Alias | Model | Weight | Notes |
|---|---|---|---|
| `dream-7b-q4` | Dream 7B Instruct Q4_K_M | Q4_K_M | Recommended first local chat model. |
| `llada-8b-q4` | LLaDA 8B Instruct Q4_K_M | Q4_K_M | Discrete diffusion instruct model. |

The runtime code is broader than the curated download list. Utopic has native
paths for GGUF models by architecture family:

| Family | Path | Supported weight classes |
|---|---|---|
| LLaDA | masked | GGUF tensor types the linked llama.cpp build can load |
| Dream | masked | GGUF tensor types the linked llama.cpp build can load |
| DiffusionGemma | canvas / entropy-bound | Experimental GGUF path |

DiffusionGemma GGUF paths are still experimental in this package. GB10 CUDA
currently fails inside ggml CUDA SOFT_MAX for the tested Q4_K_M and BF16 GGUF
files, so DiffusionGemma is not exposed as a one-command curated download until
that backend path is fixed and revalidated.

FP8 file names are recognized through common GGUF markers such as `FP8`,
`F8_E4M3`, `F8_E5M2`, `E4M3`, and `E5M2`. Quantized GGUF weights include common
`Q8_0`, `Q6_K`, `Q5_*`, `Q4_*`, `Q3_*`, `Q2_K`, `IQ*`, and `NVFP4` markers,
subject to what the package-managed llama.cpp build can actually load on the
target backend.

## Commands

The package installs these launchers:

- `utopic`
- `utopic-server`
- `utopic-mcp`
- `utopic-acp`

Show help:

```sh
utopic --help
utopic --version
utopic chat --help
utopic run --help
utopic models --help
```

Start the terminal chat UI:

```sh
utopic chat
```

Use a model alias or local GGUF path:

```sh
utopic chat dream-7b-q4
utopic chat -m /path/to/model.gguf
```

Inside chat:

```text
/help
/clear
/system You are concise.
/exit
```

List or pull curated models:

```sh
utopic models list
utopic models pull dream-7b-q4
utopic models path dream-7b-q4
```

Start an OpenAI-compatible local server and print the live URL:

```sh
utopic run dream-7b-q4 --port 8910 -ngl 99
```

The server endpoint is:

```text
http://127.0.0.1:8910/v1/chat/completions
```

`utopic run` is the server process, not an interactive prompt. To chat with a
server that is already running:

```sh
utopic chat --server http://127.0.0.1:8910
```

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

- use the packaged Utopic native source
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

Rebuild the bundled chat UI after editing `node/utopic-chat.ts`:

```sh
npm install
npm run build:chat
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
