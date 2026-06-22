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

If you already installed an older Utopic package:

```sh
uv tool upgrade utopic
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

If a Mac cannot initialize Metal, or you want a portable CPU-only build:

```sh
utopic setup --backend cpu
```

To force Metal on macOS:

```sh
utopic setup --backend metal
```

## Commands

The package installs these launchers:

- `utopic`
- `utopic-server`
- `utopic-mcp`
- `utopic-acp`

Show help:

```sh
utopic --help
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

Run a one-shot prompt:

```sh
utopic run -m /path/to/model.gguf -p "Answer with one word: 2+2?" -n 16
```

For DiffusionGemma-style canvas models, use the entropy-bound path:

```sh
utopic run -m /path/to/diffusiongemma.gguf -p "Answer with one word: 2+2?" -n 16 --eb-steps 48
```

Run the OpenAI-compatible local server:

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
- configure the native build for CPU or CUDA, including CUDA compiler and architecture detection
- build the dependency layer and Utopic
- copy the final binaries into the Utopic cache

The published wheel stays pure Python and does not fetch or compile native code
during `pip install`. Users should not need to clone dependency repositories or
run build-system commands directly for normal setup.

Use the package-managed binary produced by `utopic setup` for user-facing runs.
On the 2026-06-21 GB10 smoke, `/home/adavya/.cache/utopic-current/bin/utopic`
successfully generated from the installed Dream Q4, LLaDA Q4, DiffusionGemma
BF16, and DiffusionGemma Q4 GGUFs. The repo-local native build loaded the same
files, but was stale for DiffusionGemma prompt wrapping.

## Development

Rebuild the bundled chat UI after editing `node/utopic-chat.ts`:

```sh
npm install
npm run build:chat
npm run check:chat
```

Build a wheel:

```sh
python -m pip wheel . --no-deps -w dist/
```
