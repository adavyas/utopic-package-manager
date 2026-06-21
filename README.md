# Utopic Package Manager

Python package management for the Utopic native runtime.

This repository is intentionally thin. The wheel installs Python launchers only.
Native source checkout, build configuration, and binary installation all happen
later through `utopic setup`.

## Install

```sh
pip install git+https://github.com/adavyas/utopic-package-manager.git
utopic setup
```

On Linux distributions that enforce PEP 668, install the launcher in an isolated
environment instead of the system Python:

```sh
python3 -m venv ~/.venvs/utopic
~/.venvs/utopic/bin/pip install git+https://github.com/adavyas/utopic-package-manager.git
~/.venvs/utopic/bin/utopic setup
```

`utopic setup` builds from package-managed native sources and installs the
runtime binaries under `~/.cache/utopic/bin`.

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

For local development from this checkout:

```sh
git clone https://github.com/adavyas/utopic-package-manager.git
cd utopic-package-manager
pip install .
utopic setup
```

## Commands

The package installs these launchers:

- `utopic`
- `utopic-server`
- `utopic-mcp`
- `utopic-acp`

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

- fetch the pinned compatible native runtime and dependency sources
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

Run the Python wrapper tests:

```sh
PYTHONPATH=python python -m unittest discover -s tests -p 'test_*.py'
```

Build a wheel:

```sh
python -m pip wheel . --no-deps -w dist/
```
