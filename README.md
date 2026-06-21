# Utopic Package Manager

Python package management for the Utopic native runtime.

This repository is intentionally thin. The wheel installs Python launchers only.
Native source checkout, compatibility patching, CMake configuration, and binary
installation all happen later through `utopic setup`.

## Install

```sh
pip install git+https://github.com/adavyas/utopic-package-manager.git
utopic setup
```

`utopic setup` builds from package-managed native sources and installs the
runtime binaries under `~/.cache/utopic/bin`.

On NVIDIA hosts, build the CUDA backend:

```sh
utopic setup --backend cuda
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

- fetch the pinned compatible native sources
- apply Utopic's compatibility overlay
- configure and build with CMake
- copy the final binaries into the Utopic cache

The published wheel stays pure Python and does not fetch or compile native code
during `pip install`.

## Development

Run the Python wrapper tests:

```sh
PYTHONPATH=python python -m unittest discover -s tests -p 'test_*.py'
```

Build a wheel:

```sh
python -m pip wheel . --no-deps -w dist/
```
