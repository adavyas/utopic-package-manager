# Utopic Package Manager

Python packaging for the Utopic native runtime.

This repository is intentionally thin. It does not carry the C++ runtime source.
`pip install` only installs Python launchers. The native C++ build happens later
through `utopic setup`, which fetches pinned sources, builds them with CMake, and
caches the resulting binaries under `~/.cache/utopic/bin`.

## Install

Install the package:

```sh
git clone https://github.com/adavyas/utopic-package-manager.git
cd utopic-package-manager
pip install .
```

Build and cache the native runtime:

```sh
utopic setup
```

`utopic setup` fetches the package-managed compatible llama.cpp source, applies
the Utopic compatibility patch, builds llama.cpp, then builds the Utopic native
binaries.

For CUDA llama.cpp builds, pass:

```sh
utopic setup --cuda
```

If you are developing against a local llama.cpp checkout, it must export
Utopic's diffusion APIs, including `llama_diffusion_set_sc`,
`llama_diffusion_device_sample`, `llama_diffusion_set_phase`, and
`llama_diffusion_set_block_decode`:

```sh
utopic setup --llama-dir /path/to/compatible/llama.cpp
```

Setup will build that checkout before building Utopic. If it is already built
and you only want to rebuild the Utopic binaries:

```sh
utopic setup --llama-dir /path/to/compatible/llama.cpp --skip-llama-build
```

To make setup fetch a different llama.cpp source, use the advanced override:

```sh
UTOPIC_LLAMA_REPO=https://github.com/your-org/llama.cpp.git \
UTOPIC_LLAMA_REF=your-compatible-ref \
utopic setup
```

If you are developing against a local native Utopic checkout:

```sh
utopic setup --native-dir /path/to/Utopic --llama-dir /path/to/llama.cpp
```

## Commands

The package installs:

- `utopic`
- `utopic-server`
- `utopic-mcp`
- `utopic-acp`

Example:

```sh
utopic run -m /path/to/model.gguf -p "Answer with one word: 2+2?" -n 16
```

Server:

```sh
utopic-server -m /path/to/model.gguf --host 127.0.0.1 --port 8910 -ngl 99
```

## Development

Run the Python wrapper tests:

```sh
PYTHONPATH=python python -m unittest discover -s tests -p 'test_*.py'
```

Build a wheel:

```sh
python -m pip wheel . --no-deps -w dist/
```

The wheel is pure Python. It should not fetch or compile native code.
