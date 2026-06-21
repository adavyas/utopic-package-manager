# Utopic Package Manager

Python packaging for the Utopic native runtime.

This repository is intentionally thin. It does not carry the C++ runtime source.
During a pip build it fetches the pinned native Utopic repository, builds the
native binaries with CMake through `scikit-build-core`, and installs Python
console entrypoints that exec those packaged binaries.

## Install

Utopic currently links against a diffusion-capable llama.cpp checkout. Build
that dependency first, then point the package build at it:

```sh
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -B ~/llama.cpp/build -S ~/llama.cpp
cmake --build ~/llama.cpp/build -j

git clone https://github.com/adavyas/utopic-package-manager.git
cd utopic-package-manager
UTOPIC_LLAMACPP_DIR=~/llama.cpp pip install .
```

For CUDA llama.cpp builds:

```sh
cmake -B ~/llama.cpp/build -S ~/llama.cpp -DGGML_CUDA=ON
cmake --build ~/llama.cpp/build -j
```

If you are developing against a local Utopic native checkout instead of the
pinned GitHub revision:

```sh
UTOPIC_LLAMACPP_DIR=/path/to/llama.cpp \
pip install . --config-settings=cmake.define.UTOPIC_NATIVE_SOURCE_DIR=/path/to/Utopic
```

## Commands

The package installs:

- `utopic`
- `utopic-server`
- `utopic-mcp`
- `utopic-acp`

Example:

```sh
utopic -m /path/to/model.gguf -p "Answer with one word: 2+2?" -n 16
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
UTOPIC_LLAMACPP_DIR=/path/to/llama.cpp python -m pip wheel . --no-deps -w dist/
```
