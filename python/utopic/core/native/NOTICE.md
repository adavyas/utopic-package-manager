# Credits

`diffusion_driver.cpp`, `diffusion_driver.h`, and `main.cpp` are derived from the diffusion
example in **llama.cpp** (https://github.com/ggml-org/llama.cpp), MIT License — the denoising
loop, the entropy-bound/canvas path, the masked-absorbing schedule helpers
(`calculate_confidence`, `calculate_transfer_count`, `get_num_transfer_tokens`), and the
self-conditioning hook. We link `libllama`/`ggml` as a library (the forward + kernels + GGUF)
and own the loop on top.

Diffusion-LM arch support in the pinned llama.cpp dependency currently centers
the shippable DiffusionGemma GGUF path.
Model authors: Google (DiffusionGemma).

This directory is the native Utopic runtime:
own the denoise loop + policies + freeze + calibration; rent ggml kernels + GGUF.
