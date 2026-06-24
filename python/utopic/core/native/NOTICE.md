# Credits

`diffusion_driver.cpp`, `diffusion_driver.h`, and `main.cpp` are derived from the diffusion
example in **llama.cpp** (https://github.com/ggml-org/llama.cpp), MIT License — the denoising
loop, the entropy-bound/canvas path, the masked-absorbing schedule helpers
(`calculate_confidence`, `calculate_transfer_count`, `get_num_transfer_tokens`), and the
self-conditioning hook. We link `libllama`/`ggml` as a library (the forward + kernels + GGUF)
and own the loop on top.

Diffusion-LM arch support in llama.cpp: DiffusionGemma; LLaDA (PRs #14771, #16003); Dream (#14644).
Model authors: Google (DiffusionGemma), GSAI-ML (LLaDA), Dream-org (Dream).

This directory is the native Utopic runtime:
own the denoise loop + policies + freeze + calibration; rent ggml kernels + GGUF.
