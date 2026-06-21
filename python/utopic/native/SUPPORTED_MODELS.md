# Supported dLLM GGUF Models

Utopic support is architecture-based, not tied to one quantized checkpoint. If
the linked package-managed llama.cpp build can load the GGUF tensor type, Utopic
drives the same denoising path for that model family.

| Model family | GGUF architecture/name signal | Path | Supported weight classes |
|---|---|---|---|
| LLaDA | `general.architecture=llada` | masked | BF16, FP8, F16, F32, Q*/IQ* GGUF weights |
| Dream | `general.architecture=dream` | masked | BF16, FP8, F16, F32, Q*/IQ* GGUF weights |
| DiffusionGemma | `general.architecture=diffusion-gemma` | canvas / entropy-bound | BF16, FP8, F16, F32, Q*/IQ* GGUF weights |

FP8 names are normalized from common GGUF file-name markers including `FP8`,
`F8_E4M3`, `F8_E5M2`, `E4M3`, and `E5M2`. GGUF quantized weight classes include
markers such as `Q8_0`, `Q6_K`, `Q5_*`, `Q4_*`, `Q3_*`, `Q2_K`, `IQ*`, and
`NVFP4`, subject to what the linked package-managed llama.cpp build can load.
