# Supported dLLM GGUF Models

Utopic support is architecture-based, not tied to one quantized checkpoint. If
the linked package-managed llama.cpp build can load the GGUF tensor type, Utopic
drives the same denoising path for that model family.

| Model family | GGUF architecture/name signal | Path | Release status |
|---|---|---|---|
| LLaDA | `general.architecture=llada` | masked | Curated Q4_K_M alias verified on CUDA |
| Dream | `general.architecture=dream` | masked | Curated Q4_K_M alias verified on Metal and CUDA |
| DiffusionGemma | `general.architecture=diffusion-gemma` | canvas / entropy-bound | Experimental GGUF path |

FP8 names are normalized from common GGUF file-name markers including `FP8`,
`F8_E4M3`, `F8_E5M2`, `E4M3`, and `E5M2`. GGUF quantized weight classes include
markers such as `Q8_0`, `Q6_K`, `Q5_*`, `Q4_*`, `Q3_*`, `Q2_K`, `IQ*`, and
`NVFP4`, subject to what the linked package-managed llama.cpp build can load on
the target backend.

DiffusionGemma is not listed in the curated Python model catalog yet. The
current GB10 CUDA validation fails inside ggml CUDA SOFT_MAX for Q4_K_M and
BF16 GGUF files, so treat that path as a runtime-development target rather than
a release-ready download alias.
