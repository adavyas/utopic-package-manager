# Supported Utopic Models

Utopic's packaged model catalog is intentionally DiffusionGemma-first for text
and uses one runtime schema for all local modalities. Native text models are
GGUF files driven by the C++ runtime. Image, TTS, music, video, and misc entries are
bridge runtimes today, but they share the same cache, OpenAI endpoint, MCP tool,
and metadata contract that future native engines will use.

| Model family | Modality | Runtime | Release status |
|---|---|---|---|
| DiffusionGemma | text | native GGUF / canvas entropy-bound | Curated Q4_K_M/Q5_K_M/Q6_K/Q8_0 aliases plus local GGUF paths |
| Qwen-Image | image | bridge | Cataloged behind `/v1/images/generations` and MCP image generation |
| FLUX.1-schnell | image | bridge | Cataloged behind `/v1/images/generations` and MCP image generation |
| Krea 2 Raw | image | bridge | Cataloged behind `/v1/images/generations` and MCP image generation |
| Cosmos3 Super Text2Image | image | bridge | Cataloged with GPU-memory preflight before bridge execution |
| Kokoro / Chatterbox / Dia | tts | bridge | Cataloged behind `/v1/audio/speech` and MCP speech generation |
| ACE-Step | music | bridge | Cataloged behind `/v1/audio/generations` and MCP music generation |
| Wan2.1 T2V / LTX-Video | video | bridge | Cataloged behind `/v1/videos/generations` and MCP video generation |

FP8 names are normalized from common GGUF file-name markers including `FP8`,
`F8_E4M3`, `F8_E5M2`, `E4M3`, and `E5M2`. GGUF quantized weight classes include
markers such as `Q8_0`, `Q6_K`, `Q5_*`, `Q4_*`, `Q3_*`, `Q2_K`, `IQ*`, and
`NVFP4`, subject to what the linked package-managed llama.cpp build can load on
the target backend.

The BF16 DiffusionGemma file is intentionally left to local GGUF paths because
it is about 47 GiB before runtime buffers, which is too close to the usable
memory ceiling on 48 GB Apple Silicon. The package-managed CUDA setup and native
DiffusionGemma Q4_K_M smoke path pass on GB10/DGX Spark, a 6x RTX 4090 host, and
a 4x A100 host. Q5_K_M, Q6_K, and Q8_0 smoke paths also pass on the 4x A100
host.
