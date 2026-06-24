# Supported Utopic Models

Utopic's packaged model catalog is intentionally DiffusionGemma-first for text
and uses one runtime schema for all local modalities. Native text models are
GGUF files driven by the C++ runtime. The native image path is backed by
stable-diffusion.cpp. TTS, music, video, and misc entries still share the same
cache, OpenAI endpoint, MCP tool, and metadata contract that native C++ engines
use. TTS now has a native Sherpa-ONNX plugin path for Kokoro-class offline
speech models; Python bridges remain as transitional debug paths where native
model engines have not landed yet.

| Model family | Modality | Runtime | Release status |
|---|---|---|---|
| DiffusionGemma | text | native GGUF / canvas entropy-bound | Curated Q4_K_M/Q5_K_M/Q6_K/Q8_0 aliases plus local GGUF paths |
| Qwen-Image | image | bridge | Cataloged behind `/v1/images/generations` and MCP image generation |
| FLUX.1-schnell | image | native stable-diffusion.cpp / bridge | Native Q4 component bundle plus bridge fallback entries behind `/v1/images/generations` and MCP image generation |
| Krea 2 Raw | image | bridge | Cataloged behind `/v1/images/generations` and MCP image generation |
| Cosmos3 Super Text2Image | image | bridge | Cataloged with GPU-memory preflight before bridge execution |
| Kokoro | tts | native Sherpa-ONNX plugin / bridge | Native C++ plugin writes local WAV artifacts through the runner plugin ABI; bridge fallback remains cataloged behind `/v1/audio/speech` and MCP speech generation |
| Chatterbox / Dia | tts | bridge | Cataloged behind `/v1/audio/speech` and MCP speech generation |
| ACE-Step | music | bridge | Cataloged behind `/v1/audio/generations` and MCP music generation |
| Wan2.1 T2V / LTX-Video | video | bridge | Cataloged behind `/v1/videos/generations` and MCP video generation |

Native non-text plugins use the public C ABI in `runner_plugin_api.h`. A plugin
exports `utopic_native_generate` by default, receives the `utopic-runner/v1`
request JSON, and writes a JSON response into the provided buffer. The runner
loads that plugin through catalog `native_library` metadata and passes the same
artifact, backend, and readiness options used by built-in native engines.

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
