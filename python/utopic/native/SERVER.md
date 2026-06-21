# Utopic runtime — CLI & OpenAI-compatible server

Local serving for diffusion LMs (LLaDA / Dream / DiffusionGemma) over GGUF, Ollama-style.
Both frontends share one generation core (`utopic_core.h`): the confidence/convergence gate, the
entropy-bound path for canvas models, schema-constrained decoding, and the tolerant tool extractor.

Supported dLLM GGUF families:

| Model family | Runtime path | Required weight classes |
|---|---|---|
| LLaDA | masked | BF16, FP8, F16, F32, Q*/IQ* GGUF weights |
| Dream | masked | BF16, FP8, F16, F32, Q*/IQ* GGUF weights |
| DiffusionGemma | canvas / entropy-bound | BF16, FP8, F16, F32, Q*/IQ* GGUF weights |

See `SUPPORTED_MODELS.md` for exact architecture/name and quantization markers.

## Setup

Install the package manager and let it build the native runtime:

```sh
pip install git+https://github.com/adavyas/utopic-package-manager.git
utopic setup
```

The package manager installs:

- `utopic` - CLI
- `utopic-server` - OpenAI-compatible server
- `utopic-mcp` - MCP stdio server
- `utopic-acp` - ACP stdio agent

## CLI (one-shot, like `ollama run`)
```
utopic run -m model.gguf -p "Name three primary colors."
# tool calling:
utopic run -m m.gguf -p "weather in Paris?" --tools "get_weather(city, unit)"
# structured output (typed JSON, hard guarantee):
utopic run -m m.gguf -p "Sam is 41." --schema '{"name":"__s8__","age":"__d4__"}'
# reasoning (prompt-level think-then-answer):
utopic run -m m.gguf -p "3 apples, eat 1, how many?" --reasoning
# system message + gate knobs:
utopic run -m m.gguf -p "..." --system "You are terse." --confidence 0.9 --converge 2
```
Schema slots: `__s__` string, `__d__` integer, `__n__` number; optional length `__s12__`/`__d6__`.
The gate (confidence 0.9 + convergence 2 + EOS-stop) is on by default — that's the shipped fast config.

## Server (resident model, like `ollama serve`)
```
utopic-server -m model.gguf --host 127.0.0.1 --port 8910 -ngl 99 --ctx-size 4096
```
Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`.

Works with any OpenAI client:
```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8910/v1", api_key="x")
c.chat.completions.create(model="local", messages=[{"role":"user","content":"2+2?"}])
```

Supported request fields:
- `messages` (system/user/assistant; tool/unknown roles fold into user)
- `stream` (SSE `chat.completion.chunk` deltas + `[DONE]`) - **diffusion-native live streaming**: the
  masked path (LLaDA/Dream) emits the committed canvas prefix as deltas while it denoises (tokens resolve
  over steps, not strictly left-to-right; deltas are additive/OpenAI-safe). The entropy-bound path
  (DiffusionGemma) streams the cleaned answer on completion.
- `temperature`, `max_tokens`, `seed`
- `tools` (OpenAI function tools) -> injected into the prompt; output harvested into
  `message.tool_calls` (`finish_reason: tool_calls`)
- `response_format`:
  - `{"type":"json_schema","json_schema":{"schema":{...}}}` -> typed-slot constrained decoding
    (flat + nested objects mapped to `__s__/__d__/__n__`; arrays fall back to a string slot)
  - `{"type":"json_object"}` -> "reply only JSON" instruction (no hard guarantee)
- `reasoning` / `reasoning_effort` -> prompt-level think-then-answer

## Notes & limits
- Generation is **serialized** (one resident context) — a local single-user server, not a fleet.
- DiffusionGemma runs here on GB10 where stock llama.cpp crashes. Its native `<|channel>thought ...
  <channel|>` reasoning markers are parsed out: the answer is returned clean in `content`, and when
  the model emits a separate thought channel it is surfaced as `message.reasoning_content` (server)
  / `[reasoning]` on stderr (CLI). The model emits the full think/answer split inconsistently, so
  `reasoning_content` is best-effort; the markers never leak into the answer.
- Tool-call argument fidelity and schema value fidelity are model-dependent (the structure/typing
  is guaranteed; the values are the model's job).
- Streaming (masked path) is live: deltas are emitted from inside the denoise loop as the canvas
  commits, so they arrive during generation. Because diffusion commits out of order, a delta is sent
  only when the committed prefix extends (additive/OpenAI-safe); non-monotonic mid-step churn isn't
  shown. The entropy-bound path (DiffusionGemma) streams its cleaned answer on completion.
