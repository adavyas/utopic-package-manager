import base64
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
import urllib.error
import urllib.request

from . import __version__, _native, bridge, installer, models


JSON_HEADERS = {"content-type": "application/json"}
_native_binary_path = _native.binary_path

HELP = """usage: utopic-runtime [options]

Start the unified Utopic OpenAI-compatible and MCP gateway.

Options:
  --host HOST          Bind host. Default: 127.0.0.1
  --port PORT          Bind port. Default: 8911
  --native-base-url URL
                       Forward native text OpenAI requests to an existing
                       utopic-server, for example http://127.0.0.1:8910
  -h, --help           Show this help.
  --version            Show version.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/responses
  POST /v1/images/generations
  POST /v1/audio/speech
  POST /v1/audio/generations
  POST /v1/videos/generations
  POST /v1/utopic/misc/generations
  POST /mcp
"""


OPENAI_TOOL_BY_ENDPOINT = {
    "/v1/chat/completions": "utopic_chat",
    "/v1/responses": "utopic_chat",
    "/v1/images/generations": "utopic_generate_image",
    "/v1/audio/speech": "utopic_speak",
    "/v1/audio/generations": "utopic_generate_music",
    "/v1/videos/generations": "utopic_generate_video",
    "/v1/utopic/misc/generations": "utopic_generate_misc",
}


def _schema(required: list[str], properties: dict[str, str | dict[str, Any]]) -> dict[str, Any]:
    normalized_properties: dict[str, dict[str, Any]] = {}
    for key, value in properties.items():
        if isinstance(value, dict):
            normalized_properties[key] = value
        else:
            normalized_properties[key] = {"type": value}
    return {
        "type": "object",
        "required": required,
        "properties": normalized_properties,
    }


EXPERIMENTAL_BRIDGE_PROPERTY = {
    "type": "boolean",
    "description": (
        "Set true only when you explicitly want to run an experimental Python bridge model. "
        "Production generation defaults to native C++ runners; omit this for native-ready models."
    ),
}


MCP_TOOLS = [
    {
        "name": "utopic_chat",
        "description": (
            "Generate a text answer locally with Utopic's OpenAI-compatible text runtime. "
            "Use this when an agent needs private/offline drafting, reasoning, summarization, "
            "classification, extraction, or coding help from the local DiffusionGemma catalog. "
            "Returns the raw OpenAI chat-completions JSON as text; call utopic_models_check first "
            "when you need to confirm the selected model is downloaded and runnable."
        ),
        "inputSchema": _schema(
            ["prompt"],
            {
                "model": {
                    "type": "string",
                    "description": "Optional catalog id, for example diffusiongemma-26b-a4b-q4. Defaults to the recommended local text model.",
                },
                "prompt": {
                    "type": "string",
                    "description": "User request to answer. Prefer complete instructions; the gateway converts this into one OpenAI user message.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Optional response budget for short local answers.",
                },
            },
        ),
    },
    {
        "name": "utopic_generate_image",
        "description": (
            "Create an image locally from a text prompt using a Utopic image model such as "
            "FLUX, Qwen-Image, Krea, or Cosmos when the bridge is installed. Use for private "
            "local image generation, concept art, product shots, UI assets, and prompt tests. "
            "Returns OpenAI-compatible image/artifact JSON with local artifact metadata."
        ),
        "inputSchema": _schema(
            ["prompt"],
            {
                "model": {
                    "type": "string",
                    "description": "Optional image model id, for example flux-1-schnell, qwen-image, krea-2-raw, or cosmos3-super.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Image prompt. Include subject, style, composition, and constraints.",
                },
                "size": {
                    "type": "string",
                    "description": "Optional output size such as 1024x1024 when supported by the model.",
                },
                "experimental_bridge": EXPERIMENTAL_BRIDGE_PROPERTY,
            },
        ),
    },
    {
        "name": "utopic_speak",
        "description": (
            "Generate local speech audio from text with Utopic TTS models. Use for narration, "
            "voice prototypes, accessibility previews, and agent-read summaries without sending "
            "text to a remote TTS API. Returns artifact JSON pointing at generated audio."
        ),
        "inputSchema": _schema(
            ["input"],
            {
                "model": {
                    "type": "string",
                    "description": "Optional TTS model id, for example kokoro-82m, chatterbox, or dia-1.6b.",
                },
                "input": {
                    "type": "string",
                    "description": "Text to synthesize into speech.",
                },
                "voice": {
                    "type": "string",
                    "description": "Optional model-specific voice name.",
                },
                "experimental_bridge": EXPERIMENTAL_BRIDGE_PROPERTY,
            },
        ),
    },
    {
        "name": "utopic_generate_music",
        "description": (
            "Generate local music audio from a text prompt. Use for private soundtrack ideas, "
            "loops, mood boards, and sonic sketches. Returns artifact JSON pointing at generated audio."
        ),
        "inputSchema": _schema(
            ["prompt"],
            {
                "model": {
                    "type": "string",
                    "description": "Optional music model id, for example ace-step-1.5.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Music prompt with genre, instrumentation, mood, tempo, and duration hints.",
                },
                "experimental_bridge": EXPERIMENTAL_BRIDGE_PROPERTY,
            },
        ),
    },
    {
        "name": "utopic_generate_video",
        "description": (
            "Generate local video from a text prompt with Utopic video models. Use for short "
            "text-to-video clips, motion studies, and prototype visuals. Some video models require "
            "GB10 or high-memory CUDA; use utopic_models_check before running large jobs."
        ),
        "inputSchema": _schema(
            ["prompt"],
            {
                "model": {
                    "type": "string",
                    "description": "Optional video model id, for example wan2.1-t2v-1.3b, wan2.1-t2v-14b, or ltx-video.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Video prompt with subject, motion, camera, style, and duration hints.",
                },
                "size": {
                    "type": "string",
                    "description": "Optional output size such as 832x480 when supported by the model.",
                },
                "experimental_bridge": EXPERIMENTAL_BRIDGE_PROPERTY,
            },
        ),
    },
    {
        "name": "utopic_generate_misc",
        "description": (
            "Run a local miscellaneous artifact workflow such as ZUNA signal processing. "
            "Use for local file-in/file-out models that are not text, image, speech, music, "
            "or video yet. Returns artifact JSON with generated local files."
        ),
        "inputSchema": _schema(
            ["artifact"],
            {
                "model": {
                    "type": "string",
                    "description": "Optional misc model id, for example zuna.",
                },
                "artifact": {
                    "type": "string",
                    "description": "Path to a local input artifact file.",
                },
                "artifact_type": {
                    "type": "string",
                    "description": "Optional MIME type for the input artifact.",
                },
                "experimental_bridge": EXPERIMENTAL_BRIDGE_PROPERTY,
            },
        ),
    },
    {
        "name": "utopic_models_list",
        "description": (
            "List every Utopic catalog model with modality, runtime, endpoints, outputs, "
            "hardware hints, and bridge setup metadata. Agents should call this before choosing "
            "a model for text, image, TTS, music, video, or misc generation."
        ),
        "inputSchema": _schema([], {}),
    },
    {
        "name": "utopic_models_check",
        "description": (
            "Check whether one model or the whole catalog is ready on this machine. Reports "
            "download/cache state, bridge dependency gaps, hardware requirements, and local "
            "runtime readiness so agents can avoid doomed OOM or missing-dependency calls."
        ),
        "inputSchema": _schema(
            [],
            {
                "model": {
                    "type": "string",
                    "description": "Catalog id to check, for example diffusiongemma-26b-a4b-q4 or flux-1-schnell.",
                },
                "all": {
                    "type": "boolean",
                    "description": "When true, check every catalog model and summarize readiness.",
                },
            },
        ),
    },
    {
        "name": "utopic_models_pull",
        "description": (
            "Download or prepare a Utopic catalog model for local use. Use this when an agent "
            "has permission to fetch weights or set up bridge model metadata. Supports one model "
            "or all=true; returns paths and setup results."
        ),
        "inputSchema": _schema(
            [],
            {
                "model": {
                    "type": "string",
                    "description": "Catalog id to download or prepare.",
                },
                "all": {
                    "type": "boolean",
                    "description": "When true, prepare every catalog model. Do not combine with model.",
                },
                "force": {
                    "type": "boolean",
                    "description": "When true, redownload or refresh cached model metadata.",
                },
            },
        ),
    },
]


def model_cache_path(model_id: str) -> Path:
    entry = models.get_model(model_id)
    if entry is None:
        return models.models_dir() / model_id
    return entry.path


def handle_openai_request(
    method: str,
    path: str,
    body: Optional[dict[str, Any]],
    *,
    native_base_url: Optional[str] = None,
) -> tuple[int, dict[str, str], bytes]:
    if method == "GET" and path == "/health":
        return _json(200, {"status": "ok", "version": __version__})
    if method == "GET" and path == "/v1/models":
        return _json(200, {"object": "list", "data": [_model_payload(entry) for entry in models.list_models()]})
    if method == "GET" and path.startswith("/v1/utopic/runs/") and path.endswith("/events"):
        run_id = path.removeprefix("/v1/utopic/runs/").removesuffix("/events").strip("/")
        return _run_progress_response(run_id)
    if method != "POST":
        return _json(404, {"error": {"message": f"unknown route: {method} {path}", "code": "not_found"}})
    if path not in OPENAI_TOOL_BY_ENDPOINT:
        return _json(404, {"error": {"message": f"unknown route: {path}", "code": "not_found"}})
    request = body or {}
    model_id = str(request.get("model") or _default_model_for_endpoint(path))
    entry = models.get_model(model_id)
    if entry is None:
        return _json(404, {"error": {"message": f"unknown model: {model_id}", "code": "model_not_found"}})
    if path not in entry.endpoints:
        return _json(
            400,
            {
                "error": {
                    "message": f"model {entry.id} does not support {path}",
                    "code": "endpoint_not_supported",
                    "model": entry.id,
                    "modality": entry.modality,
                }
            },
        )
    runtime_request = _normalize_request_for_runtime(path, entry, request)
    if entry.runtime == "bridge":
        if not _experimental_bridge_enabled(runtime_request):
            return _bridge_requires_explicit_opt_in(entry, path)
        preflight = _bridge_runtime_preflight(entry)
        if preflight is not None:
            return preflight
        command = _bridge_command(entry)
        if command is None:
            return _bridge_not_installed(entry, path)
        return _run_bridge(entry, path, runtime_request, command)
    if entry.modality != "text":
        return _run_native_runner(entry, path, runtime_request)
    if native_base_url and entry.modality == "text":
        native_path = "/v1/chat/completions" if path == "/v1/responses" else path
        status, headers, response_body = _forward_native_text(native_base_url, native_path, runtime_request)
        if path == "/v1/responses" and status < 400:
            return _json(status, _native_chat_to_response(entry, response_body))
        return status, headers, response_body
    return _run_native_runner(entry, path, runtime_request)


def _experimental_bridge_enabled(request: dict[str, Any]) -> bool:
    return request.get("experimental_bridge") is True or request.get("allow_experimental_bridge") is True


def _bridge_requires_explicit_opt_in(entry: models.ModelEntry, endpoint: str) -> tuple[int, dict[str, str], bytes]:
    return _json(
        501,
        {
            "error": {
                "message": (
                    f"model {entry.id} is cataloged as an experimental bridge model. "
                    "Set experimental_bridge=true to run the Python bridge explicitly; "
                    "native runners are the default production path."
                ),
                "code": "native_runner_not_ready",
                "model": entry.id,
                "modality": entry.modality,
                "engine": entry.engine,
                "runtime": entry.runtime,
                "native_status": entry.native_status,
            },
            "native": {
                "runner": entry.runner,
                "native_status": entry.native_status,
                "supported_backends": list(entry.supported_backends),
                "endpoint": endpoint,
            },
            "bridge": _bridge_contract(entry, endpoint),
        },
    )


def handle_mcp_request(
    request: dict[str, Any],
    *,
    native_base_url: Optional[str] = None,
) -> tuple[int, dict[str, str], bytes]:
    method = request.get("method")
    request_id = request.get("id")
    if method == "initialize":
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        protocol_version = params.get("protocolVersion")
        if not isinstance(protocol_version, str) or not protocol_version:
            protocol_version = "2025-06-18"
        return _json(
            200,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "utopic-runtime", "version": __version__},
                },
            },
        )
    if method == "notifications/initialized":
        return _json(200, {"jsonrpc": "2.0", "id": request_id, "result": {}})
    if method == "ping":
        return _json(200, {"jsonrpc": "2.0", "id": request_id, "result": {}})
    if method == "tools/list":
        return _json(200, {"jsonrpc": "2.0", "id": request_id, "result": {"tools": MCP_TOOLS}})
    if method == "tools/call":
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        return _json(200, _mcp_tool_call(request_id, str(name), arguments, native_base_url=native_base_url))
    return _json(
        200,
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"unknown method: {method}"},
        },
    )


def _mcp_tool_call(
    request_id: Any,
    name: str,
    arguments: dict[str, Any],
    *,
    native_base_url: Optional[str],
) -> dict[str, Any]:
    if name == "utopic_models_list":
        content = json.dumps([_model_payload(entry) for entry in models.list_models()], indent=2)
        return _mcp_text_result(request_id, content, is_error=False)
    if name == "utopic_models_check":
        try:
            if arguments.get("all") is True:
                payload = models._all_model_checks()
            else:
                model_id = str(arguments.get("model") or "")
                if not model_id:
                    raise RuntimeError("utopic_models_check requires model or all=true")
                payload = models.model_check(model_id)
        except RuntimeError as exc:
            return _mcp_text_result(request_id, str(exc), is_error=True)
        return _mcp_text_result(
            request_id,
            json.dumps(payload, indent=2, sort_keys=True),
            is_error=not bool(payload.get("ready")),
        )
    if name == "utopic_models_pull":
        model_id = str(arguments.get("model") or "")
        try:
            if arguments.get("all") is True:
                if model_id:
                    raise RuntimeError("pull accepts either a model alias or all=true, not both")
                payload = models._pull_all_models(force=bool(arguments.get("force", False)))
                return _mcp_text_result(
                    request_id,
                    json.dumps(payload, indent=2, sort_keys=True),
                    is_error=False,
                )
            if not model_id:
                raise RuntimeError("utopic_models_pull requires model or all=true")
            path = models.pull_model(model_id, force=bool(arguments.get("force", False)))
        except RuntimeError as exc:
            return _mcp_text_result(request_id, str(exc), is_error=True)
        return _mcp_text_result(request_id, str(path), is_error=False)
    endpoint_by_tool = {
        "utopic_chat": "/v1/chat/completions",
        "utopic_generate_image": "/v1/images/generations",
        "utopic_speak": "/v1/audio/speech",
        "utopic_generate_music": "/v1/audio/generations",
        "utopic_generate_video": "/v1/videos/generations",
        "utopic_generate_misc": "/v1/utopic/misc/generations",
    }
    endpoint = endpoint_by_tool.get(name)
    if endpoint is None:
        return _mcp_text_result(request_id, f"unknown tool: {name}", is_error=True)
    arguments = _normalize_mcp_tool_arguments(name, arguments)
    status, _headers, body = handle_openai_request(
        "POST",
        endpoint,
        arguments,
        native_base_url=native_base_url,
    )
    payload = json.loads(body.decode("utf-8"))
    return _mcp_text_result(request_id, json.dumps(payload, sort_keys=True), is_error=status >= 400)


def _normalize_mcp_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "utopic_chat" or "messages" in arguments:
        return arguments
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str):
        return arguments
    normalized = {key: value for key, value in arguments.items() if key != "prompt"}
    normalized["messages"] = [{"role": "user", "content": prompt}]
    return normalized


def _mcp_text_result(request_id: Any, text: str, *, is_error: bool) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "isError": is_error,
            "content": [{"type": "text", "text": text}],
        },
    }


def _default_model_for_endpoint(path: str) -> str:
    for entry in models.list_models():
        if entry.recommended and path in entry.endpoints:
            return entry.id
    for entry in models.list_models():
        if path in entry.endpoints:
            return entry.id
    return models.default_model().id


def _bridge_not_installed(entry: models.ModelEntry, endpoint: str) -> tuple[int, dict[str, str], bytes]:
    return _json(
        501,
        {
            "error": {
                "message": f"{entry.engine} bridge for {entry.id} is not installed yet",
                "code": "bridge_engine_not_installed",
                "model": entry.id,
                "modality": entry.modality,
                "engine": entry.engine,
            },
            "contract": _bridge_contract(entry, endpoint),
        },
    )


def _bridge_contract(entry: models.ModelEntry, endpoint: str) -> dict[str, Any]:
    input_key = _input_key_for_modality(entry.modality)
    first_output = entry.outputs[0] if entry.outputs else "application/octet-stream"
    return {
        "schema_version": "utopic-bridge/v1",
        "input": input_key,
        "outputs": list(entry.outputs),
        "cache_path": str(entry.path),
        "repo": entry.repo,
        "requirements": entry.requirements or {},
        "environment_variable": _bridge_command_env_var(entry),
        "request_schema": {
            "endpoint": endpoint,
            "input": input_key,
            "repo": entry.repo,
            "model_cache_path": str(entry.path),
            "output_dir": "<run-dir>/outputs",
            "progress_path": "<run-dir>/progress.jsonl",
        },
        "artifact_schema": {
            "type": first_output,
            "path": "<absolute-output-path>",
            "metadata": {},
        },
        "progress_event_schema": {
            "event": "queued|loading|generating|completed|failed",
            "progress": 0.0,
            "message": "human-readable status",
        },
    }


def _bridge_runtime_preflight(entry: models.ModelEntry) -> Optional[tuple[int, dict[str, str], bytes]]:
    requirements = entry.requirements or {}
    minimum = requirements.get("min_gpu_memory_gib")
    allow_cpu = requirements.get("allow_cpu", True)
    if minimum is None and allow_cpu is not False:
        return None
    if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
        return None

    detected = _detect_runtime_capacity()
    detected_memory = detected.get("gpu_memory_gib")
    has_enough_gpu = (
        isinstance(detected_memory, (int, float))
        and not isinstance(detected_memory, bool)
        and detected_memory >= float(minimum)
    )
    if has_enough_gpu:
        return None
    if allow_cpu is not False and detected.get("backend") == "cpu":
        return None

    detected_text = _detected_runtime_text(detected)
    return _json(
        507,
        {
            "error": {
                "message": (
                    f"model {entry.id} requires at least {minimum:g} GiB GPU memory; "
                    f"detected {detected_text}. This model is too large for this host."
                ),
                "code": "bridge_model_oom_preflight",
                "model": entry.id,
                "modality": entry.modality,
                "engine": entry.engine,
                "required_gpu_memory_gib": minimum,
                "detected": detected,
                "next_steps": [
                    "Use GB10 or high-memory CUDA infrastructure.",
                    "Choose a smaller image model such as krea-2-raw, qwen-image, or flux-1-schnell.",
                ],
            }
        },
    )


def _detected_runtime_text(detected: dict[str, Any]) -> str:
    device = detected.get("device") if isinstance(detected.get("device"), str) else "unknown device"
    memory = detected.get("gpu_memory_gib")
    if isinstance(memory, (int, float)) and not isinstance(memory, bool):
        return f"{device} with {memory:.1f} GiB GPU memory"
    return device


def _detect_runtime_capacity() -> dict[str, Any]:
    configured_memory = _float_env("UTOPIC_GPU_MEMORY_GIB")
    if configured_memory is not None:
        return {
            "backend": os.environ.get("UTOPIC_RUNTIME_BACKEND", "configured"),
            "device": os.environ.get("UTOPIC_RUNTIME_DEVICE", "configured runtime"),
            "gpu_memory_gib": configured_memory,
        }

    cuda = _detect_cuda_capacity()
    if cuda is not None:
        return cuda

    if sys.platform == "darwin":
        memory = _darwin_unified_memory_gib()
        return {
            "backend": "metal",
            "device": _darwin_device_name(),
            "gpu_memory_gib": memory * 0.84 if memory is not None else None,
            "unified_memory_gib": memory,
        }

    return {"backend": "cpu", "device": "CPU", "gpu_memory_gib": None}


def _float_env(name: str) -> Optional[float]:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _detect_cuda_capacity() -> Optional[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    names: list[str] = []
    total_mib = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        names.append(parts[0])
        try:
            total_mib += float(parts[1])
        except ValueError:
            continue
    if total_mib <= 0:
        return None
    return {
        "backend": "cuda",
        "device": ", ".join(names) if names else "CUDA",
        "gpu_memory_gib": total_mib / 1024.0,
        "gpu_count": len(names),
    }


def _darwin_unified_memory_gib() -> Optional[float]:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip()) / (1024.0 ** 3)
    except ValueError:
        return None


def _darwin_device_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Apple Silicon"
    name = result.stdout.strip()
    return name or "Apple Silicon"


def _run_bridge(
    entry: models.ModelEntry,
    endpoint: str,
    request: dict[str, Any],
    command: list[str],
) -> tuple[int, dict[str, str], bytes]:
    run_id = "run_" + uuid.uuid4().hex
    run_dir = _runs_dir() / run_id
    output_dir = run_dir / "outputs"
    progress_path = run_dir / "progress.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "utopic-bridge/v1",
        "run_id": run_id,
        "endpoint": endpoint,
        "model": entry.id,
        "repo": entry.repo,
        "name": entry.name,
        "family": entry.family,
        "modality": entry.modality,
        "engine": entry.engine,
        "input": _bridge_input(entry, request),
        "parameters": {
            key: value
            for key, value in request.items()
            if key not in {"model", "prompt", "input", "messages", "artifact", "input_file"}
        },
        "model_cache_path": str(entry.path),
        "output_dir": str(output_dir),
        "progress_path": str(progress_path),
        "metadata": {
            "outputs": list(entry.outputs),
            "hardware": list(entry.hardware),
            "repo": entry.repo,
            "url": entry.url,
        },
    }
    (run_dir / "request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        result = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=_bridge_timeout_seconds(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _bridge_failed(entry, str(exc), run_id=run_id, progress_path=progress_path)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        return _bridge_failed(entry, stderr, run_id=run_id, progress_path=progress_path)
    try:
        bridge_payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return _bridge_failed(entry, f"invalid bridge JSON: {exc}", run_id=run_id, progress_path=progress_path)
    if isinstance(bridge_payload, dict) and isinstance(bridge_payload.get("error"), dict):
        return _bridge_adapter_error(entry, bridge_payload["error"], run_id=run_id, progress_path=progress_path)

    artifacts, artifact_error = _normalize_artifacts(bridge_payload.get("artifacts"), entry, output_dir)
    if artifact_error:
        return _bridge_failed(entry, artifact_error, run_id=run_id, progress_path=progress_path)
    if not artifacts:
        return _bridge_failed(entry, "bridge returned no artifacts", run_id=run_id, progress_path=progress_path)
    progress = _read_progress(progress_path)
    response = {
        "id": run_id,
        "object": "utopic.artifact.response",
        "created": int(time.time()),
        "model": entry.id,
        "modality": entry.modality,
        "engine": entry.engine,
        "artifacts": artifacts,
        "progress": progress,
        "progress_url": f"/v1/utopic/runs/{run_id}/events",
        "metadata": bridge_payload.get("metadata") if isinstance(bridge_payload.get("metadata"), dict) else {},
    }
    if entry.modality == "image":
        response["data"] = _image_generation_data(artifacts, request)
    if endpoint == "/v1/responses":
        return _json(200, _artifact_response_to_responses(entry, response))
    return _json(200, response)


def _run_native_runner(
    entry: models.ModelEntry,
    endpoint: str,
    request: dict[str, Any],
) -> tuple[int, dict[str, str], bytes]:
    if entry.runner == "utopic_ace" or entry.engine == "ace-step":
        return _run_native_ace_step(entry, endpoint, request)
    if entry.runner == "utopic_hidream_o1" or entry.engine == "hidream-o1":
        return _run_native_hidream_o1(entry, endpoint, request)

    run_id = "run_" + uuid.uuid4().hex
    run_dir = _runs_dir() / run_id
    output_dir = run_dir / "outputs"
    progress_path = run_dir / "progress.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "utopic-runner/v1",
        "run_id": run_id,
        "task": "chat" if entry.modality == "text" else entry.modality,
        "model": entry.id,
        "input": _native_runner_input(entry, request),
        "options": _native_runner_options(entry, request),
        "output_dir": str(output_dir),
        "progress_path": str(progress_path),
    }
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    try:
        result = subprocess.run(
            [str(_native_binary_path("utopic_runner")), "--json-request", str(request_path)],
            text=True,
            capture_output=True,
            check=False,
            timeout=_runner_timeout_seconds(),
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return _native_runner_failed(entry, str(exc), run_id=run_id, progress_path=progress_path)
    runner_payload, parse_error = _parse_runner_json(result.stdout)
    if result.returncode != 0:
        if isinstance(runner_payload, dict) and isinstance(runner_payload.get("error"), dict):
            return _native_runner_error(entry, runner_payload["error"], run_id=run_id, progress_path=progress_path)
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        return _native_runner_failed(entry, message, run_id=run_id, progress_path=progress_path)
    if parse_error:
        return _native_runner_failed(entry, parse_error, run_id=run_id, progress_path=progress_path)
    if not isinstance(runner_payload, dict) or runner_payload.get("ok") is not True:
        return _native_runner_failed(entry, "native runner returned an invalid response", run_id=run_id, progress_path=progress_path)

    if entry.modality == "text":
        return _native_text_runner_response(entry, endpoint, runner_payload, run_id=run_id, progress_path=progress_path)

    artifacts, artifact_error = _normalize_artifacts(runner_payload.get("artifacts"), entry, output_dir)
    if artifact_error:
        return _native_runner_failed(entry, artifact_error, run_id=run_id, progress_path=progress_path)
    if not artifacts:
        return _native_runner_failed(entry, "native runner returned no artifacts", run_id=run_id, progress_path=progress_path)
    response = {
        "id": run_id,
        "object": "utopic.artifact.response",
        "created": int(time.time()),
        "model": entry.id,
        "modality": entry.modality,
        "engine": entry.engine,
        "artifacts": artifacts,
        "progress": _read_progress(progress_path),
        "progress_url": f"/v1/utopic/runs/{run_id}/events",
        "metadata": {
            "backend": runner_payload.get("backend"),
            "device": runner_payload.get("device"),
            "metrics": runner_payload.get("metrics") if isinstance(runner_payload.get("metrics"), dict) else {},
            "runtime": entry.runtime,
        },
    }
    if entry.modality == "image":
        response["data"] = _image_generation_data(artifacts, request)
    if endpoint == "/v1/responses":
        return _json(200, _artifact_response_to_responses(entry, response))
    return _json(200, response)


def _run_native_ace_step(
    entry: models.ModelEntry,
    endpoint: str,
    request: dict[str, Any],
) -> tuple[int, dict[str, str], bytes]:
    run_id = "run_" + uuid.uuid4().hex
    run_dir = _runs_dir() / run_id
    output_dir = run_dir / "outputs"
    progress_path = run_dir / "progress.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = str(request.get("prompt") or "")
    if not prompt:
        return _native_runner_failed(entry, "prompt is required", run_id=run_id, progress_path=progress_path)
    output_path = output_dir / "music.wav"
    try:
        native_ace = str(_native_binary_path("utopic_ace"))
    except RuntimeError as exc:
        return _native_runner_failed(entry, str(exc), run_id=run_id, progress_path=progress_path)
    command = [
        native_ace,
        "--prompt",
        prompt,
        "--out",
        str(output_path),
        "--models",
        str(entry.path),
    ]
    duration = request.get("seconds", request.get("duration"))
    if duration is not None:
        command.extend(["--seconds", str(duration)])
    steps = request.get("steps", request.get("num_inference_steps", request.get("inference_steps")))
    if steps is not None:
        command.extend(["--steps", str(steps)])
    seed = request.get("seed")
    if seed is not None:
        command.extend(["--seed", str(seed)])
    lyrics = request.get("lyrics")
    if isinstance(lyrics, str):
        command.extend(["--lyrics", lyrics])

    request_path = run_dir / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "utopic-native-ace/v1",
                "run_id": run_id,
                "model": entry.id,
                "input": {"prompt": prompt},
                "parameters": {
                    key: value
                    for key, value in request.items()
                    if key not in {"model", "prompt", "input", "messages", "response_format"}
                },
                "output_dir": str(output_dir),
                "progress_path": str(progress_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=_runner_timeout_seconds(),
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return _native_runner_failed(entry, str(exc), run_id=run_id, progress_path=progress_path)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        return _native_runner_failed(entry, message, run_id=run_id, progress_path=progress_path)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return _native_runner_failed(entry, "native ACE runner produced no music.wav", run_id=run_id, progress_path=progress_path)

    artifact = {
        "type": "audio/wav",
        "path": str(output_path),
        "url": f"file://{output_path}",
        "metadata": {"runner": "utopic_ace"},
    }
    response = {
        "id": run_id,
        "object": "utopic.artifact.response",
        "created": int(time.time()),
        "model": entry.id,
        "modality": entry.modality,
        "engine": entry.engine,
        "artifacts": [artifact],
        "progress": _read_progress(progress_path),
        "progress_url": f"/v1/utopic/runs/{run_id}/events",
        "metadata": {
            "backend": "native",
            "device": None,
            "metrics": {},
            "runtime": entry.runtime,
            "runner": "utopic_ace",
        },
    }
    if endpoint == "/v1/responses":
        return _json(200, _artifact_response_to_responses(entry, response))
    return _json(200, response)


def _run_native_hidream_o1(
    entry: models.ModelEntry,
    endpoint: str,
    request: dict[str, Any],
) -> tuple[int, dict[str, str], bytes]:
    run_id = "run_" + uuid.uuid4().hex
    run_dir = _runs_dir() / run_id
    output_dir = run_dir / "outputs"
    progress_path = run_dir / "progress.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = str(request.get("prompt") or "")
    if not prompt:
        return _native_runner_failed(entry, "prompt is required", run_id=run_id, progress_path=progress_path)
    output_path = output_dir / "image.png"
    try:
        native_hidream = str(_native_binary_path("utopic_hidream_o1"))
    except RuntimeError as exc:
        return _native_runner_failed(entry, str(exc), run_id=run_id, progress_path=progress_path)
    model_dir = request.get("model_dir") or os.environ.get("UTOPIC_HIDREAM_MODEL_DIR")
    if not isinstance(model_dir, str) or not model_dir:
        model_dir = str(models._native_artifact_cache_path(entry, entry.filename).parent)
    command = [
        native_hidream,
        "--prompt",
        prompt,
        "--out",
        str(output_path),
        "--model-dir",
        model_dir,
    ]
    for request_key, flag in (
        ("width", "--width"),
        ("height", "--height"),
        ("steps", "--steps"),
        ("seed", "--seed"),
        ("source_dir", "--source-dir"),
        ("torch_python", "--torch-python"),
    ):
        value = request.get(request_key)
        if value is not None:
            command.extend([flag, str(value)])
    extra_args = request.get("extra_args")
    if isinstance(extra_args, str) and extra_args:
        command.extend(["--extra-args", extra_args])

    request_path = run_dir / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "utopic-native-hidream-o1/v1",
                "run_id": run_id,
                "model": entry.id,
                "input": {"prompt": prompt},
                "parameters": {
                    key: value
                    for key, value in request.items()
                    if key not in {"model", "prompt", "input", "messages", "response_format"}
                },
                "model_dir": model_dir,
                "output_dir": str(output_dir),
                "progress_path": str(progress_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=_runner_timeout_seconds(),
        )
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return _native_runner_failed(entry, str(exc), run_id=run_id, progress_path=progress_path)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        return _native_runner_failed(entry, message, run_id=run_id, progress_path=progress_path)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return _native_runner_failed(entry, "native HiDream-O1 runner produced no image.png", run_id=run_id, progress_path=progress_path)

    artifact = {
        "type": "image/png",
        "path": str(output_path),
        "url": f"file://{output_path}",
        "metadata": {"runner": "utopic_hidream_o1"},
    }
    response = {
        "id": run_id,
        "object": "utopic.artifact.response",
        "created": int(time.time()),
        "model": entry.id,
        "modality": entry.modality,
        "engine": entry.engine,
        "artifacts": [artifact],
        "data": _image_generation_data([artifact], request),
        "progress": _read_progress(progress_path),
        "progress_url": f"/v1/utopic/runs/{run_id}/events",
        "metadata": {
            "backend": "native",
            "device": None,
            "metrics": {},
            "runtime": entry.runtime,
            "runner": "utopic_hidream_o1",
        },
    }
    if endpoint == "/v1/responses":
        return _json(200, _artifact_response_to_responses(entry, response))
    return _json(200, response)


def _native_runner_input(entry: models.ModelEntry, request: dict[str, Any]) -> dict[str, Any]:
    if entry.modality == "text":
        input_payload: dict[str, Any] = {}
        for key in ("messages", "prompt", "system"):
            if key in request:
                input_payload[key] = request[key]
        return input_payload
    return _bridge_input(entry, request)


def _native_runner_options(entry: models.ModelEntry, request: dict[str, Any]) -> dict[str, Any]:
    excluded = {"model", "prompt", "input", "messages", "artifact", "input_file", "response_format"}
    options = {key: value for key, value in request.items() if key not in excluded}
    artifact_paths = _native_artifact_paths(entry)
    options.update(
        {
            "model_path": str(entry.path),
            "modality": entry.modality,
            "engine": entry.engine,
            "runtime": entry.runtime,
            "runner": "utopic_runner",
            "catalog_runner": entry.runner,
            "native_status": entry.native_status,
            "supported_backends": list(entry.supported_backends),
            "artifact_filenames": list(entry.artifact_filenames),
            "expected_vram_gib": entry.expected_vram_gib,
            "expected_ram_gib": entry.expected_ram_gib,
            "oom_policy": entry.oom_policy,
            "hardware": list(entry.hardware),
            "outputs": list(entry.outputs),
            "repo": entry.repo,
            "url": entry.url,
        }
    )
    if artifact_paths:
        options["artifact_paths"] = artifact_paths
        _apply_native_image_artifact_paths(options, artifact_paths)
        _apply_native_sherpa_tts_artifact_paths(options, entry, artifact_paths)
    if entry.requirements:
        options["requirements"] = entry.requirements
    if entry.native_library:
        options["native_library_path"] = str(_native_library_path(entry.native_library))
    if entry.native_entrypoint:
        options["native_entrypoint"] = entry.native_entrypoint
    return options


def _native_library_path(native_library: str) -> Path:
    path = Path(native_library).expanduser()
    path = _native_shared_library_path(path)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return installer.bin_dir() / path


def _native_shared_library_path(path: Path) -> Path:
    if path.suffix:
        return path
    if sys.platform == "darwin":
        suffix = ".dylib"
    elif os.name == "nt":
        suffix = ".dll"
    else:
        suffix = ".so"
    return path.with_name(path.name + suffix)


def _native_text_runner_response(
    entry: models.ModelEntry,
    endpoint: str,
    runner_payload: dict[str, Any],
    *,
    run_id: str,
    progress_path: Path,
) -> tuple[int, dict[str, str], bytes]:
    text = runner_payload.get("text") if isinstance(runner_payload.get("text"), str) else ""
    metrics = runner_payload.get("metrics") if isinstance(runner_payload.get("metrics"), dict) else {}
    prompt_tokens = _int_metric(metrics, "prompt_tokens")
    completion_tokens = _int_metric(metrics, "answer_tokens")
    chat_payload = {
        "id": run_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": entry.id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "metadata": {
            "backend": runner_payload.get("backend"),
            "device": runner_payload.get("device"),
            "metrics": metrics,
            "runtime": entry.runtime,
            "engine": entry.engine,
            "progress": _read_progress(progress_path),
            "progress_url": f"/v1/utopic/runs/{run_id}/events",
        },
    }
    reasoning = runner_payload.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        chat_payload["metadata"]["reasoning"] = reasoning
    if endpoint == "/v1/responses":
        return _json(200, _native_chat_to_response(entry, json.dumps(chat_payload).encode("utf-8")))
    return _json(200, chat_payload)


def _int_metric(metrics: dict[str, Any], key: str) -> int:
    value = metrics.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value >= 0:
        return int(value)
    return 0


def _native_artifact_paths(entry: models.ModelEntry) -> dict[str, str]:
    if entry.runtime != "native" or entry.modality == "text":
        return {}
    if not entry.artifact_filenames:
        return {}
    if len(entry.artifact_filenames) == 1 and not entry.native_library:
        return {entry.artifact_filenames[0]: str(entry.path)}
    base_dir = models.models_dir() / entry.id
    return {filename: str(base_dir / filename) for filename in entry.artifact_filenames}


def _apply_native_image_artifact_paths(options: dict[str, Any], artifact_paths: dict[str, str]) -> None:
    if options.get("modality") != "image":
        return
    for filename, path in artifact_paths.items():
        role = _native_image_artifact_role(filename)
        if role:
            if role == "diffusion_model_path" and filename.lower().endswith(".gguf"):
                options[role] = path
            else:
                options.setdefault(role, path)


def _native_image_artifact_role(filename: str) -> Optional[str]:
    normalized = filename.lower().replace("-", "_")
    if "clip_l" in normalized or "text_encoder" in normalized:
        return "clip_l_path"
    if "clip_g" in normalized:
        return "clip_g_path"
    if "t5xxl" in normalized or normalized.startswith("t5_") or "_t5_" in normalized:
        return "t5xxl_path"
    if "vae" in normalized:
        return "vae_path"
    if normalized.endswith(".gguf"):
        return "diffusion_model_path"
    if "diffusion" in normalized or "unet" in normalized or "model" in normalized:
        return "diffusion_model_path"
    return None


def _apply_native_sherpa_tts_artifact_paths(
    options: dict[str, Any],
    entry: models.ModelEntry,
    artifact_paths: dict[str, str],
) -> None:
    if entry.modality != "tts" or entry.engine != "sherpa-onnx":
        return
    if "model.onnx" in artifact_paths:
        options["model_path"] = artifact_paths["model.onnx"]
    if "voices.bin" in artifact_paths:
        options["voices_path"] = artifact_paths["voices.bin"]
    if "tokens.txt" in artifact_paths:
        options["tokens_path"] = artifact_paths["tokens.txt"]
    if "espeak-ng-data" in artifact_paths:
        options["data_dir"] = artifact_paths["espeak-ng-data"]


def _parse_runner_json(stdout: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    for line in reversed(stdout.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "native runner did not return JSON"


def _native_runner_failed(
    entry: models.ModelEntry,
    message: str,
    *,
    run_id: Optional[str],
    progress_path: Optional[Path],
) -> tuple[int, dict[str, str], bytes]:
    payload: dict[str, Any] = {
        "message": f"native runner failed for {entry.id}: {message}",
        "code": "native_runner_failed",
        "model": entry.id,
        "modality": entry.modality,
        "engine": entry.engine,
    }
    _attach_bridge_run_context(payload, run_id=run_id, progress_path=progress_path)
    return _json(502, {"error": payload})


def _native_runner_error(
    entry: models.ModelEntry,
    error: dict[str, Any],
    *,
    run_id: Optional[str],
    progress_path: Optional[Path],
) -> tuple[int, dict[str, str], bytes]:
    code = error.get("code") if isinstance(error.get("code"), str) else "native_runner_failed"
    status = 507 if code == "oom" else 501 if code == "unsupported_model" else 404 if code == "missing_model" else 502
    payload: dict[str, Any] = {
        "code": code,
        "message": error.get("message") if isinstance(error.get("message"), str) else "native runner failed",
        "model": entry.id,
        "modality": entry.modality,
        "engine": entry.engine,
        "detail": error.get("detail") if isinstance(error.get("detail"), dict) else {},
    }
    _attach_bridge_run_context(payload, run_id=run_id, progress_path=progress_path)
    return _json(status, {"error": payload})


def _image_generation_data(artifacts: list[dict[str, Any]], request: dict[str, Any]) -> list[dict[str, str]]:
    if request.get("response_format") != "b64_json":
        return [{"url": artifact["url"]} for artifact in artifacts if isinstance(artifact.get("url"), str)]

    data: list[dict[str, str]] = []
    for artifact in artifacts:
        path = artifact.get("path")
        if not isinstance(path, str):
            continue
        try:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        except OSError:
            continue
        data.append({"b64_json": encoded})
    return data


def _bridge_failed(
    entry: models.ModelEntry,
    message: str,
    *,
    run_id: Optional[str] = None,
    progress_path: Optional[Path] = None,
) -> tuple[int, dict[str, str], bytes]:
    payload: dict[str, Any] = {
        "message": f"{entry.engine} bridge failed for {entry.id}: {message}",
        "code": "bridge_engine_failed",
        "model": entry.id,
        "modality": entry.modality,
        "engine": entry.engine,
    }
    _attach_bridge_run_context(payload, run_id=run_id, progress_path=progress_path)
    return _json(
        502,
        {"error": payload},
    )


def _bridge_adapter_error(
    entry: models.ModelEntry,
    error: dict[str, Any],
    *,
    run_id: Optional[str] = None,
    progress_path: Optional[Path] = None,
) -> tuple[int, dict[str, str], bytes]:
    code = error.get("code") if isinstance(error.get("code"), str) else "bridge_engine_failed"
    status = 501 if code in {"bridge_dependency_missing", "bridge_adapter_not_implemented"} else 502
    payload = {
        "code": code,
        "message": error.get("message") if isinstance(error.get("message"), str) else "bridge adapter failed",
        "engine": error.get("engine") if isinstance(error.get("engine"), str) else entry.engine,
        "install_hint": error.get("install_hint") if isinstance(error.get("install_hint"), str) else "",
        "model": entry.id,
        "modality": entry.modality,
    }
    _attach_bridge_run_context(payload, run_id=run_id, progress_path=progress_path)
    return _json(status, {"error": payload})


def _attach_bridge_run_context(
    payload: dict[str, Any],
    *,
    run_id: Optional[str],
    progress_path: Optional[Path],
) -> None:
    if not run_id:
        return
    payload["run_id"] = run_id
    payload["progress_url"] = f"/v1/utopic/runs/{run_id}/events"
    if progress_path is not None:
        payload["progress"] = _read_progress(progress_path)


def _bridge_command(entry: models.ModelEntry) -> Optional[list[str]]:
    value = os.environ.get(_bridge_command_env_var(entry)) or os.environ.get("UTOPIC_BRIDGE_COMMAND")
    if value:
        return shlex.split(value)
    return [sys.executable, "-m", "utopic.bridge", entry.engine]


def _bridge_command_env_var(entry: models.ModelEntry) -> str:
    normalized = "".join(char if char.isalnum() else "_" for char in entry.engine.upper()).strip("_")
    return f"UTOPIC_BRIDGE_{normalized}_COMMAND"


def _bridge_input(entry: models.ModelEntry, request: dict[str, Any]) -> dict[str, Any]:
    key = _input_key_for_modality(entry.modality)
    if key in request:
        return {key: request[key]}
    if "prompt" in request:
        return {key: request["prompt"]}
    if "messages" in request:
        return {key: request["messages"]}
    return {key: ""}


def _normalize_request_for_runtime(
    path: str,
    entry: models.ModelEntry,
    request: dict[str, Any],
) -> dict[str, Any]:
    if path != "/v1/responses":
        return request

    normalized = {key: value for key, value in request.items() if key != "input"}
    text = _responses_input_text(request.get("input"))
    if not text and isinstance(request.get("prompt"), str):
        text = request["prompt"]
    if entry.modality == "text":
        normalized.pop("prompt", None)
        normalized["messages"] = _responses_input_messages(request.get("input"))
        if not normalized["messages"] and text:
            normalized["messages"] = [{"role": "user", "content": text}]
        if "max_output_tokens" in normalized and "max_tokens" not in normalized:
            normalized["max_tokens"] = normalized.pop("max_output_tokens")
        return normalized
    if entry.modality == "tts":
        normalized["input"] = text
        return normalized
    if entry.modality == "misc":
        if isinstance(request.get("artifact"), str) and request.get("artifact"):
            return normalized
        normalized["artifact"] = text
        return normalized
    normalized["prompt"] = text
    return normalized


def _responses_input_messages(value: object) -> list[dict[str, str]]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    messages: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            role = item.get("role") if isinstance(item.get("role"), str) else "user"
            content = _responses_content_text(item.get("content"))
            if content:
                messages.append({"role": role, "content": content})
    if messages:
        return messages
    text = _responses_input_text(value)
    return [{"role": "user", "content": text}] if text else []


def _responses_input_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = _responses_content_text(item.get("content"))
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(value, dict):
        return _responses_content_text(value.get("content") or value.get("text"))
    return ""


def _responses_content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def _input_key_for_modality(modality: str) -> str:
    if modality == "misc":
        return "artifact"
    return "input" if modality == "tts" else "prompt"


def _bridge_timeout_seconds() -> int:
    value = os.environ.get("UTOPIC_BRIDGE_TIMEOUT_SECONDS", "3600")
    try:
        parsed = int(value)
    except ValueError:
        return 3600
    return max(1, parsed)


def _runner_timeout_seconds() -> int:
    value = os.environ.get("UTOPIC_RUNNER_TIMEOUT_SECONDS", "3600")
    try:
        parsed = int(value)
    except ValueError:
        return 3600
    return max(1, parsed)


def _normalize_artifacts(
    raw_artifacts: object,
    entry: models.ModelEntry,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not isinstance(raw_artifacts, list):
        return [], None
    output_root = output_dir.expanduser().resolve()
    artifacts: list[dict[str, Any]] = []
    supported_types = set(entry.outputs)
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            continue
        path_value = raw.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        artifact_path = Path(path_value).expanduser().resolve()
        if not artifact_path.is_file() or not _path_is_relative_to(artifact_path, output_root):
            continue
        artifact_type = raw.get("type") if isinstance(raw.get("type"), str) else entry.outputs[0]
        if supported_types and artifact_type not in supported_types:
            return [], f"unsupported artifact type {artifact_type}; expected one of {sorted(supported_types)}"
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        artifacts.append(
            {
                "type": artifact_type,
                "path": str(artifact_path),
                "url": artifact_path.as_uri(),
                "metadata": metadata,
            }
        )
    return artifacts, None


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _runs_dir() -> Path:
    configured = os.environ.get("UTOPIC_RUNS_DIR")
    if configured:
        return Path(configured).expanduser()
    return installer.cache_root() / "runs"


def _run_progress_response(run_id: str) -> tuple[int, dict[str, str], bytes]:
    if not _is_safe_run_id(run_id):
        return _json(404, {"error": {"message": f"unknown run: {run_id}", "code": "run_not_found"}})
    progress_path = _runs_dir() / run_id / "progress.jsonl"
    if not progress_path.is_file():
        return _json(404, {"error": {"message": f"unknown run: {run_id}", "code": "run_not_found"}})
    return _json(200, {"object": "list", "data": _read_progress(progress_path)})


def _is_safe_run_id(run_id: str) -> bool:
    return run_id.startswith("run_") and all(char.isalnum() or char == "_" for char in run_id)


def _read_progress(progress_path: Path) -> list[dict[str, Any]]:
    if not progress_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _forward_native_text(
    native_base_url: str,
    path: str,
    request_body: dict[str, Any],
) -> tuple[int, dict[str, str], bytes]:
    target = _join_native_url(native_base_url, path)
    data = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        target,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            content_type = response.headers.get("content-type", "application/json")
            return response.getcode(), {"content-type": content_type}, response.read()
    except urllib.error.HTTPError as exc:
        content_type = exc.headers.get("content-type", "application/json")
        return exc.code, {"content-type": content_type}, exc.read()
    except urllib.error.URLError as exc:
        return _json(
            502,
            {
                "error": {
                    "message": f"native text server unavailable: {exc.reason}",
                    "code": "native_text_server_unavailable",
                }
            },
        )


def _join_native_url(native_base_url: str, path: str) -> str:
    base = native_base_url.rstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        return base + path[len("/v1") :]
    return base + path


def _native_chat_to_response(entry: models.ModelEntry, body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    text = _chat_completion_text(payload)
    chat_id = payload.get("id") if isinstance(payload, dict) and isinstance(payload.get("id"), str) else uuid.uuid4().hex
    created = payload.get("created") if isinstance(payload, dict) and isinstance(payload.get("created"), int) else int(time.time())
    return {
        "id": f"resp_{chat_id}",
        "object": "response",
        "created_at": created,
        "model": entry.id,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "output_text": text,
        "metadata": {
            "source_object": payload.get("object") if isinstance(payload, dict) else None,
            "runtime": entry.runtime,
            "engine": entry.engine,
        },
    }


def _artifact_response_to_responses(
    entry: models.ModelEntry,
    artifact_response: dict[str, Any],
) -> dict[str, Any]:
    artifacts = artifact_response.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    content = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        content.append(
            {
                "type": _response_content_type_for_artifact(artifact.get("type"), entry),
                "url": artifact.get("url"),
                "metadata": artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {},
            }
        )
    return {
        "id": artifact_response.get("id"),
        "object": "response",
        "created_at": artifact_response.get("created"),
        "model": entry.id,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": content,
            }
        ],
        "artifacts": artifacts,
        "progress": artifact_response.get("progress", []),
        "progress_url": artifact_response.get("progress_url"),
        "metadata": artifact_response.get("metadata", {}),
    }


def _response_content_type_for_artifact(content_type: object, entry: models.ModelEntry) -> str:
    content_type_text = content_type if isinstance(content_type, str) else ""
    if entry.modality == "image" or content_type_text.startswith("image/"):
        return "output_image"
    if entry.modality in {"tts", "music"} or content_type_text.startswith("audio/"):
        return "output_audio"
    if entry.modality == "video" or content_type_text.startswith("video/"):
        return "output_video"
    return "output_file"


def _chat_completion_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    content = first.get("text")
    return content if isinstance(content, str) else ""


def _model_payload(entry: models.ModelEntry) -> dict[str, Any]:
    payload = {
        "id": entry.id,
        "object": "model",
        "created": 0,
        "owned_by": "utopic",
        "name": entry.name,
        "family": entry.family,
        "modality": entry.modality,
        "engine": entry.engine,
        "runtime": entry.runtime,
        "hardware": list(entry.hardware),
        "endpoints": list(entry.endpoints),
        "outputs": list(entry.outputs),
        "requirements": entry.requirements or {},
        "native_status": entry.native_status,
        "runner": entry.runner,
        "supported_backends": list(entry.supported_backends),
        "artifact_filenames": list(entry.artifact_filenames),
        "expected_vram_gib": entry.expected_vram_gib,
        "expected_ram_gib": entry.expected_ram_gib,
        "oom_policy": entry.oom_policy,
        "repo": entry.repo,
        "url": entry.url,
        "description": entry.description,
    }
    if entry.runtime == "bridge":
        payload["bridge"] = _bridge_model_payload(entry)
    return payload


def _bridge_model_payload(entry: models.ModelEntry) -> dict[str, Any]:
    adapter = bridge.ADAPTERS.get(entry.engine)
    return {
        "schema_version": bridge.SCHEMA_VERSION,
        "engine": entry.engine,
        "command": f"utopic-bridge {entry.engine}",
        "environment_variable": _bridge_command_env_var(entry),
        "install_hint": adapter.install_hint if adapter is not None else "",
        "input": _input_key_for_modality(entry.modality),
        "outputs": list(entry.outputs),
        "progress_events": ["queued", "loading", "generating", "completed", "failed"],
    }


def _json(status: int, payload: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    return status, JSON_HEADERS, json.dumps(payload).encode("utf-8")


class GatewayHandler(BaseHTTPRequestHandler):
    native_base_url: Optional[str] = None

    def do_GET(self) -> None:
        self._send(
            *handle_openai_request(
                "GET",
                self.path,
                None,
                native_base_url=self.native_base_url,
            )
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(*_json(400, {"error": {"message": "invalid JSON", "code": "invalid_json"}}))
            return
        if self.path == "/mcp":
            self._send(*handle_mcp_request(body, native_base_url=self.native_base_url))
            return
        self._send(
            *handle_openai_request(
                "POST",
                self.path,
                body,
                native_base_url=self.native_base_url,
            )
        )

    def log_message(self, *_args: object) -> None:
        return

    def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str = "127.0.0.1", port: int = 8911, native_base_url: Optional[str] = None) -> None:
    class ConfiguredGatewayHandler(GatewayHandler):
        pass

    ConfiguredGatewayHandler.native_base_url = native_base_url
    server = ThreadingHTTPServer((host, port), ConfiguredGatewayHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in ("-h", "--help") for arg in args):
        print(HELP)
        return 0
    if "--version" in args:
        print(f"utopic-runtime {__version__}")
        return 0
    try:
        host = _value_after(args, "--host", "127.0.0.1")
        port_text = _value_after(args, "--port", "8911")
        native_base_url = _value_after(args, "--native-base-url", "")
    except ValueError as exc:
        print(f"utopic-runtime: {exc}", file=sys.stderr)
        return 1
    try:
        port = int(port_text)
    except ValueError:
        print("utopic-runtime: --port must be an integer", file=sys.stderr)
        return 1
    print(f"Utopic runtime gateway: http://{host}:{port}")
    print(f"OpenAI-compatible models: http://{host}:{port}/v1/models")
    print(f"MCP endpoint: http://{host}:{port}/mcp")
    if native_base_url:
        print(f"Native text server: {native_base_url.rstrip('/')}")
    try:
        serve(host, port, native_base_url=native_base_url or None)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"utopic-runtime: failed to start server: {exc}", file=sys.stderr)
        return 1
    return 0


def _value_after(args: list[str], flag: str, default: str) -> str:
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 >= len(args) or args[index + 1].startswith("-"):
                raise ValueError(f"expected a value after {flag}")
            return args[index + 1]
        if arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
            if not value or value.startswith("-"):
                raise ValueError(f"expected a value after {flag}")
            return value
    return default


if __name__ == "__main__":
    raise SystemExit(main())
