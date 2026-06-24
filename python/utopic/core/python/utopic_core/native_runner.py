import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from . import _native, models


SCHEMA_VERSION = "utopic-runner/v1"


def chat_completion(entry: models.ModelEntry, request: dict[str, Any]) -> dict[str, Any]:
    if not entry.path.exists():
        return _error(
            "missing_model",
            f"model is not downloaded: {entry.id}",
            {"model": entry.id, "model_path": str(entry.path)},
        )

    return _invoke_runner(_runner_request(entry, "chat", {"messages": request.get("messages", [])}, request))


def generation(entry: models.ModelEntry, endpoint: str, request: dict[str, Any]) -> dict[str, Any]:
    runner_input = _generation_input(entry, request)
    payload = _invoke_runner(
        _runner_request(entry, entry.modality, runner_input, request, endpoint=endpoint),
        binary_name=entry.runner or "utopic_runner",
        binary_unavailable_payload=native_binary_unavailable_error(entry),
    )
    return _enrich_native_readiness_error(entry, payload)


def _invoke_runner(
    runner_request: dict[str, Any],
    *,
    binary_name: str = "utopic_runner",
    binary_unavailable_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        runner = _native.binary_path(binary_name)
    except RuntimeError as exc:
        if binary_unavailable_payload is not None:
            return binary_unavailable_payload
        return _error("backend_unavailable", str(exc), {"binary": binary_name})

    with tempfile.TemporaryDirectory(prefix="utopic-runner-") as tmp:
        request_path = Path(tmp) / "request.json"
        request_path.write_text(json.dumps(runner_request), encoding="utf-8")
        started = time.time()
        try:
            completed = subprocess.run(
                [str(runner), "--json-request", str(request_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=_runner_timeout_seconds(),
            )
        except subprocess.TimeoutExpired:
            return _error(
                "runner_failed",
                f"native runner timed out after {_runner_timeout_seconds()} seconds",
                {"model": runner_request.get("model")},
            )

    try:
        payload = _load_runner_stdout(completed.stdout)
    except ValueError as exc:
        return _error(
            "runner_failed",
            f"native runner returned invalid JSON: {exc}",
            {"stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]},
        )

    if not isinstance(payload, dict):
        return _error("runner_failed", "native runner returned a non-object response")
    if completed.returncode != 0 and payload.get("ok") is not False:
        return _error(
            "runner_failed",
            f"native runner exited with status {completed.returncode}",
            {"stderr": completed.stderr[-4000:]},
        )
    if payload.get("ok") is False:
        return payload

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    metrics.setdefault("wall_ms", round((time.time() - started) * 1000.0, 3))
    payload["metrics"] = metrics
    return payload


def _runner_request(
    entry: models.ModelEntry,
    task: str,
    runner_input: dict[str, Any],
    request: dict[str, Any],
    *,
    endpoint: str | None = None,
) -> dict[str, Any]:
    cache_path = _entry_path_string(entry)
    options: dict[str, Any] = {
        "endpoint": endpoint or "/v1/chat/completions",
        "modality": entry.modality,
        "engine": entry.engine,
        "runtime": entry.runtime,
        "runner": entry.runner,
        "native_status": entry.native_status,
        "outputs": list(entry.outputs),
        "supported_backends": list(entry.supported_backends),
    }
    if cache_path is not None:
        options["model_cache_path"] = cache_path
        if entry.runtime == "native":
            options["model_path"] = cache_path
    if entry.expected_vram_gib is not None:
        options["expected_vram_gib"] = entry.expected_vram_gib
    if entry.expected_ram_gib is not None:
        options["expected_ram_gib"] = entry.expected_ram_gib
    if entry.requirements:
        options["requirements"] = dict(entry.requirements)
    _copy_option(request, options, "max_tokens", "max_tokens")
    _copy_option(request, options, "temperature", "temperature")
    _copy_option(request, options, "seed", "seed")
    _copy_option(request, options, "gpu_layers", "gpu_layers")
    _copy_option(request, options, "diffusion_canvas_tokens", "canvas")
    _copy_option(request, options, "diffusion_steps", "steps")
    _copy_option(request, options, "diffusion_block_length", "diffusion_block_length")
    _copy_option(request, options, "confidence", "confidence")
    _copy_option(request, options, "converge", "converge")
    _copy_option(request, options, "eb_steps", "eb_steps")
    _copy_option(request, options, "slot_len", "slot_len")
    _copy_option(request, options, "schema", "schema")
    _copy_option(request, options, "schema_mode", "schema_mode")
    _copy_option(request, options, "tools", "tools")
    _copy_option(request, options, "size", "size")
    _copy_option(request, options, "response_format", "response_format")
    _copy_option(request, options, "voice", "voice")
    _copy_option(request, options, "duration", "duration")

    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "model": entry.id,
        "input": runner_input,
        "options": options,
        "output_dir": str(_allocate_output_dir()),
    }


def _generation_input(entry: models.ModelEntry, request: dict[str, Any]) -> dict[str, Any]:
    if entry.modality == "tts":
        return {"input": request.get("input", "")}
    if entry.modality == "misc":
        return {
            "artifact": request.get("artifact") or request.get("input_file") or "",
            "artifact_type": request.get("artifact_type", ""),
        }
    return {"prompt": request.get("prompt", "")}


def _entry_path_string(entry: models.ModelEntry) -> str | None:
    try:
        return str(entry.path)
    except RuntimeError:
        return None


def _load_runner_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("native runner returned a non-object response")
        return payload
    raise ValueError("native runner did not write a response")


def native_readiness_error(entry: models.ModelEntry) -> dict[str, Any]:
    payload = _error(
        "unsupported_model",
        (
            f"{entry.modality} generation for {entry.id} is cataloged but does not "
            "have a native C++ runner yet"
        ),
        {
            "model": entry.id,
            "modality": entry.modality,
            "engine": entry.engine,
            "runtime": entry.runtime,
            "runner": entry.runner,
            "native_status": entry.native_status,
            "supported_backends": list(entry.supported_backends),
            "expected_vram_gib": entry.expected_vram_gib,
            "expected_ram_gib": entry.expected_ram_gib,
        },
    )
    payload["error"]["model"] = entry.id
    payload["error"]["modality"] = entry.modality
    payload["error"]["engine"] = entry.engine
    payload["error"]["runner"] = entry.runner
    payload["error"]["native_status"] = entry.native_status
    payload["error"]["supported_backends"] = list(entry.supported_backends)
    return payload


def native_binary_unavailable_error(entry: models.ModelEntry) -> dict[str, Any]:
    runner = entry.runner or "utopic_runner"
    payload = _error(
        "backend_unavailable",
        (
            f"native runner binary is not installed for {entry.id}: {runner}. "
            "Run `utopic setup` to build and cache package-managed native runners."
        ),
        {
            "model": entry.id,
            "modality": entry.modality,
            "engine": entry.engine,
            "runtime": entry.runtime,
            "runner": runner,
            "binary": runner,
            "native_status": entry.native_status,
            "supported_backends": list(entry.supported_backends),
            "expected_vram_gib": entry.expected_vram_gib,
            "expected_ram_gib": entry.expected_ram_gib,
            "setup_command": "utopic setup",
        },
    )
    payload["error"]["model"] = entry.id
    payload["error"]["modality"] = entry.modality
    payload["error"]["engine"] = entry.engine
    payload["error"]["runner"] = runner
    payload["error"]["native_status"] = entry.native_status
    payload["error"]["supported_backends"] = list(entry.supported_backends)
    return payload


def _enrich_native_readiness_error(entry: models.ModelEntry, payload: dict[str, Any]) -> dict[str, Any]:
    error = payload.get("error")
    if payload.get("ok") is not False or not isinstance(error, dict):
        return payload
    if error.get("code") != "unsupported_model":
        return payload

    detail = error.get("detail")
    if not isinstance(detail, dict):
        detail = {}
        error["detail"] = detail
    detail.setdefault("model", entry.id)
    detail.setdefault("modality", entry.modality)
    detail.setdefault("engine", entry.engine)
    detail.setdefault("runner", entry.runner)
    detail.setdefault("native_status", entry.native_status)
    detail.setdefault("supported_backends", list(entry.supported_backends))
    detail.setdefault("expected_vram_gib", entry.expected_vram_gib)
    detail.setdefault("expected_ram_gib", entry.expected_ram_gib)

    error["model"] = entry.id
    error["modality"] = entry.modality
    error["engine"] = entry.engine
    error["runner"] = entry.runner
    error["native_status"] = entry.native_status
    error["supported_backends"] = list(entry.supported_backends)
    return payload


def _copy_option(source: dict[str, Any], dest: dict[str, Any], source_key: str, dest_key: str) -> None:
    if source_key in source:
        dest[dest_key] = source[source_key]


def _runs_dir() -> Path:
    configured = os.environ.get("UTOPIC_RUNS_DIR")
    if configured:
        return Path(configured).expanduser()
    try:
        return models.installer.cache_root() / "runs"
    except RuntimeError:
        return Path(tempfile.gettempdir()) / "utopic" / "runs"


def _allocate_output_dir() -> Path:
    root = _runs_dir()
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / ("run_" + uuid.uuid4().hex)
    output_dir.mkdir()
    return output_dir


def _runner_timeout_seconds() -> int:
    value = os.environ.get("UTOPIC_RUNNER_TIMEOUT")
    if not value:
        return 600
    try:
        parsed = int(value)
    except ValueError:
        return 600
    return parsed if parsed > 0 else 600


def _error(code: str, message: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "detail": detail or {},
        },
    }
