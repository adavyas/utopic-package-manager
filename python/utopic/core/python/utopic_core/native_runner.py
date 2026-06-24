import json
import os
import subprocess
import tempfile
import time
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

    try:
        runner = _native.binary_path("utopic_runner")
    except RuntimeError as exc:
        return _error("backend_unavailable", str(exc), {"binary": "utopic_runner"})

    runner_request = _runner_request(entry, request)
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
                {"model": entry.id},
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


def _runner_request(entry: models.ModelEntry, request: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model_path": str(entry.path),
    }
    _copy_option(request, options, "max_tokens", "max_tokens")
    _copy_option(request, options, "temperature", "temperature")
    _copy_option(request, options, "seed", "seed")
    _copy_option(request, options, "diffusion_canvas_tokens", "canvas")
    _copy_option(request, options, "diffusion_steps", "steps")
    _copy_option(request, options, "diffusion_block_length", "diffusion_block_length")

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "chat",
        "model": entry.id,
        "input": {
            "messages": request.get("messages", []),
        },
        "options": options,
        "output_dir": str(_runs_dir()),
    }


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
