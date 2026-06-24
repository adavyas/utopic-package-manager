import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional


SCHEMA_VERSION = "utopic-bridge/v1"
RETIRED_MESSAGE = (
    "The packaged Python bridge adapter has been retired. "
    "Use utopic setup plus the local native runner, gateway, or MCP surfaces."
)
RETIRED_DESCRIPTION = "Compatibility shim only; production generation uses the native runner."


@dataclass(frozen=True)
class BridgeAdapter:
    engine: str
    packages: tuple[str, ...] = ()
    install_hint: str = ""
    description: str = RETIRED_DESCRIPTION


ADAPTERS = {
    engine: BridgeAdapter(engine=engine)
    for engine in (
        "diffusers",
        "cosmos",
        "kokoro",
        "chatterbox",
        "dia",
        "ace-step",
        "wan",
        "ltx",
        "artifact",
    )
}


HELP = """usage: utopic-bridge ENGINE [--check]

Compatibility shim for retired Utopic Python bridge adapters.

Production generation now goes through the local native runner, gateway, or MCP
surfaces after `utopic setup`. This command remains so existing scripts receive
structured JSON diagnostics instead of a missing executable.

Known retired engines:
  diffusers
  cosmos
  kokoro
  chatterbox
  dia
  ace-step
  wan
  ltx
  artifact
"""


def main(argv: Optional[list[str]] = None, *, stdin: Optional[str] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or any(arg in ("-h", "--help") for arg in args):
        print(HELP)
        return 0
    engine = args[0]
    adapter = ADAPTERS.get(engine)
    if adapter is None:
        print(json.dumps(_error(engine, f"unknown bridge engine: {engine}", "bridge_engine_unknown", "")))
        return 0
    if "--check" in args:
        print(json.dumps(_check_adapter(adapter)))
        return 0
    raw_request = sys.stdin.read() if stdin is None else stdin
    try:
        request = json.loads(raw_request or "{}")
    except json.JSONDecodeError as exc:
        print(json.dumps(_error(adapter.engine, f"invalid bridge request JSON: {exc}", "bridge_invalid_request", "")))
        return 0
    validation_error = _validate_bridge_request(adapter, request)
    if validation_error is not None:
        print(json.dumps(_error(adapter.engine, validation_error, "bridge_invalid_request", "", request=request)))
        return 0
    print(json.dumps(_error(adapter.engine, RETIRED_MESSAGE, "native_runner_required", "utopic setup", request=request)))
    return 0


def _experimental_bridge_enabled() -> bool:
    value = os.environ.get("UTOPIC_EXPERIMENTAL_BRIDGE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _check_adapter(adapter: BridgeAdapter) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": adapter.engine,
        "status": "retired",
        "ready": False,
        "packages": [],
        "missing": [],
        "install_hint": "",
        "description": adapter.description,
        "message": RETIRED_MESSAGE,
    }


def _validate_bridge_request(adapter: BridgeAdapter, request: object) -> Optional[str]:
    if not isinstance(request, dict):
        return "bridge request must be a JSON object"
    if request.get("schema_version") != SCHEMA_VERSION:
        return f"schema_version must be {SCHEMA_VERSION}"
    engine = request.get("engine")
    if engine != adapter.engine:
        return f"engine must match adapter {adapter.engine}"
    for field in ("model", "modality", "output_dir", "progress_path"):
        value = request.get(field)
        if not isinstance(value, str) or not value:
            return f"{field} must be a non-empty string"
    input_value = request.get("input")
    if not isinstance(input_value, dict) or not input_value:
        return "input must be a non-empty object"
    if request["modality"] == "tts":
        if not isinstance(input_value.get("input"), str) or not input_value.get("input"):
            return "input.input must be a non-empty string"
    elif request["modality"] == "misc":
        if not isinstance(input_value.get("artifact"), str) or not input_value.get("artifact"):
            return "input.artifact must be a non-empty string"
    elif not isinstance(input_value.get("prompt"), str) or not input_value.get("prompt"):
        return "input.prompt must be a non-empty string"
    parameters = request.get("parameters", {})
    if parameters is not None and not isinstance(parameters, dict):
        return "parameters must be an object"
    metadata = request.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        return "metadata must be an object"
    model_cache_path = request.get("model_cache_path")
    if model_cache_path is not None and not isinstance(model_cache_path, str):
        return "model_cache_path must be a string"
    return None


def _error(
    engine: str,
    message: str,
    code: str,
    install_hint: str,
    *,
    request: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    model = request.get("model") if isinstance(request, dict) else None
    modality = request.get("modality") if isinstance(request, dict) else None
    return {
        "error": {
            "code": code,
            "message": message,
            "engine": engine,
            "install_hint": install_hint,
        },
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "engine": engine,
            "model": model,
            "modality": modality,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
