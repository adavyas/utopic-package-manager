import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __version__, _native, bridge, chat, gateway, installer, mcp, models, native_runner


_RUN_VALUE_FLAGS = {"--host", "--port", "--native-port", "-ngl", "--ctx-size"}
_PROMPT_VALUE_FLAGS = (_RUN_VALUE_FLAGS - {"--native-port"}) | {
    "-p",
    "--prompt",
    "-n",
    "--temp",
    "--seed",
    "--system",
    "--tools",
    "--schema",
    "--confidence",
    "--converge",
    "--steps",
    "--diffusion-block-length",
    "--canvas",
    "--eb-steps",
    "--slot-len",
}
_RUN_NUMERIC_FLAGS = {
    "--port": (1, 65535, "an integer from 1 to 65535"),
    "--native-port": (1, 65535, "an integer from 1 to 65535"),
    "-ngl": (0, None, "a non-negative integer"),
    "--ctx-size": (1, None, "a positive integer"),
}
_PROMPT_INTEGER_FLAGS = {
    "-n": (1, None, "a positive integer"),
    "--seed": (None, None, "an integer"),
    "--steps": (1, None, "a positive integer"),
    "--diffusion-block-length": (1, None, "a positive integer"),
    "--canvas": (0, None, "a non-negative integer"),
    "--eb-steps": (0, None, "a non-negative integer"),
    "--slot-len": (1, None, "a positive integer"),
    "--converge": (0, None, "a non-negative integer"),
}
_PROMPT_NUMBER_FLAGS = {
    "--temp": (0.0, None, "a non-negative number"),
    "--confidence": (None, None, "a number"),
}
_MODEL_VALUE_FLAGS = {"-m", "--model"}
_LEGACY_NATIVE_VALUE_FLAGS = {
    "-m",
    "-p",
    "-n",
    "--temp",
    "--seed",
    "--system",
    "--tools",
    "--schema",
    "--confidence",
    "--converge",
    "--steps",
    "--diffusion-block-length",
    "--canvas",
    "--eb-steps",
    "--slot-len",
    "-ngl",
}
_LEGACY_NATIVE_NUMERIC_VALUE_FLAGS = {
    "-n",
    "--temp",
    "--seed",
    "--confidence",
    "--converge",
    "--steps",
    "--diffusion-block-length",
    "--canvas",
    "--eb-steps",
    "--slot-len",
    "-ngl",
}
_LEGACY_NATIVE_BOOLEAN_FLAGS = {"--reasoning", "--soft-schema"}


def _ensure_setup(enabled: bool = True, binary_name: str = "utopic") -> None:
    if enabled and not installer.native_installation_is_current((binary_name,)):
        try:
            code = installer.setup([])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"setup command failed: {_format_command(exc.cmd)}") from exc
        if code != 0:
            raise SystemExit(code)


def _has_prompt(args: Sequence[str]) -> bool:
    for arg in args:
        if arg in ("-p", "--prompt") or arg.startswith("--prompt="):
            return True
    return False


def _without_flag(args: Sequence[str], flag: str) -> list[str]:
    return [arg for arg in args if arg != flag]


def _validate_run_value_flags(args: Sequence[str]) -> None:
    for index, arg in enumerate(args):
        if arg in _MODEL_VALUE_FLAGS:
            if (
                index + 1 >= len(args)
                or args[index + 1] == ""
                or args[index + 1].startswith("-")
            ):
                raise RuntimeError("expected a value after -m/--model")
        if arg.startswith("--model="):
            value = arg.split("=", 1)[1]
            if value == "" or value.startswith("-"):
                raise RuntimeError("expected a value after -m/--model")
        if arg in _RUN_VALUE_FLAGS:
            if index + 1 >= len(args):
                raise RuntimeError(f"expected a value after {arg}")
            value = args[index + 1]
            if value.startswith("-") and not (arg in _RUN_NUMERIC_FLAGS and _looks_like_negative_number(value)):
                raise RuntimeError(f"expected a value after {arg}")
            if arg in _RUN_NUMERIC_FLAGS:
                _validate_run_numeric_flag(arg, value)
        for flag in _RUN_VALUE_FLAGS:
            if not flag.startswith("--") or not arg.startswith(f"{flag}="):
                continue
            value = arg.split("=", 1)[1]
            if value == "":
                raise RuntimeError(f"expected a value after {flag}")
            if flag in _RUN_NUMERIC_FLAGS:
                _validate_run_numeric_flag(flag, value)


def _validate_prompt_value_flags(args: Sequence[str]) -> None:
    prompt_only_flags = _PROMPT_VALUE_FLAGS - _RUN_VALUE_FLAGS
    for index, arg in enumerate(args):
        if arg in prompt_only_flags:
            if index + 1 >= len(args):
                raise RuntimeError(f"expected a value after {arg}")
            value = args[index + 1]
            if value == "" or (value.startswith("-") and not _looks_like_negative_number(value)):
                raise RuntimeError(f"expected a value after {arg}")
            _validate_prompt_numeric_flag(arg, value)
        for flag in prompt_only_flags:
            if not flag.startswith("--") or not arg.startswith(f"{flag}="):
                continue
            value = arg.split("=", 1)[1]
            if value == "":
                raise RuntimeError(f"expected a value after {flag}")
            _validate_prompt_numeric_flag(flag, value)


def _validate_model_argument_count(args: Sequence[str], value_flags: set[str]) -> None:
    count = 0
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in _MODEL_VALUE_FLAGS:
            count += 1
            index += 2
            continue
        if arg.startswith("--model="):
            count += 1
            index += 1
            continue
        if arg in value_flags:
            index += 2
            continue
        if any(arg.startswith(flag + "=") for flag in value_flags if flag.startswith("--")):
            index += 1
            continue
        if not arg.startswith("-"):
            count += 1
        index += 1

    if count > 1:
        raise RuntimeError("expected at most one model argument")


def _validate_server_options(args: Sequence[str]) -> None:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in _MODEL_VALUE_FLAGS or arg in _RUN_VALUE_FLAGS:
            index += 2
            continue
        if arg.startswith("--model="):
            index += 1
            continue
        if any(arg.startswith(flag + "=") for flag in _RUN_VALUE_FLAGS if flag.startswith("--")):
            index += 1
            continue
        if arg.startswith("-"):
            raise RuntimeError(f"unknown option: {arg}")
        index += 1


def _validate_legacy_native_options(args: Sequence[str]) -> None:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in _LEGACY_NATIVE_VALUE_FLAGS:
            if index + 1 >= len(args):
                raise RuntimeError(f"expected a value after {arg}")
            value = args[index + 1]
            allow_negative = arg in _LEGACY_NATIVE_NUMERIC_VALUE_FLAGS
            if value == "" or (
                value.startswith("-")
                and not (allow_negative and _looks_like_negative_number(value))
            ):
                raise RuntimeError(f"expected a value after {arg}")
            index += 2
            continue
        if arg in _LEGACY_NATIVE_BOOLEAN_FLAGS:
            index += 1
            continue
        if arg.startswith("-"):
            raise RuntimeError(f"unknown option: {arg}")
        index += 1


def _looks_like_negative_number(value: str) -> bool:
    return len(value) > 1 and value[0] == "-" and value[1].isdigit()


def _validate_run_numeric_flag(flag: str, value: str) -> None:
    minimum, maximum, label = _RUN_NUMERIC_FLAGS[flag]
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{flag} must be {label}") from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        raise RuntimeError(f"{flag} must be {label}")


def _validate_prompt_numeric_flag(flag: str, value: str) -> None:
    if flag in _PROMPT_INTEGER_FLAGS:
        minimum, maximum, label = _PROMPT_INTEGER_FLAGS[flag]
        try:
            parsed = int(value)
        except ValueError as exc:
            raise RuntimeError(f"{flag} must be {label}") from exc
        if (
            (minimum is not None and parsed < minimum)
            or (maximum is not None and parsed > maximum)
        ):
            raise RuntimeError(f"{flag} must be {label}")
    if flag in _PROMPT_NUMBER_FLAGS:
        minimum, maximum, label = _PROMPT_NUMBER_FLAGS[flag]
        try:
            parsed = float(value)
        except ValueError as exc:
            raise RuntimeError(f"{flag} must be {label}") from exc
        if (
            not math.isfinite(parsed)
            or (minimum is not None and parsed < minimum)
            or (maximum is not None and parsed > maximum)
        ):
            raise RuntimeError(f"{flag} must be {label}")


def _extract_model(args: Sequence[str]) -> tuple[Optional[str], list[str]]:
    remaining = list(args)
    for index, arg in enumerate(remaining):
        if arg in ("-m", "--model"):
            if index + 1 >= len(remaining):
                raise SystemExit("utopic run: expected a value after -m/--model")
            value = remaining[index + 1]
            del remaining[index : index + 2]
            return value, remaining
        if arg.startswith("--model="):
            value = arg.split("=", 1)[1]
            del remaining[index]
            return value, remaining

    index = 0
    while index < len(remaining):
        arg = remaining[index]
        if arg in _RUN_VALUE_FLAGS:
            index += 2
            continue
        if any(arg.startswith(flag + "=") for flag in _RUN_VALUE_FLAGS if flag.startswith("--")):
            index += 1
            continue
        if not arg.startswith("-"):
            del remaining[index]
            return arg, remaining
        index += 1
    return None, remaining


def _resolve_prompt_model_args(args: Sequence[str]) -> list[str]:
    resolved = _normalize_prompt_native_args(args)
    for index, arg in enumerate(resolved):
        if arg == "-m":
            resolved[index + 1] = str(models.ensure_model(resolved[index + 1]))
            return resolved
    model_arg, remaining = _extract_prompt_positional_model(resolved)
    return ["-m", str(models.ensure_model(model_arg)), *remaining]


def _normalize_prompt_native_args(args: Sequence[str]) -> list[str]:
    resolved = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-m", "--model"):
            resolved.extend(["-m", args[index + 1]])
            index += 2
            continue
        if arg.startswith("--model="):
            resolved.extend(["-m", arg.split("=", 1)[1]])
            index += 1
            continue
        if arg in ("-p", "--prompt"):
            resolved.extend(["-p", args[index + 1]])
            index += 2
            continue
        if arg.startswith("--prompt="):
            resolved.extend(["-p", arg.split("=", 1)[1]])
            index += 1
            continue
        if arg in _PROMPT_VALUE_FLAGS:
            resolved.extend([arg, args[index + 1]])
            index += 2
            continue
        expanded = False
        for flag in _PROMPT_VALUE_FLAGS:
            if flag.startswith("--") and arg.startswith(flag + "="):
                resolved.extend([flag, arg.split("=", 1)[1]])
                expanded = True
                break
        if not expanded:
            resolved.append(arg)
        index += 1
    return resolved


def _extract_prompt_positional_model(args: Sequence[str]) -> tuple[Optional[str], list[str]]:
    remaining = list(args)
    index = 0
    while index < len(remaining):
        arg = remaining[index]
        if arg in _PROMPT_VALUE_FLAGS:
            index += 2
            continue
        if any(arg.startswith(flag + "=") for flag in _PROMPT_VALUE_FLAGS if flag.startswith("--")):
            index += 1
            continue
        if not arg.startswith("-"):
            del remaining[index]
            return arg, remaining
        index += 1
    return None, remaining


def _extract_prompt_model(args: Sequence[str]) -> tuple[Optional[str], list[str]]:
    remaining = list(args)
    for index, arg in enumerate(remaining):
        if arg == "-m":
            value = remaining[index + 1]
            del remaining[index : index + 2]
            return value, remaining
    return _extract_prompt_positional_model(remaining)


def _value_after(args: Sequence[str], flag: str, default: str) -> str:
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def _prompt_value(args: Sequence[str], flag: str) -> Optional[str]:
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
    return None


def _prompt_flag(args: Sequence[str], flag: str) -> bool:
    return flag in args


def _set_int_option(request: dict[str, Any], args: Sequence[str], flag: str, key: str) -> None:
    value = _prompt_value(args, flag)
    if value is not None:
        request[key] = int(value)


def _set_float_option(request: dict[str, Any], args: Sequence[str], flag: str, key: str) -> None:
    value = _prompt_value(args, flag)
    if value is not None:
        request[key] = float(value)


def _set_string_option(request: dict[str, Any], args: Sequence[str], flag: str, key: str) -> None:
    value = _prompt_value(args, flag)
    if value is not None:
        request[key] = value


def _prompt_runner_entry(model_arg: Optional[str], model_path: Path) -> Any:
    if model_arg:
        entry = models.get_model(model_arg)
        if entry is not None:
            return entry
        return models.local_text_entry(model_arg, model_path)
    return models.default_model()


def _prompt_runner_request(args: Sequence[str]) -> tuple[Any, dict[str, Any]]:
    normalized = _normalize_prompt_native_args(args)
    model_arg, remaining = _extract_prompt_model(normalized)
    prompt = _prompt_value(remaining, "-p") or ""
    model_path = models.ensure_model(model_arg)
    entry = _prompt_runner_entry(model_arg, model_path)

    messages: list[dict[str, str]] = []
    system_prompt = _prompt_value(remaining, "--system")
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    request: dict[str, Any] = {
        "model": entry.id,
        "messages": messages,
    }
    _set_int_option(request, remaining, "-n", "max_tokens")
    _set_float_option(request, remaining, "--temp", "temperature")
    _set_int_option(request, remaining, "--seed", "seed")
    _set_int_option(request, remaining, "--steps", "diffusion_steps")
    _set_int_option(request, remaining, "--diffusion-block-length", "diffusion_block_length")
    _set_int_option(request, remaining, "--canvas", "diffusion_canvas_tokens")
    _set_float_option(request, remaining, "--confidence", "confidence")
    _set_int_option(request, remaining, "--converge", "converge")
    _set_int_option(request, remaining, "--eb-steps", "eb_steps")
    _set_int_option(request, remaining, "--slot-len", "slot_len")
    _set_int_option(request, remaining, "-ngl", "gpu_layers")
    _set_string_option(request, remaining, "--schema", "schema")
    if _prompt_flag(remaining, "--soft-schema"):
        request["schema_mode"] = "prompt"
    if _prompt_value(remaining, "--tools") is not None:
        request["tools"] = True
    return entry, request


def _run_prompt(args: Sequence[str], setup_enabled: bool) -> int:
    _validate_prompt_value_flags(args)
    _validate_run_value_flags(args)
    _validate_model_argument_count(args, _PROMPT_VALUE_FLAGS)
    _ensure_setup(setup_enabled, "utopic_runner")
    _native.binary_path("utopic_runner")
    entry, request = _prompt_runner_request(args)
    payload = native_runner.chat_completion(entry, request)
    if payload.get("ok") is False:
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else "native runner failed"
        print(f"utopic run: {message}", file=sys.stderr)
        return 1
    text = payload.get("text", "")
    if not isinstance(text, str):
        print("utopic run: native runner returned non-text output", file=sys.stderr)
        return 1
    if text:
        print(text)
    return 0


def _client_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def _server_url(host: str, port: str) -> str:
    return f"http://{_client_host(host)}:{port}/v1/chat/completions"


def _server_base_url(host: str, port: str) -> str:
    return f"http://{_client_host(host)}:{port}"


def _server_health_url(host: str, port: str) -> str:
    return f"http://{_client_host(host)}:{port}/health"


def _endpoint_url(host: str, port: str, endpoint: str) -> str:
    return f"{_server_base_url(host, port)}{endpoint}"


def _default_native_port(public_port: str) -> str:
    try:
        parsed = int(public_port)
    except ValueError:
        return "8911"
    if parsed < 65535:
        return str(parsed + 1)
    return str(parsed - 1)


def _native_server_args(args: Sequence[str]) -> list[str]:
    native_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"--host", "--port", "--native-port"}:
            index += 2
            continue
        if any(arg.startswith(flag + "=") for flag in ("--host", "--port", "--native-port")):
            index += 1
            continue
        native_args.append(arg)
        index += 1
    return native_args


def _wait_for_health(process: subprocess.Popen, health_url: str, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            if code < 0:
                raise RuntimeError(
                    f"utopic-server exited before it became healthy (signal {-code})"
                )
            raise RuntimeError(f"utopic-server exited before it became healthy (code {code})")
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.3)
    raise RuntimeError(f"timed out waiting for {health_url}")


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _run_server(
    model_path: str,
    server_args: Sequence[str],
    host: str,
    port: str,
    native_port: str,
) -> int:
    exe = _native.binary_path("utopic_server")
    native_host = "127.0.0.1"
    native_base_url = _server_base_url(native_host, native_port)
    process = subprocess.Popen(
        [
            str(exe),
            "-m",
            str(model_path),
            "--host",
            native_host,
            "--port",
            native_port,
            *server_args,
        ]
    )
    try:
        _wait_for_health(process, _server_health_url(native_host, native_port))
        print(f"OpenAI-compatible URL: {_server_url(host, port)}", flush=True)
        print(f"OpenAI-compatible models: {_server_base_url(host, port)}/v1/models", flush=True)
        print(f"MCP endpoint: {_server_base_url(host, port)}/mcp", flush=True)
        print(f"Native text server: {native_base_url}", flush=True)
        print(f"Chat with this server: utopic chat --server {_server_base_url(host, port)}", flush=True)
        gateway.serve(host, int(port), native_base_url=native_base_url)
        _stop_process(process)
        return 0
    except KeyboardInterrupt:
        _stop_process(process)
        return 130
    except (RuntimeError, OSError) as exc:
        _stop_process(process)
        print(f"utopic run: {exc}", file=sys.stderr)
        code = process.returncode
        return code if code and code > 0 else 1


def _run_gateway_only(
    host: str,
    port: str,
    entry: Optional[models.ModelEntry] = None,
    active_text_model_path: Optional[Path] = None,
    active_text_model_id: str = "utopic",
) -> int:
    if entry is None or entry.modality == "text":
        print(f"OpenAI-compatible URL: {_server_url(host, port)}", flush=True)
    else:
        for endpoint in entry.endpoints:
            print(f"OpenAI-compatible endpoint: {_endpoint_url(host, port, endpoint)}", flush=True)
    print(f"OpenAI-compatible models: {_server_base_url(host, port)}/v1/models", flush=True)
    print(f"MCP endpoint: {_server_base_url(host, port)}/mcp", flush=True)
    if entry is None or entry.modality == "text":
        print(f"Chat with this server: utopic chat --server {_server_base_url(host, port)}", flush=True)
    try:
        gateway.serve(
            host,
            int(port),
            native_base_url=None,
            active_text_model_path=active_text_model_path,
            active_text_model_id=active_text_model_id,
        )
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"utopic run: {exc}", file=sys.stderr)
        return 1
    return 0


def _print_top_help() -> None:
    print(
        """usage: utopic <command> [options]

Commands:
  chat      Start the bundled chat TUI. Runs setup on first use.
  run       Start the unified OpenAI-compatible and MCP runtime, or run one-shot prompts with -p.
  serve     Alias for `utopic run` in server mode.
  generate  Generate image, speech, music, video, or misc artifacts.
  gateway   Start the unified multimodal OpenAI-compatible and MCP gateway.
  mcp       Start the MCP stdio server. Use --runtime for the Python gateway tools.
  setup     Build and cache native binaries for this host.
  models    List, pull, and locate curated GGUF models.
  doctor    Print local setup diagnostics without building anything.

Examples:
  utopic --version
  utopic chat
  utopic chat diffusiongemma-26b-a4b-q4
  utopic run diffusiongemma-26b-a4b-q4 --port 8910 -ngl 99
  utopic serve diffusiongemma-26b-a4b-q4 --port 8910 -ngl 99
  utopic generate video --quality high -p "cinematic glass city sunrise" -o video.mp4
  utopic generate misc zuna --artifact /path/to/input.bin -o output.bin
  utopic gateway --port 8911
  utopic mcp --runtime
  utopic doctor
  utopic run -m /path/to/model.gguf -p "Answer with one word: 2+2?" -n 16

Run `utopic <command> --help` for command-specific help.
"""
    )


_GENERATE_ENDPOINTS = {
    "image": "/v1/images/generations",
    "speech": "/v1/audio/speech",
    "music": "/v1/audio/generations",
    "video": "/v1/videos/generations",
    "misc": "/v1/utopic/misc/generations",
}

_GENERATE_MODALITY = {
    "image": "image",
    "speech": "tts",
    "music": "music",
    "video": "video",
    "misc": "misc",
}

_GENERATE_QUALITY_MODELS = {
    "fast": {
        "image": "flux-1-schnell",
        "speech": "kokoro-82m",
        "music": "ace-step-3.5b",
        "video": "wan2.1-t2v-1.3b",
        "misc": "zuna",
    },
    "balanced": {
        "image": "qwen-image",
        "speech": "kokoro-82m",
        "music": "ace-step-3.5b",
        "video": "wan2.1-t2v-1.3b",
        "misc": "zuna",
    },
    "high": {
        "image": "qwen-image",
        "speech": "dia-1.6b",
        "music": "ace-step-3.5b",
        "video": "wan2.1-t2v-14b",
        "misc": "zuna",
    },
}


def _print_run_help() -> None:
    print(
        """usage: utopic run [model-alias|/path/to/model.gguf] [server options]
       utopic run -m model.gguf -p prompt [native one-shot options]

Without -p/--prompt, `utopic run` starts the unified runtime gateway and a
private native text server behind it, then prints the local OpenAI-compatible
and MCP URLs. With -p/--prompt, it keeps the native one-shot behavior.
To chat with a running server, use `utopic chat --server http://127.0.0.1:8910`.

Server options:
  -m, --model VALUE     Model alias or GGUF path. Defaults to the recommended model.
  --host HOST           Bind host for the public runtime gateway. Default: 127.0.0.1
  --port PORT           Bind port for the public runtime gateway. Default: 8910
  --native-port PORT    Internal native text server port. Default: PORT+1
  -ngl N                GPU layers to offload. Default: native default
  --ctx-size N          Server context size. Default: native default
  --no-setup            Do not run setup automatically if binaries are missing.

Examples:
  utopic run diffusiongemma-26b-a4b-q4
  utopic chat --server http://127.0.0.1:8910
  utopic run -m /path/to/model.gguf --port 8910 -ngl 99
  utopic run -m /path/to/model.gguf -p "Hello" -n 128
"""
    )


def _format_command(command: object) -> str:
    if isinstance(command, (list, tuple)):
        return shlex.join(str(part) for part in command)
    return str(command)


def _generate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="utopic generate",
        description="Generate local image, speech, music, video, or misc artifacts.",
    )
    parser.add_argument("--version", action="store_true", help="Show version.")
    subparsers = parser.add_subparsers(dest="kind")

    image = subparsers.add_parser("image", help="Generate an image artifact.")
    _add_prompt_generate_args(image)
    _add_generation_common_args(image)
    image.add_argument("--size", help="Image size, for example 1024x1024.")
    image.add_argument("--steps", type=int, help="Diffusion steps.")
    image.add_argument("--guidance-scale", type=float, help="Classifier-free guidance scale.")
    image.add_argument("--true-cfg-scale", type=float, help="Qwen-Image true CFG scale.")
    image.add_argument("--negative-prompt", help="Negative prompt.")

    speech = subparsers.add_parser(
        "speech",
        aliases=["tts"],
        help="Generate a speech audio artifact.",
    )
    speech.add_argument("model", nargs="?", help="TTS model alias.")
    speech.add_argument("--input", required=True, help="Text to speak.")
    _add_generation_common_args(speech)
    speech.add_argument("--voice", help="Voice name.")
    speech.add_argument("--sample-rate", type=int, help="Output sample rate when supported.")

    music = subparsers.add_parser("music", help="Generate a music audio artifact.")
    _add_prompt_generate_args(music)
    _add_generation_common_args(music)
    music.add_argument("--duration", type=float, help="Requested audio duration in seconds.")
    music.add_argument("--lyrics", help="Lyrics text. Use an empty string for instrumental output.")
    music.add_argument("--sample-rate", type=int, help="Output sample rate metadata when supported.")
    music.add_argument("--steps", type=int, help="Generation steps when supported.")
    music.add_argument("--guidance-scale", type=float, help="Guidance scale when supported.")

    video = subparsers.add_parser("video", help="Generate a video artifact.")
    _add_prompt_generate_args(video)
    _add_generation_common_args(video)
    video.add_argument("--size", help="Video size, for example 832x480.")
    video.add_argument("--frames", type=int, help="Number of frames.")
    video.add_argument("--fps", type=int, help="Frames per second.")
    video.add_argument("--steps", type=int, help="Diffusion steps.")
    video.add_argument("--guidance-scale", type=float, help="Classifier-free guidance scale.")
    video.add_argument("--negative-prompt", help="Negative prompt.")

    misc = subparsers.add_parser("misc", help="Run a misc file-in/file-out artifact model.")
    misc.add_argument("model", nargs="?", help="Misc model alias.")
    misc.add_argument("--artifact", "--input-file", dest="artifact", required=True, help="Input artifact path.")
    _add_generation_common_args(misc)
    misc.add_argument("--artifact-type", help="Output MIME type metadata.")

    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    if args.version:
        print(f"utopic generate {__version__}")
        return 0
    if args.kind is None:
        parser.print_help()
        return 0

    try:
        request = _generate_request(args)
        model_id = str(request["model"])
        models.pull_model(model_id)
        status, _headers, body = gateway.handle_openai_request(
            "POST",
            _GENERATE_ENDPOINTS[_canonical_generate_kind(args.kind)],
            request,
        )
        payload = json.loads(body.decode("utf-8"))
        if status >= 400:
            raise RuntimeError(_gateway_error_message(payload))
        _print_generate_result(payload, output=args.output)
        return 0
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"utopic generate: {exc}", file=sys.stderr)
        return 1


def _add_prompt_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", nargs="?", help="Model alias.")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt text.")


def _add_generation_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quality",
        choices=("fast", "balanced", "high"),
        default="balanced",
        help="Default model preset when no model is provided.",
    )
    parser.add_argument("-o", "--output", help="Copy the first generated artifact to this path.")
    parser.add_argument("--seed", type=int, help="Random seed when supported.")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Pass an extra bridge parameter. Values are parsed as JSON when possible.",
    )


def _canonical_generate_kind(kind: str) -> str:
    return "speech" if kind == "tts" else kind


def _generate_request(args: argparse.Namespace) -> dict[str, Any]:
    kind = _canonical_generate_kind(str(args.kind))
    model_id = args.model or _default_generate_model(kind, args.quality)
    entry = models.get_model(model_id)
    if entry is None:
        raise RuntimeError(f"unknown model: {model_id}")
    expected_modality = _GENERATE_MODALITY[kind]
    if entry.modality != expected_modality:
        raise RuntimeError(
            f"model {entry.id} is {entry.modality}, but `utopic generate {kind}` requires {expected_modality}"
        )
    if entry.runtime != "bridge":
        raise RuntimeError(
            f"model {entry.id} is a native text model; use `utopic run` or `utopic chat`"
        )

    request: dict[str, Any] = {"model": entry.id}
    if kind == "speech":
        request["input"] = args.input
    elif kind == "misc":
        request["artifact"] = args.artifact
    else:
        request["prompt"] = args.prompt
    _add_optional_generate_parameters(args, request)
    for value in args.param:
        key, parsed = _parse_generate_param(value)
        request[key] = parsed
    return request


def _default_generate_model(kind: str, quality: str) -> str:
    preferred = _GENERATE_QUALITY_MODELS.get(quality, {}).get(kind)
    if preferred and models.get_model(preferred) is not None:
        return preferred
    expected_modality = _GENERATE_MODALITY[kind]
    for entry in models.list_models():
        if entry.modality == expected_modality and entry.runtime == "bridge":
            return entry.id
    raise RuntimeError(f"no catalog model supports `utopic generate {kind}`")


def _add_optional_generate_parameters(args: argparse.Namespace, request: dict[str, Any]) -> None:
    for attr, key in (
        ("size", "size"),
        ("seed", "seed"),
        ("voice", "voice"),
        ("fps", "fps"),
        ("lyrics", "lyrics"),
        ("duration", "duration"),
        ("sample_rate", "sample_rate"),
        ("guidance_scale", "guidance_scale"),
        ("true_cfg_scale", "true_cfg_scale"),
        ("negative_prompt", "negative_prompt"),
        ("artifact_type", "artifact_type"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            request[key] = value
    steps = getattr(args, "steps", None)
    if steps is not None:
        request["num_inference_steps"] = steps
    frames = getattr(args, "frames", None)
    if frames is not None:
        request["num_frames"] = frames


def _parse_generate_param(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise RuntimeError("--param must be KEY=JSON")
    key, raw = value.split("=", 1)
    if not key:
        raise RuntimeError("--param must include a non-empty key")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def _gateway_error_message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return json.dumps(payload, sort_keys=True)


def _print_generate_result(payload: dict[str, Any], *, output: Optional[str]) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("gateway returned no artifacts")
    first_path: Optional[Path] = None
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        artifact_type = str(artifact.get("type") or "application/octet-stream")
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        path = Path(path_value).expanduser()
        display_path = path
        if index == 0 and output:
            display_path = _copy_generated_artifact(path, Path(output).expanduser())
        if first_path is None:
            first_path = display_path
        print(f"Generated {artifact_type}: {display_path}")
    if first_path is None:
        raise RuntimeError("gateway returned artifacts without local paths")
    progress_url = payload.get("progress_url")
    if isinstance(progress_url, str) and progress_url:
        print(f"Progress: {progress_url}")


def _copy_generated_artifact(source: Path, destination: Path) -> Path:
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def _print_doctor_help() -> None:
    print(
        """usage: utopic doctor

Print local setup diagnostics without cloning, building, downloading, or
starting the native runtime.

Checks:
  - package version
  - cache and binary directories
  - detected backend, device, and reason
  - whether cached native binaries are current
  - required setup tools: cmake and git
  - optional chat tool: Node.js
  - optional bridge engines for image, speech, music, video, and misc artifacts
"""
    )


def _node_status() -> str:
    node = shutil.which("node")
    if node is None:
        return "missing (Python fallback chat remains available)"
    try:
        version = subprocess.check_output(
            [node, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return f"{node} (version check failed)"
    return f"{node} ({version})"


def _bridge_doctor_line(payload: dict[str, object]) -> str:
    engine = str(payload.get("engine") or "unknown")
    status = str(payload.get("status") or "unknown")
    if payload.get("ready") is True:
        return f"  {engine}: ready"
    message = payload.get("message")
    if isinstance(message, str) and message:
        if status == "api_mismatch" and "Failed to import diffusers." in message:
            return (
                f"  {engine}: {status} - installed diffusers/transformers/torch stack is "
                f"incompatible; run utopic-bridge {engine} --check for details"
            )
        return f"  {engine}: {status} - {_first_message_line(message)}"
    missing = payload.get("missing")
    install_hint = payload.get("install_hint")
    details: list[str] = []
    if isinstance(missing, list) and missing:
        details.append("missing " + ", ".join(str(item) for item in missing))
    if isinstance(install_hint, str) and install_hint:
        details.append(install_hint)
    suffix = f" - {'; '.join(details)}" if details else ""
    return f"  {engine}: {status}{suffix}"


def _first_message_line(message: str) -> str:
    for line in message.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return message.strip()


def _bridge_doctor_lines() -> list[str]:
    lines = ["Bridge engines:"]
    for engine in sorted(bridge.ADAPTERS):
        adapter = bridge.ADAPTERS[engine]
        try:
            payload = bridge._check_adapter(adapter)
        except Exception as exc:
            payload = {
                "engine": engine,
                "status": "check_failed",
                "ready": False,
                "message": str(exc) or exc.__class__.__name__,
            }
        lines.append(_bridge_doctor_line(payload))
    return lines


def _doctor(argv: Sequence[str]) -> int:
    if any(arg in ("-h", "--help") for arg in argv):
        _print_doctor_help()
        return 0
    if "--version" in argv:
        print(f"utopic doctor {__version__}")
        return 0

    requested_backend = os.environ.get("UTOPIC_BACKEND", "auto")
    cuda_architectures = os.environ.get("UTOPIC_CUDA_ARCHITECTURES")
    try:
        decision = installer._resolve_backend(requested_backend, cuda_architectures)
    except ValueError as exc:
        print(f"utopic doctor: {exc}", file=sys.stderr)
        return 1

    required_tools = ("cmake", "git")
    missing_required: list[str] = []
    tool_paths = {}
    for name in required_tools:
        path = shutil.which(name)
        tool_paths[name] = path
        if path is None:
            missing_required.append(name)

    print(f"Utopic {__version__}")
    print(f"Cache root: {installer.cache_root()}")
    print(f"Bin dir: {installer.bin_dir()}")
    print(f"Backend: {decision.backend}")
    print(f"Device: {decision.device}")
    print(f"Reason: {decision.reason}")
    if decision.cuda_architectures:
        print(f"CUDA architectures: {decision.cuda_architectures}")
    if decision.cuda_graphs:
        print(f"CUDA graphs: {decision.cuda_graphs}")
    native_cache = (
        "current"
        if installer.native_installation_is_current(installer.BIN_NAMES)
        else "missing or stale"
    )
    print(f"Native cache: {native_cache}")
    for name in required_tools:
        print(f"{name}: {tool_paths[name] or 'missing'}")
    print(f"Node.js: {_node_status()}")
    for line in _bridge_doctor_lines():
        print(line)

    if missing_required:
        print(
            f"Missing required setup tools: {', '.join(missing_required)}",
            file=sys.stderr,
        )
        return 1
    return 0


def _run(argv: Sequence[str]) -> int:
    args = list(argv)
    if any(arg in ("-h", "--help") for arg in args):
        _print_run_help()
        return 0
    if "--version" in args:
        print(f"utopic run {__version__}")
        return 0

    setup_enabled = "--no-setup" not in args
    args = _without_flag(args, "--no-setup")

    try:
        if _has_prompt(args):
            return _run_prompt(args, setup_enabled)

        _validate_run_value_flags(args)
        _validate_model_argument_count(args, _RUN_VALUE_FLAGS)
        _validate_server_options(args)
        model_arg, server_args = _extract_model(args)
        host = _value_after(server_args, "--host", "127.0.0.1")
        port = _value_after(server_args, "--port", "8910")
        if model_arg:
            entry = models.get_model(model_arg)
            if entry is not None and entry.runtime == "bridge":
                models.pull_model(entry.id)
                return _run_gateway_only(host, port, entry)

        _ensure_setup(setup_enabled, "utopic_runner")
        _native.binary_path("utopic_runner")
        model_path = models.ensure_model(model_arg)
        active_model_id = model_arg if model_arg and models.get_model(model_arg) is not None else "utopic"
        return _run_gateway_only(
            host,
            port,
            active_text_model_path=model_path,
            active_text_model_id=active_model_id,
        )
    except RuntimeError as exc:
        print(f"utopic run: {exc}", file=sys.stderr)
        return 1


def main(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_top_help()
        return 0
    if args[0] == "--version":
        print(f"utopic {__version__}")
        return 0

    command = args[0]
    rest = args[1:]
    if command == "setup":
        if "--version" in rest:
            print(f"utopic setup {__version__}")
            return 0
        try:
            return installer.setup(rest)
        except subprocess.CalledProcessError as exc:
            print(
                f"utopic setup: command failed: {_format_command(exc.cmd)}",
                file=sys.stderr,
            )
            return exc.returncode if isinstance(exc.returncode, int) and exc.returncode > 0 else 1
        except RuntimeError as exc:
            print(f"utopic setup: {exc}", file=sys.stderr)
            return 1
    if command == "chat":
        raise SystemExit(chat.launch(rest))
    if command == "models":
        raise SystemExit(models.main(rest))
    if command in {"run", "serve"}:
        return _run(rest)
    if command == "generate":
        return _generate(rest)
    if command == "gateway":
        return gateway.main(rest)
    if command == "mcp":
        return mcp.main(rest)
    if command == "doctor":
        return _doctor(rest)

    if not command.startswith("-"):
        print(f"utopic: unknown command: {command}", file=sys.stderr)
        return 1

    try:
        _validate_legacy_native_options(args)
    except RuntimeError as exc:
        print(f"utopic: {exc}", file=sys.stderr)
        return 1

    _ensure_setup(True)
    _native.main("utopic", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
