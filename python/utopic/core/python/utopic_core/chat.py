import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from . import installer, models


PACKAGE_DIR = Path(__file__).resolve().parent
NODE_CHAT_SCRIPT = PACKAGE_DIR / "node" / "utopic-chat.js"
VALUE_FLAGS = {
    "-m": "-m/--model",
    "--model": "-m/--model",
    "--server": "--server",
    "--host": "--host",
    "--port": "--port",
    "-ngl": "-ngl",
    "--ctx-size": "--ctx-size",
    "--max-tokens": "--max-tokens",
    "--temperature": "--temperature",
}
LONG_VALUE_FLAGS = {
    "--model": "-m/--model",
    "--server": "--server",
    "--host": "--host",
    "--port": "--port",
    "--ctx-size": "--ctx-size",
    "--max-tokens": "--max-tokens",
    "--temperature": "--temperature",
}
NUMERIC_VALUE_FLAGS = {"--port", "-ngl", "--ctx-size", "--max-tokens", "--temperature"}
INTEGER_VALUE_RULES = {
    "--port": (1, 65535, "an integer from 1 to 65535"),
    "-ngl": (0, None, "a non-negative integer"),
    "--ctx-size": (1, None, "a positive integer"),
    "--max-tokens": (1, None, "a positive integer"),
}
NUMBER_VALUE_RULES = {
    "--temperature": (0.0, None, "a non-negative number"),
}
MIN_NODE_MAJOR = 18
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer naturally, directly, and with useful specifics. "
    "When asked what you can do, give a concise overview with examples across "
    "writing, research, coding, math, planning, and conversation. Do not claim access to "
    "the internet, private files, or real-world actions unless the user provides tools or context."
)


CHAT_HELP = """usage: utopic chat [model-alias|/path/to/model.gguf] [options]

Start an Ollama-style terminal chat backed by the local Utopic server.
Uses the bundled TypeScript/Node TUI when Node.js 18+ is available; otherwise
falls back to a minimal built-in Python chat loop.

Options:
  -m, --model VALUE     Model alias or GGUF path.
  --server URL          Connect to an existing OpenAI-compatible Utopic server.
  --host HOST           Host for an auto-started server. Default: 127.0.0.1
  --port PORT           Port for an auto-started server. Default: 8910
  -ngl N                GPU layers for an auto-started server. Default: 99
  --ctx-size N          Context size for an auto-started server. Default: 4096
  --max-tokens N        Max response tokens. Default: 512
  --temperature N       Sampling temperature. Default: 0
  --no-setup            Skip Python-side first-use setup.
  -h, --help            Show this help.

Chat commands:
  /help                 Show chat commands.
  /models               Show native text chat models.
  /pull MODEL           Download a native text model. Restart chat to switch.
  /serve                Show local OpenAI-compatible endpoints.
  /clear                Clear this session's conversation.
  /system TEXT          Set or replace the system prompt.
  /exit                 Quit.

Examples:
  utopic chat
  utopic chat diffusiongemma-26b-a4b-q4
  utopic chat -m /path/to/model.gguf -ngl 99
  utopic chat --server http://127.0.0.1:8910
"""


class NodeUnavailable(RuntimeError):
    pass


def _chat_script() -> Path:
    return NODE_CHAT_SCRIPT


def _wants_setup(argv: Sequence[str]) -> bool:
    if _wants_help(argv):
        return False
    if _uses_existing_server(argv):
        return False
    return "--no-setup" not in argv


def _wants_help(argv: Sequence[str]) -> bool:
    return any(arg in ("-h", "--help") for arg in argv)


def _wants_version(argv: Sequence[str]) -> bool:
    return any(arg == "--version" for arg in argv)


def _uses_existing_server(argv: Sequence[str]) -> bool:
    return any(arg == "--server" or arg.startswith("--server=") for arg in argv)


def _validate_server_url_arg(args: Sequence[str]) -> None:
    server = _value_after(args, "--server", "")
    if not server:
        return
    if "://" not in server:
        raise RuntimeError("--server must be a URL")
    try:
        parsed = urllib.parse.urlsplit(server)
    except ValueError as exc:
        raise RuntimeError("--server must be a URL") from exc
    if parsed.scheme not in {"http", "https"}:
        if parsed.scheme:
            raise RuntimeError("--server must use http:// or https://")
        raise RuntimeError("--server must be a URL")
    if not parsed.netloc:
        raise RuntimeError("--server must be a URL")


def _looks_like_negative_number(value: str) -> bool:
    return len(value) > 1 and value[0] == "-" and value[1].isdigit()


def _validate_numeric_value(flag: str, value: str) -> None:
    if flag in INTEGER_VALUE_RULES:
        minimum, maximum, label = INTEGER_VALUE_RULES[flag]
        try:
            parsed = int(value)
        except ValueError as exc:
            raise RuntimeError(f"{flag} must be {label}") from exc
        if parsed < minimum or (maximum is not None and parsed > maximum):
            raise RuntimeError(f"{flag} must be {label}")
    if flag in NUMBER_VALUE_RULES:
        minimum, maximum, label = NUMBER_VALUE_RULES[flag]
        try:
            parsed = float(value)
        except ValueError as exc:
            raise RuntimeError(f"{flag} must be {label}") from exc
        if not math.isfinite(parsed) or parsed < minimum or (maximum is not None and parsed > maximum):
            raise RuntimeError(f"{flag} must be {label}")


def _validate_value_args(argv: Sequence[str]) -> None:
    model_args = 0
    for index, arg in enumerate(argv):
        value_for_previous_flag = index > 0 and argv[index - 1] in VALUE_FLAGS
        if arg in VALUE_FLAGS:
            label = VALUE_FLAGS[arg]
            if index + 1 >= len(argv):
                raise RuntimeError(f"expected a value after {label}")
            value = argv[index + 1]
            allow_negative = arg in NUMERIC_VALUE_FLAGS
            if value == "" or (
                value.startswith("-")
                and not (allow_negative and _looks_like_negative_number(value))
            ):
                raise RuntimeError(f"expected a value after {label}")
            _validate_numeric_value(arg, value)
            if arg in ("-m", "--model"):
                model_args += 1

        for flag, label in LONG_VALUE_FLAGS.items():
            if not arg.startswith(f"{flag}="):
                continue
            value = arg.split("=", 1)[1]
            allow_negative = flag in NUMERIC_VALUE_FLAGS
            if value == "" or (
                value.startswith("-")
                and not (allow_negative and _looks_like_negative_number(value))
            ):
                raise RuntimeError(f"expected a value after {label}")
            _validate_numeric_value(flag, value)
            if flag == "--model":
                model_args += 1

        if not arg.startswith("-") and (
            index == 0 or argv[index - 1] not in VALUE_FLAGS
        ):
            model_args += 1
        if (
            arg.startswith("-")
            and not value_for_previous_flag
            and arg not in VALUE_FLAGS
            and arg not in ("-h", "--help", "--no-setup")
            and not any(arg.startswith(f"{flag}=") for flag in LONG_VALUE_FLAGS)
        ):
            raise RuntimeError(f"unknown option: {arg}")

    if model_args > 1:
        raise RuntimeError("expected at most one model argument")


def _node_command(argv: Sequence[str]) -> list[str]:
    node = shutil.which("node")
    if node is None:
        raise NodeUnavailable(
            "Node.js was not found on PATH. Install Node.js, then rerun `utopic chat`."
        )
    _ensure_node_version(node)
    script = _chat_script()
    if not script.exists():
        raise RuntimeError(f"Bundled Utopic chat app was not found: {script}")
    return [node, str(script), *argv]


def _ensure_node_version(node: str) -> None:
    try:
        output = subprocess.check_output(
            [node, "--version"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NodeUnavailable(
            "Could not determine Node.js version. Node.js 18 or newer is required for `utopic chat`."
        ) from exc

    major = _parse_node_major(output)
    if major is None:
        raise NodeUnavailable(
            f"Could not determine Node.js version from {output!r}. Node.js 18 or newer is required for `utopic chat`."
        )
    if major < MIN_NODE_MAJOR:
        raise NodeUnavailable(
            f"Node.js {MIN_NODE_MAJOR} or newer is required; found {output}"
        )


def _parse_node_major(value: str) -> Optional[int]:
    version = value.strip()
    if version.startswith("v"):
        version = version[1:]
    major = version.split(".", 1)[0]
    try:
        return int(major)
    except ValueError:
        return None


def _format_command(command: object) -> str:
    if isinstance(command, (list, tuple)):
        return shlex.join(str(part) for part in command)
    return str(command)


def _value_after(args: Sequence[str], flag: str, default: str) -> str:
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def _model_arg(args: Sequence[str]) -> Optional[str]:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-m", "--model"):
            return args[index + 1] if index + 1 < len(args) else None
        if arg.startswith("--model="):
            return arg.split("=", 1)[1]
        if arg in VALUE_FLAGS:
            index += 2
            continue
        if any(arg.startswith(flag + "=") for flag in LONG_VALUE_FLAGS):
            index += 1
            continue
        if not arg.startswith("-"):
            return arg
        index += 1
    return None


def _choose_model_arg(args: Sequence[str]) -> Optional[str]:
    existing = _model_arg(args)
    if existing or not sys.stdin.isatty():
        return existing

    catalog = _chat_model_catalog()
    if not catalog:
        raise RuntimeError("No native text chat models are available in this catalog.")
    recommended = next((entry for entry in catalog if entry.recommended), catalog[0])

    print("\nAvailable chat models:")
    for index, entry in enumerate(catalog, start=1):
        marker = "*" if entry.recommended else " "
        status = "downloaded" if models.is_model_downloaded(entry) else "not downloaded"
        print(f"{index}. {marker} {entry.id} ({entry.size}, {status})")
        print(f"   {entry.name}")

    try:
        answer = input(f"\nChoose a model [{recommended.id}]: ").strip()
    except EOFError:
        print()
        return recommended.id
    if not answer:
        return recommended.id
    try:
        selected = int(answer)
    except ValueError:
        return answer
    if 1 <= selected <= len(catalog):
        return catalog[selected - 1].id
    return answer


def _is_chat_model(entry: object) -> bool:
    return getattr(entry, "modality", "text") == "text" and getattr(entry, "runtime", "native") == "native"


def _chat_model_catalog() -> list[object]:
    return [entry for entry in models.list_models() if _is_chat_model(entry)]


def _server_base_url(args: Sequence[str]) -> Optional[str]:
    server = _value_after(args, "--server", "")
    if not server:
        return None
    parsed = urllib.parse.urlsplit(server)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        parsed = parsed._replace(
            path=path[: -len("/v1/chat/completions")] or "",
            query="",
            fragment="",
        )
    elif path.endswith("/v1"):
        parsed = parsed._replace(
            path=path[: -len("/v1")] or "",
            query="",
            fragment="",
        )
    return urllib.parse.urlunsplit(parsed).rstrip("/")


def _chat_completions_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path == "/v1/chat/completions":
        parsed = parsed._replace(query="", fragment="")
    elif path == "/v1":
        parsed = parsed._replace(path="/v1/chat/completions", query="", fragment="")
    else:
        parsed = parsed._replace(
            path=f"{path}/v1/chat/completions" if path else "/v1/chat/completions",
            query="",
            fragment="",
        )
    return urllib.parse.urlunsplit(parsed)


def _server_health_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/health"


def _server_binary() -> Path:
    name = "utopic_server.exe" if sys.platform == "win32" else "utopic_server"
    binary = installer.bin_dir() / name
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("Utopic native binaries are missing. Run `utopic setup`, then retry.")
    return binary


def _local_server_base(args: Sequence[str]) -> str:
    host = _value_after(args, "--host", "127.0.0.1")
    port = _value_after(args, "--port", "8910")
    client_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    return f"http://{client_host}:{port}"


def _server_args(args: Sequence[str]) -> list[str]:
    server_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("-m", "--model", "--server", "--max-tokens", "--temperature"):
            index += 2
            continue
        if (
            arg.startswith("--model=")
            or arg.startswith("--server=")
            or arg.startswith("--max-tokens=")
            or arg.startswith("--temperature=")
        ):
            index += 1
            continue
        if arg in {"--host", "--port", "-ngl", "--ctx-size"}:
            server_args.extend([arg, args[index + 1]])
            index += 2
            continue
        if any(arg.startswith(flag + "=") for flag in ("--host", "--port", "--ctx-size")):
            flag, value = arg.split("=", 1)
            server_args.extend([flag, value])
            index += 1
            continue
        if arg == "--no-setup":
            index += 1
            continue
        if not arg.startswith("-"):
            index += 1
            continue
        server_args.append(arg)
        index += 1
    return server_args


def _wait_for_health(process: subprocess.Popen, health_url: str, log_path: Path, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            detail = f"signal {-code}" if code < 0 else f"code {code}"
            raise RuntimeError(
                f"utopic-server exited before it became healthy ({detail}); see {log_path}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.3)
    raise RuntimeError(f"timed out waiting for {health_url}; see {log_path}")


def _chat_model_id(value: Optional[str]) -> str:
    if value and models.get_model(value) is not None:
        return value
    if value is None:
        return models.default_model().id
    return "utopic"


def _request_chat_completion(
    base_url: str,
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _chat_completions_url(base_url),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("server returned an invalid chat completion response") from exc
    if not isinstance(content, str):
        raise RuntimeError("server returned a non-text chat completion response")
    return content


def _python_chat_loop(
    base_url: str,
    args: Sequence[str],
    fallback_reason: str = "Node.js was not found",
    model: Optional[str] = None,
) -> int:
    max_tokens = int(_value_after(args, "--max-tokens", "512"))
    temperature = float(_value_after(args, "--temperature", "0"))
    model_arg = _model_arg(args)
    model_id = model or (_chat_model_id(model_arg) if model_arg else "utopic")
    messages: list[dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    print(f"utopic chat: {fallback_reason}; using the built-in Python chat fallback.")
    print(f"OpenAI-compatible URL: {_chat_completions_url(base_url)}")
    print("Type /help for commands, /models for chat models, /serve for endpoints, /exit to quit.")

    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            print()
            return 0
        text = prompt.strip()
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return 0
        if text == "/help":
            _print_python_chat_commands()
            continue
        if text == "/models":
            _print_python_chat_models()
            continue
        if text == "/serve":
            _print_python_chat_endpoints(base_url)
            continue
        if text.startswith("/pull "):
            model_to_pull = text.removeprefix("/pull ").strip()
            if not model_to_pull:
                print("usage: /pull MODEL", file=sys.stderr)
                continue
            try:
                _pull_python_chat_model(model_to_pull)
            except RuntimeError as exc:
                print(f"utopic chat: {exc}", file=sys.stderr)
            continue
        if text == "/clear":
            messages.clear()
            messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
            print("Conversation cleared.")
            continue
        if text.startswith("/system "):
            system_text = text.removeprefix("/system ").strip()
            messages = [message for message in messages if message["role"] != "system"]
            if system_text:
                messages.insert(0, {"role": "system", "content": system_text})
            print("System prompt updated.")
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            print("Thinking...")
            answer = _request_chat_completion(
                base_url,
                messages,
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except (OSError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"utopic chat: request failed: {exc}", file=sys.stderr)
            return 1
        messages.append({"role": "assistant", "content": answer})
        print(f"{answer}\n")


def _print_python_chat_commands() -> None:
    print("/models       Show native text chat models.")
    print("/pull MODEL   Download a native text model. Restart chat to switch.")
    print("/serve        Show local OpenAI-compatible endpoints.")
    print("/clear        Clear conversation history.")
    print("/system TEXT  Set or replace the system prompt.")
    print("/exit         Quit.")


def _print_python_chat_models() -> None:
    entries = _chat_model_catalog()
    if not entries:
        print("\nNo native text chat models are available in this catalog.")
        return
    print("\nAvailable chat models:")
    for index, entry in enumerate(entries, start=1):
        marker = "*" if entry.recommended else " "
        status = "downloaded" if models.is_model_downloaded(entry) else "not downloaded"
        print(f"{index}. {marker} {entry.id} ({entry.size}, {status})")
        print(f"   {entry.name}")


def _print_python_chat_endpoints(base_url: str) -> None:
    clean_base = base_url.rstrip("/")
    print(f"Chat completions: {clean_base}/v1/chat/completions")
    print(f"Models: {clean_base}/v1/models")
    print(f"MCP: {clean_base}/mcp")


def _pull_python_chat_model(model_id: str) -> None:
    entry = models.get_model(model_id)
    if entry is None:
        raise RuntimeError(f"unknown model '{model_id}'. Run /models to see aliases.")
    if entry.modality != "text" or entry.runtime != "native":
        raise RuntimeError(
            f"model '{model_id}' is {entry.modality} / {entry.runtime} / {entry.native_status}; "
            "use `utopic serve` or the runtime gateway for non-chat modalities."
        )
    model_path = models.ensure_model(model_id)
    print(f"Pulled {model_id} to {model_path}")
    print("Restart chat with that model to switch the running server.")


def _python_fallback_launch(
    argv: Sequence[str],
    fallback_reason: str = "Node.js was not found",
) -> int:
    args = list(argv)

    selected_model = _choose_model_arg(args)

    def chat_loop(base_url: str, chat_model: str) -> int:
        if fallback_reason == "Node.js was not found":
            return _python_chat_loop(base_url, args, model=chat_model)
        return _python_chat_loop(base_url, args, fallback_reason, model=chat_model)

    existing_server = _server_base_url(args)
    if existing_server:
        return chat_loop(
            existing_server,
            _chat_model_id(selected_model) if selected_model else "utopic",
        )

    models.ensure_model(selected_model)
    run_model = selected_model or models.default_model().id
    chat_model = _chat_model_id(run_model)
    base_url = _local_server_base(args)
    log_dir = installer.cache_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "utopic-chat-server.log"
    command = [
        os.environ.get("UTOPIC_CLI", "utopic"),
        "run",
        run_model,
        *_server_args(args),
        "--no-setup",
    ]
    print(f"Starting Utopic server: {base_url}")
    print(f"Server log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        _wait_for_health(process, _server_health_url(base_url), log_path)
        return chat_loop(base_url, chat_model)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def launch(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if _wants_help(args):
        print(CHAT_HELP)
        return 0
    if _wants_version(args):
        print(f"utopic chat {__version__}")
        return 0

    try:
        _validate_value_args(args)
        _validate_server_url_arg(args)
        if shutil.which("node") is None:
            if _wants_setup(args) and not installer.native_installation_is_current(("utopic_runner",)):
                try:
                    code = installer.setup([])
                except subprocess.CalledProcessError as exc:
                    print(
                        f"utopic chat: setup command failed: {_format_command(exc.cmd)}",
                        file=sys.stderr,
                    )
                    return exc.returncode if isinstance(exc.returncode, int) and exc.returncode > 0 else 1
                if code != 0:
                    return code
            return _python_fallback_launch(args)
        try:
            command = _node_command(args)
        except NodeUnavailable as exc:
            if _wants_setup(args) and not installer.native_installation_is_current(("utopic_runner",)):
                try:
                    code = installer.setup([])
                except subprocess.CalledProcessError as setup_exc:
                    print(
                        f"utopic chat: setup command failed: {_format_command(setup_exc.cmd)}",
                        file=sys.stderr,
                    )
                    return setup_exc.returncode if isinstance(setup_exc.returncode, int) and setup_exc.returncode > 0 else 1
                if code != 0:
                    return code
            return _python_fallback_launch(args, fallback_reason=str(exc))
        if _wants_setup(args) and not installer.native_installation_is_current(("utopic_runner",)):
            try:
                code = installer.setup([])
            except subprocess.CalledProcessError as exc:
                print(
                    f"utopic chat: setup command failed: {_format_command(exc.cmd)}",
                    file=sys.stderr,
                )
                return exc.returncode if isinstance(exc.returncode, int) and exc.returncode > 0 else 1
            if code != 0:
                return code

        env = os.environ.copy()
        env.setdefault("UTOPIC_BIN_DIR", str(installer.bin_dir()))
        env.setdefault("UTOPIC_MODELS_DIR", str(models.models_dir()))
        env.setdefault("UTOPIC_MODELS_CATALOG", str(models.catalog_path()))
        subprocess.run(command, env=env, check=True)
        return 0
    except RuntimeError as exc:
        print(f"utopic chat: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode
