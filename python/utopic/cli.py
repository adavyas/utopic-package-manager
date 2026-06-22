import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence

from . import _native, chat, installer, models


_RUN_VALUE_FLAGS = {"--host", "--port", "-ngl", "--ctx-size"}


def _binary_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def _native_binary_exists(name: str) -> bool:
    return (installer.bin_dir() / f"{name}{_binary_suffix()}").exists()


def _ensure_setup(enabled: bool = True, binary_name: str = "utopic") -> None:
    if enabled and not _native_binary_exists(binary_name):
        code = installer.setup([])
        if code != 0:
            raise SystemExit(code)


def _has_prompt(args: Sequence[str]) -> bool:
    for arg in args:
        if arg in ("-p", "--prompt") or arg.startswith("--prompt="):
            return True
    return False


def _without_flag(args: Sequence[str], flag: str) -> list[str]:
    return [arg for arg in args if arg != flag]


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


def _value_after(args: Sequence[str], flag: str, default: str) -> str:
    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def _client_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def _server_url(host: str, port: str) -> str:
    return f"http://{_client_host(host)}:{port}/v1/chat/completions"


def _server_health_url(host: str, port: str) -> str:
    return f"http://{_client_host(host)}:{port}/health"


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


def _run_server(model_path: str, server_args: Sequence[str], host: str, port: str) -> int:
    exe = _native.binary_path("utopic_server")
    process = subprocess.Popen([str(exe), "-m", str(model_path), *server_args])
    try:
        _wait_for_health(process, _server_health_url(host, port))
        print(f"OpenAI-compatible URL: {_server_url(host, port)}", flush=True)
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return 130
    except RuntimeError as exc:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print(f"utopic run: {exc}", file=sys.stderr)
        code = process.returncode
        return code if code and code > 0 else 1


def _print_top_help() -> None:
    print(
        """usage: utopic <command> [options]

Commands:
  chat      Start the bundled chat TUI. Runs setup on first use.
  run       Start an OpenAI-compatible server, or run one-shot prompts with -p.
  setup     Build and cache native binaries for this host.
  models    List, pull, and locate curated GGUF models.

Examples:
  utopic chat
  utopic chat dream-7b-q4
  utopic run dream-7b-q4 --port 8910 -ngl 99
  utopic run -m /path/to/model.gguf -p "Answer with one word: 2+2?" -n 16

Run `utopic <command> --help` for command-specific help.
"""
    )


def _print_run_help() -> None:
    print(
        """usage: utopic run [model-alias|/path/to/model.gguf] [server options]
       utopic run -m model.gguf -p prompt [native one-shot options]

Without -p/--prompt, `utopic run` starts `utopic-server` and prints the local
OpenAI-compatible URL. With -p/--prompt, it keeps the native one-shot behavior.

Server options:
  -m, --model VALUE     Model alias or GGUF path. Defaults to the recommended model.
  --host HOST           Bind host for the local server. Default: 127.0.0.1
  --port PORT           Bind port for the local server. Default: 8910
  -ngl N                GPU layers to offload. Default: native default
  --ctx-size N          Server context size. Default: native default
  --no-setup            Do not run setup automatically if binaries are missing.

Examples:
  utopic run dream-7b-q4
  utopic run -m /path/to/model.gguf --port 8910 -ngl 99
  utopic run -m /path/to/model.gguf -p "Hello" -n 128
"""
    )


def _run(argv: Sequence[str]) -> int:
    args = list(argv)
    if args and args[0] in ("-h", "--help"):
        _print_run_help()
        return 0

    setup_enabled = "--no-setup" not in args
    args = _without_flag(args, "--no-setup")

    try:
        if _has_prompt(args):
            _ensure_setup(setup_enabled)
            _native.main("utopic", args)
            return 0

        model_arg, server_args = _extract_model(args)
        _ensure_setup(setup_enabled, "utopic_server")
        model_path = models.ensure_model(model_arg)
        host = _value_after(server_args, "--host", "127.0.0.1")
        port = _value_after(server_args, "--port", "8910")
        return _run_server(str(model_path), server_args, host, port)
    except RuntimeError as exc:
        print(f"utopic run: {exc}", file=sys.stderr)
        return 1


def main(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        _print_top_help()
        return 0

    command = args[0]
    rest = args[1:]
    if command == "setup":
        raise SystemExit(installer.setup(rest))
    if command == "chat":
        raise SystemExit(chat.launch(rest))
    if command == "models":
        raise SystemExit(models.main(rest))
    if command == "run":
        return _run(rest)

    _ensure_setup(True)
    _native.main("utopic", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
