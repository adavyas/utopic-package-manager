import json
import sys
from typing import Any, Optional, TextIO

from . import __version__
from ._native import main as _main
from . import gateway


HELP = """usage: utopic-mcp [--native native options | --native-base-url URL]

Start the Utopic Model Context Protocol server.

Modes:
  utopic-mcp
      Start the Python stdio MCP gateway exposing the full local Utopic tool
      catalog: chat, image, speech, music, video, misc, model list/check/pull.

  utopic-mcp --native [native options]
      Start the native C++ stdio MCP server exposing diffusion_generate for one
      GGUF model. Pass native options such as -m, -ngl, and --ctx-size.

  utopic-mcp [--runtime] [--native-base-url URL]
      Start the Python stdio MCP gateway. --runtime is accepted for older docs.
      Use --native-base-url to forward text calls to a running utopic-server.

Run `utopic setup` first if the native runtime is not installed. Most users
should start with `utopic chat` or `utopic run`; this launcher is for MCP
integrations such as Claude Code that need stdio MCP directly.
"""


def _value_after(args: list[str], flag: str) -> Optional[str]:
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 >= len(args) or args[index + 1].startswith("-"):
                raise RuntimeError(f"expected a value after {flag}")
            return args[index + 1]
        prefix = f"{flag}="
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            if not value:
                raise RuntimeError(f"expected a value after {flag}")
            return value
    return None


def _without_flag(args: list[str], flag: str) -> list[str]:
    return [arg for arg in args if arg != flag]


def _has_legacy_native_args(args: list[str]) -> bool:
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--runtime", "--gateway"}:
            continue
        if arg == "--native-base-url":
            skip_next = True
            continue
        if arg.startswith("--native-base-url="):
            continue
        return True
    return False


def _runtime_stdio(stdin: TextIO, stdout: TextIO, *, native_base_url: Optional[str]) -> int:
    for line in stdin:
        text = line.strip()
        if not text:
            continue
        try:
            request = json.loads(text)
            if not isinstance(request, dict):
                raise ValueError("MCP request must be a JSON object")
            _status, _headers, body = gateway.handle_mcp_request(
                request,
                native_base_url=native_base_url,
            )
            if request.get("id") is None:
                continue
            response: Any = json.loads(body.decode("utf-8"))
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc)},
            }
        stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        stdout.flush()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg == "--version" for arg in args):
        print(f"utopic-mcp {__version__}")
        return 0
    if any(arg in ("-h", "--help") for arg in args):
        print(HELP)
        return 0
    if "--native" in args or _has_legacy_native_args(args):
        native_args = _without_flag(args, "--native")
        try:
            _main("utopic_mcp", native_args)
            return 0
        except RuntimeError as exc:
            print(f"utopic-mcp: {exc}", file=sys.stderr)
            return 1
    if "--runtime" in args or "--gateway" in args or not args or any(arg.startswith("--native-base-url") for arg in args):
        try:
            native_base_url = _value_after(args, "--native-base-url")
        except RuntimeError as exc:
            print(f"utopic-mcp: {exc}", file=sys.stderr)
            return 2
        return _runtime_stdio(
            sys.stdin,
            sys.stdout,
            native_base_url=native_base_url,
        )
    return 0
