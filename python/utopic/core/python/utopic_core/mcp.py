import sys

from . import __version__
from ._native import main as _main


HELP = """usage: utopic-mcp [native options]

Start the Utopic Model Context Protocol server.

Run `utopic setup` first if the native runtime is not installed. Most users
should start with `utopic chat` or `utopic run`; this launcher is for MCP
integrations that need the native protocol server directly.
"""


def main() -> int:
    if any(arg == "--version" for arg in sys.argv[1:]):
        print(f"utopic-mcp {__version__}")
        return 0
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(HELP)
        return 0
    try:
        _main("utopic_mcp")
        return 0
    except RuntimeError as exc:
        print(f"utopic-mcp: {exc}", file=sys.stderr)
        return 1
