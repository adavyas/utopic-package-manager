import sys

from ._native import main as _main


def main() -> int:
    try:
        _main("utopic_mcp")
        return 0
    except RuntimeError as exc:
        print(f"utopic-mcp: {exc}", file=sys.stderr)
        return 1
