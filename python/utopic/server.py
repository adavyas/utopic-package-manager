import sys

from ._native import main as _main


def main() -> int:
    try:
        _main("utopic_server")
        return 0
    except RuntimeError as exc:
        print(f"utopic-server: {exc}", file=sys.stderr)
        return 1
