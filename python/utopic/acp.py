import sys

from ._native import main as _main


HELP = """usage: utopic-acp [native options]

Start the Utopic Agent Client Protocol server.

Run `utopic setup` first if the native runtime is not installed. Most users
should start with `utopic chat` or `utopic run`; this launcher is for ACP
integrations that need the native protocol server directly.
"""


def main() -> int:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(HELP)
        return 0
    try:
        _main("utopic_acp")
        return 0
    except RuntimeError as exc:
        print(f"utopic-acp: {exc}", file=sys.stderr)
        return 1
