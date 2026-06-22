import sys

from ._native import main as _main


HELP = """usage: utopic-server -m model.gguf [options]

Start the Utopic OpenAI-compatible HTTP server.

Common options:
  -m VALUE             GGUF model path.
  --host HOST          Bind host. Default: 127.0.0.1
  --port PORT          Bind port. Default: 8910
  -ngl N               GPU layers to offload.
  --ctx-size N         Context size.
  -h, --help           Show this help.

Run `utopic setup` first if the native runtime is not installed.
"""


def main() -> int:
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(HELP)
        return 0
    try:
        _main("utopic_server")
        return 0
    except RuntimeError as exc:
        print(f"utopic-server: {exc}", file=sys.stderr)
        return 1
