import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import installer, models


PACKAGE_DIR = Path(__file__).resolve().parent
NODE_CHAT_SCRIPT = PACKAGE_DIR / "node" / "utopic-chat.js"


CHAT_HELP = """usage: utopic chat [model-alias|/path/to/model.gguf] [options]

Start an Ollama-style terminal chat backed by the local Utopic server.

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
  /clear                Clear this session's conversation.
  /system TEXT          Set or replace the system prompt.
  /exit                 Quit.

Examples:
  utopic chat
  utopic chat dream-7b-q4
  utopic chat -m /path/to/model.gguf -ngl 99
  utopic chat --server http://127.0.0.1:8910
"""


def _binary_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def _server_binary() -> Path:
    return installer.bin_dir() / f"utopic_server{_binary_suffix()}"


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


def _uses_existing_server(argv: Sequence[str]) -> bool:
    return any(arg == "--server" or arg.startswith("--server=") for arg in argv)


def _node_command(argv: Sequence[str]) -> list[str]:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError(
            "Node.js was not found on PATH. Install Node.js, then rerun `utopic chat`."
        )
    script = _chat_script()
    if not script.exists():
        raise RuntimeError(f"Bundled Utopic chat app was not found: {script}")
    return [node, str(script), *argv]


def launch(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if _wants_help(args):
        print(CHAT_HELP)
        return 0

    try:
        command = _node_command(args)
        if _wants_setup(args) and not _server_binary().exists():
            code = installer.setup([])
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
