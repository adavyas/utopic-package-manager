import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

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
        raise RuntimeError(
            "Node.js was not found on PATH. Install Node.js, then rerun `utopic chat`."
        )
    script = _chat_script()
    if not script.exists():
        raise RuntimeError(f"Bundled Utopic chat app was not found: {script}")
    return [node, str(script), *argv]


def _format_command(command: object) -> str:
    if isinstance(command, (list, tuple)):
        return shlex.join(str(part) for part in command)
    return str(command)


def launch(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if _wants_help(args):
        print(CHAT_HELP)
        return 0

    try:
        _validate_value_args(args)
        command = _node_command(args)
        if _wants_setup(args) and not installer.native_installation_is_current(("utopic_server",)):
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
