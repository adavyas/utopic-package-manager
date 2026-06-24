import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import installer


def _binary_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def binary_path(name: str) -> Path:
    suffix = _binary_suffix()
    path = installer.bin_dir() / f"{name}{suffix}"
    if not path.exists():
        raise RuntimeError(
            f"Utopic native binary is not installed: {path}. "
            "Run `utopic setup` to build and cache the native runtime."
        )
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"Utopic native binary is not an executable file: {path}. "
            "Run `utopic setup --force` to rebuild the native runtime."
        )
    return path


def main(binary_name: str, argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    exe = binary_path(binary_name)
    os.execv(str(exe), [str(exe), *args])
