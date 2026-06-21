import os
import sys
from pathlib import Path
from typing import Sequence


PACKAGE_DIR = Path(__file__).resolve().parent


def binary_path(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = PACKAGE_DIR / "bin" / f"{name}{suffix}"
    if not path.exists():
        raise RuntimeError(
            f"Utopic native binary was not installed: {path}. "
            "Reinstall the package with a compatible llama.cpp build available."
        )
    return path


def main(binary_name: str, argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    exe = binary_path(binary_name)
    os.execv(str(exe), [str(exe), *args])
