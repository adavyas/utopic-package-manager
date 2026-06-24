from __future__ import annotations

import importlib
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGED_CORE_PYTHON = PACKAGE_DIR / "core" / "python"


def _ensure_core_path() -> None:
    path = str(PACKAGED_CORE_PYTHON)
    if PACKAGED_CORE_PYTHON.exists() and path not in sys.path:
        sys.path.insert(0, path)


def load_core_module(name: str, *, installer_api: object | None = None) -> object:
    _ensure_core_path()
    runtime_env = importlib.import_module("utopic_core.runtime_env")
    if installer_api is not None:
        runtime_env.configure(installer_api=installer_api)
    return importlib.import_module(f"utopic_core.{name}")
