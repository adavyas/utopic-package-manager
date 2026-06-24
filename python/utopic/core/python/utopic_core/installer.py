from __future__ import annotations

from typing import Any

from .runtime_env import installer_attribute


def __getattr__(name: str) -> Any:
    return installer_attribute(name)
