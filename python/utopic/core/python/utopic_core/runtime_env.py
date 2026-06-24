from __future__ import annotations

from typing import Any


_installer_api: object | None = None


def configure(*, installer_api: object) -> None:
    global _installer_api
    _installer_api = installer_api


def require_installer() -> object:
    if _installer_api is None:
        raise RuntimeError("utopic_core runtime environment is not configured")
    return _installer_api


def installer_attribute(name: str) -> Any:
    return getattr(require_installer(), name)
