from __future__ import annotations

import sys

from . import installer as _installer
from .core_loader import load_core_module


_core = load_core_module("gateway", installer_api=_installer)

if __name__ == "__main__":
    raise SystemExit(_core.main())

sys.modules[__name__] = _core
