from __future__ import annotations

import sys

from . import installer as _installer
from .core_loader import load_core_module


sys.modules[__name__] = load_core_module("chat", installer_api=_installer)
