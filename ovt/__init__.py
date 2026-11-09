"""Offline Video Translator engine (UI-independent, no network by default)."""
__version__ = "0.1.0"

import sys as _sys

if _sys.platform == "win32" and getattr(_sys, "frozen", False):      # packaged program: keep every bundled DLL folder searchable
    from .winlibs import prepare_dll_search as _prepare

    _prepare()
