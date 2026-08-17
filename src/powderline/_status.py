"""Console status markers with an ASCII fallback for legacy code pages.

Emoji render fine on modern UTF-8 consoles but turn into mojibake when captured
under a legacy Windows code page (e.g. cp1252/cp437). Support is resolved once at
import: use emoji when stdout can encode them, otherwise plain ASCII.
"""
from __future__ import annotations

import sys


def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "✅⚠️❌ℹ️🚀🐢".encode(enc)
    except (LookupError, TypeError, UnicodeError):
        return False
    return True


UNICODE_OK = _supports_unicode()


def emoji(char: str, fallback: str = "") -> str:
    """Return ``char`` when the console supports Unicode, else ``fallback``."""
    return char if UNICODE_OK else fallback


CHECK = emoji("✅", "[OK]")
WARN = emoji("⚠️", "[!]")
CROSS = emoji("❌", "[X]")
INFO = emoji("ℹ️", "[i]")
