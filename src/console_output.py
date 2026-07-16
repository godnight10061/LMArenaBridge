"""Terminal-aware output encoding helpers for bridge entry points."""

from __future__ import annotations

from typing import Any

_STATUS_SYMBOL_REPLACEMENTS = (
    ("⚠️", "[WARN]"),
    ("🚀", "[START]"),
    ("✅", "[OK]"),
    ("❌", "[ERROR]"),
    ("⚠", "[WARN]"),
)


def should_force_utf8(*, is_terminal: bool) -> bool:
    """Use UTF-8 for files/pipes while preserving an interactive console code page."""
    return not bool(is_terminal)


def configure_standard_streams(*streams: Any) -> None:
    """Configure redirected text streams as UTF-8 without changing live terminals."""
    for stream in streams:
        if stream is None:
            continue
        try:
            is_terminal = bool(stream.isatty())
        except Exception:
            is_terminal = False
        if not should_force_utf8(is_terminal=is_terminal):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            continue


def render_for_encoding(text: str, encoding: str) -> str:
    """Replace unsupported symbols while retaining readable encodable text."""
    normalized_encoding = str(encoding or "utf-8")
    rendered = str(text)
    for symbol, replacement in _STATUS_SYMBOL_REPLACEMENTS:
        try:
            symbol.encode(normalized_encoding)
        except (LookupError, UnicodeEncodeError):
            rendered = rendered.replace(symbol, replacement)
    try:
        return rendered.encode(normalized_encoding, errors="replace").decode(
            normalized_encoding, errors="replace"
        )
    except LookupError:
        return rendered.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
