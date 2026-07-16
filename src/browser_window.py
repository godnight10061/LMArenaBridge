"""Shared headed-browser window placement for Windows bridge processes."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Literal, Optional


BrowserWindowMode = Literal["background", "visible"]


def browser_window_mode(config: Optional[dict[str, Any]] = None) -> BrowserWindowMode:
    configured = str(
        os.environ.get("LM_BROWSER_WINDOW_MODE")
        or (config or {}).get("browser_window_mode")
        or "background"
    ).strip().lower()
    return "visible" if configured == "visible" else "background"


def chrome_window_args(mode: BrowserWindowMode) -> list[str]:
    if mode == "visible":
        return ["--window-position=0,0"]
    return [
        "--start-minimized",
        "--window-position=-32000,-32000",
    ]


def windows_startupinfo(mode: BrowserWindowMode) -> Optional[subprocess.STARTUPINFO]:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0 if mode == "background" else 5
    return startupinfo
