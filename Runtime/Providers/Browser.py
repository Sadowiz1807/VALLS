from __future__ import annotations

import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from Runtime.Resources.Registry import resolve


def enum_windows() -> list[tuple[int, str]]:
    import win32gui
    windows = []
    win32gui.EnumWindows(lambda handle, _: windows.append((handle, win32gui.GetWindowText(handle))) if win32gui.IsWindowVisible(handle) else None, None)
    return windows


def close_tabs(handle: int, title: str) -> int:
    from pywinauto import Desktop
    closed = 0
    for _ in range(20):
        window = Desktop(backend="uia").window(handle=handle)
        tabs = [tab for tab in window.descendants(control_type="TabItem") if title.casefold() in tab.window_text().casefold()]
        if not tabs:
            return closed
        buttons = [button for tab in tabs for button in tab.descendants(control_type="Button")
                   if button.element_info.class_name == "TabCloseButton"]
        if len(buttons) != 1:
            raise RuntimeError("TAB_CLOSE_CONTROL_AMBIGUOUS")
        buttons[0].invoke()
        closed += 1
        time.sleep(0.5)
    raise RuntimeError("TAB_CLOSE_LIMIT_EXCEEDED")


class BrowserProvider:
    def __init__(self, registry: Path, opener: Callable[[str], bool] = webbrowser.open,
                 runner: Callable = subprocess.Popen):
        self.registry = registry
        self.opener = opener
        self.runner = runner

    def open(self, url: str, browser: str | None = None, execute: bool = False) -> dict:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {"ok": False, "resource_id": "browser.navigation.open", "error": "URL_INVALID"}
        selected = resolve(self.registry, browser) if browser else None
        if browser and not selected:
            return {"ok": False, "resource_id": "browser.navigation.open", "error": "BROWSER_UNSUPPORTED", "ask_to_add_whitelist": browser}
        if not execute:
            return {"ok": False, "resource_id": "browser.navigation.open", "error": "EXECUTION_DISABLED", "dry_run": True}
        try:
            if selected:
                executable = (selected.get("local") or {}).get("executable") or selected.get("executable")
                executable = shutil.which(executable) if executable else None
                if not executable:
                    raise OSError("BROWSER_NOT_FOUND")
                process = self.runner([executable, url])
                evidence = {"pid": getattr(process, "pid", None)}
                if evidence["pid"] is None:
                    raise OSError("PROCESS_EVIDENCE_MISSING")
            else:
                if not self.opener(url):
                    raise OSError("BROWSER_OPEN_FAILED")
                evidence = {"opened": True}
            return {"ok": True, "resource_id": "browser.navigation.open", "resolved": {"url": url, "browser": browser}, "evidence": evidence, "error": None}
        except OSError as exc:
            return {"ok": False, "resource_id": "browser.navigation.open", "error": str(exc)}

    @staticmethod
    def close_title(title: str, execute: bool = False) -> dict:
        matches = [(handle, text) for handle, text in enum_windows() if title.casefold() in text.casefold()]
        if len(matches) != 1:
            return {"ok": False, "resource_id": "browser.window.close", "error": "WINDOW_NOT_FOUND" if not matches else "WINDOW_AMBIGUOUS"}
        if not execute:
            return {"ok": False, "resource_id": "browser.window.close", "error": "EXECUTION_DISABLED", "dry_run": True}
        handle, text = matches[0]
        try:
            closed = close_tabs(handle, title)
        except RuntimeError as exc:
            return {"ok": False, "resource_id": "browser.window.close", "error": str(exc)}
        if not closed:
            return {"ok": False, "resource_id": "browser.window.close", "error": "TAB_NOT_FOUND"}
        return {"ok": True, "resource_id": "browser.window.close", "evidence": {"handle": handle, "title": text, "closed_tabs": closed}, "error": None}
