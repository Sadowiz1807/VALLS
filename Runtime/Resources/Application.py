from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .Registry import resolve


def observe_windows() -> list[dict[str, Any]]:
    import win32gui
    import win32process
    windows = []

    def add(handle, _):
        if win32gui.IsWindowVisible(handle) and win32gui.GetWindowText(handle):
            _, pid = win32process.GetWindowThreadProcessId(handle)
            windows.append({"handle": handle, "pid": pid, "title": win32gui.GetWindowText(handle)})

    win32gui.EnumWindows(add, None)
    return windows


def close_owned_window(handle: int) -> dict[str, Any]:
    import win32con
    import win32gui
    if not win32gui.IsWindow(handle):
        raise OSError("WINDOW_NOT_FOUND")
    win32gui.PostMessage(handle, win32con.WM_CLOSE, 0, 0)
    deadline = time.time() + 5
    while time.time() < deadline:
        if not win32gui.IsWindow(handle):
            return {"handle": handle, "closed": True}
        time.sleep(0.1)
    raise OSError("WINDOW_CLOSE_FAILED")


class Application:
    def __init__(self, registry: Path, runner: Callable = subprocess.Popen,
                 closer: Callable | None = None, observer: Callable | None = None):
        self.registry = registry
        self.runner = runner
        self.closer = closer or close_owned_window
        self.observer = observer or observe_windows
        self.owned: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _executable(app: dict[str, Any]) -> str | None:
        return (app.get("local") or {}).get("executable") or app.get("executable")

    def open(self, application: str, execute: bool = False) -> dict[str, Any]:
        app = resolve(self.registry, application)
        if not app:
            return {"ok": False, "resource_id": "application.control.open", "error": "APPLICATION_UNSUPPORTED"}
        executable = self._executable(app)
        if not executable:
            return {"ok": False, "resource_id": "application.control.open", "error": "LOCAL_CAPABILITY_UNAVAILABLE"}
        if not execute:
            return {"ok": False, "resource_id": "application.control.open", "error": "EXECUTION_DISABLED", "dry_run": True}
        try:
            before = {window["handle"] for window in self.observer()}
            self.runner([executable])
            deadline = time.time() + 5
            while time.time() < deadline:
                created = [window for window in self.observer() if window["handle"] not in before]
                if created:
                    evidence = created[0]
                    self.owned[app["app_id"]] = evidence
                    return {"ok": True, "resource_id": "application.control.open", "resolved": app["app_id"], "evidence": evidence, "error": None}
                time.sleep(0.1)
            raise OSError("WINDOW_EVIDENCE_MISSING")
        except OSError as exc:
            return {"ok": False, "resource_id": "application.control.open", "resolved": app["app_id"], "error": str(exc)}

    def close(self, application: str, execute: bool = False) -> dict[str, Any]:
        app = resolve(self.registry, application)
        if not app:
            return {"ok": False, "resource_id": "application.control.close", "error": "APPLICATION_UNSUPPORTED"}
        if not execute:
            return {"ok": False, "resource_id": "application.control.close", "error": "EXECUTION_DISABLED", "dry_run": True}
        owned = self.owned.get(app["app_id"])
        if not owned:
            return {"ok": False, "resource_id": "application.control.close", "resolved": app["app_id"], "error": "NO_OWNED_WINDOW"}
        try:
            evidence = self.closer(owned["handle"])
            self.owned.pop(app["app_id"], None)
            return {"ok": True, "resource_id": "application.control.close", "resolved": app["app_id"], "evidence": evidence, "error": None}
        except OSError as exc:
            return {"ok": False, "resource_id": "application.control.close", "resolved": app["app_id"], "error": str(exc)}
