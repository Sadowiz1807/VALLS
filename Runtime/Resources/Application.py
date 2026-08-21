from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .Registry import resolve


class Application:
    def __init__(self, registry: Path, runner: Callable = subprocess.Popen, closer: Callable | None = None):
        self.registry = registry
        self.runner = runner
        self.closer = closer or self._close

    @staticmethod
    def _executable(app: dict[str, Any]) -> str | None:
        return (app.get("local") or {}).get("executable") or app.get("executable")

    @staticmethod
    def _close(executable: str) -> dict[str, Any]:
        completed = subprocess.run(
            ["taskkill", "/IM", Path(executable).name, "/T", "/F"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if completed.returncode:
            raise OSError((completed.stderr or completed.stdout).strip() or "PROCESS_NOT_FOUND")
        return {"returncode": completed.returncode}

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
            process = self.runner([executable])
            pid = getattr(process, "pid", None)
            if pid is None:
                raise OSError("PROCESS_EVIDENCE_MISSING")
            return {"ok": True, "resource_id": "application.control.open", "resolved": app["app_id"], "evidence": {"pid": pid}, "error": None}
        except OSError as exc:
            return {"ok": False, "resource_id": "application.control.open", "resolved": app["app_id"], "error": str(exc)}

    def close(self, application: str, execute: bool = False) -> dict[str, Any]:
        app = resolve(self.registry, application)
        if not app:
            return {"ok": False, "resource_id": "application.control.close", "error": "APPLICATION_UNSUPPORTED"}
        executable = self._executable(app)
        if not executable:
            return {"ok": False, "resource_id": "application.control.close", "error": "LOCAL_CAPABILITY_UNAVAILABLE"}
        if not execute:
            return {"ok": False, "resource_id": "application.control.close", "error": "EXECUTION_DISABLED", "dry_run": True}
        try:
            evidence = self.closer(executable)
            if not evidence:
                raise OSError("PROCESS_EVIDENCE_MISSING")
            return {"ok": True, "resource_id": "application.control.close", "resolved": app["app_id"], "evidence": evidence, "error": None}
        except OSError as exc:
            return {"ok": False, "resource_id": "application.control.close", "resolved": app["app_id"], "error": str(exc)}
