from __future__ import annotations

import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .Registry import resolve


class Browser:
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
