from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from Runtime.Resources.Registry import resolve


class MediaProvider:

    @staticmethod
    def available() -> bool:
        return MediaProvider._script().is_file()

    @staticmethod
    def _script() -> Path:
        return Path.home() / "AppData/Local/hermes/skills/media/spotify-local-control/scripts/spotify_control.py"
    def __init__(self, registry: Path | None = None, command: Callable[[list[str]], dict] | None = None,
                 observer: Callable[[], dict] | None = None, wait: Callable[[float], None] = time.sleep):
        self.registry = registry
        self.command = command or self._spotify
        self.observer = observer if observer is not None else (lambda: self._spotify(["current"])) if command is None else None
        self.wait = wait

    @staticmethod
    def _spotify(args: list[str]) -> dict:
        script = MediaProvider._script()
        if not script.is_file():
            return {"ok": False, "error": "PROVIDER_UNAVAILABLE"}
        completed = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=45, check=False)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = {"ok": False, "error": "PROVIDER_INVALID_RESULT", "message": (completed.stderr or completed.stdout).strip()}
        if completed.returncode and result.get("ok"):
            result = {"ok": False, "error": "PROVIDER_FAILED"}
        return result

    def run(self, args: list[str], platform: str = "spotify", execute: bool = False) -> dict:
        provider = resolve(self.registry, platform) if self.registry else ({"provider_id": "spotify"} if platform.lower() in {"spotify", "default"} else None)
        if not provider:
            return {"ok": False, "resource_id": "media.playback", "error": "PROVIDER_UNSUPPORTED", "ask_to_add_whitelist": platform}
        if not execute:
            return {"ok": False, "resource_id": "media.playback", "error": "EXECUTION_DISABLED", "dry_run": True}
        if args[0] == "play" and provider.get("device"):
            args = [*args, "--device", provider["device"]]
        result = self.command(args)
        if result.get("ok") and args[0] in {"pause", "resume"} and self.observer:
            expected = args[0] == "resume"
            observed = None
            for attempt in range(3):
                observed = self.observer()
                if observed.get("ok") and observed.get("playing") is expected:
                    result = {**result, "observed": observed}
                    break
                if attempt < 2:
                    self.wait(1)
            else:
                result = {**result, "ok": False, "error": "POSTCONDITION_NOT_OBSERVED", "observed": observed}
        return {"resource_id": f"media.playback.{args[0]}", "provider": provider["provider_id"], **result}
