from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable


class SystemProvider:
    NIGHT_LIGHT_ON_ID = "SystemSettings_Display_BlueLight_ManualToggleOn_Button"
    NIGHT_LIGHT_OFF_ID = "SystemSettings_Display_BlueLight_ManualToggleOff_Button"
    POWER_COMMANDS = {
        "SHUTDOWN": ["shutdown.exe", "/s", "/t", "0"],
        "RESTART": ["shutdown.exe", "/r", "/t", "0"],
        "SLEEP": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
    }

    def __init__(self, registry: Path, backend: Callable | None = None, runner: Callable | None = None):
        self.config = json.loads(registry.read_text(encoding="utf-8")) if registry.is_file() else {}
        self.backend = backend or self._windows_backend
        self.runner = runner or self._run

    @staticmethod
    def _run(argv: list[str]) -> dict:
        if argv[:2] == ["rundll32.exe", "powrprof.dll,SetSuspendState"]:
            suspended = SystemProvider._suspend()
            return {"returncode": 0 if suspended else 1, "suspended": suspended,
                    "error": None if suspended else "SUSPEND_FAILED"}
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
            return {"returncode": result.returncode, "error": (result.stderr or result.stdout).strip() or None}
        except subprocess.TimeoutExpired:
            return {"returncode": 1, "error": "POWER_COMMAND_TIMEOUT"}

    @staticmethod
    def _suspend() -> bool:
        import ctypes
        from ctypes import wintypes
        suspend = ctypes.WinDLL("powrprof", use_last_error=True).SetSuspendState
        suspend.argtypes = [wintypes.BOOLEAN, wintypes.BOOLEAN, wintypes.BOOLEAN]
        suspend.restype = wintypes.BOOLEAN
        return bool(suspend(False, True, False))

    def power(self, action: str, execute: bool = False) -> dict:
        action = action.upper()
        config = self.config.get("power", {})
        if not config.get("enabled") or action not in config.get("actions", []):
            return self._failure("system.power", "CAPABILITY_DISABLED")
        if not execute:
            return self._failure("system.power", "EXECUTION_DISABLED", dry_run=True)
        evidence = self.runner(self.POWER_COMMANDS[action])
        ok = evidence.get("returncode") == 0
        return {"ok": ok, "resource_id": "system.power", "action": action, "evidence": evidence, "error": None if ok else evidence.get("error", "POWER_COMMAND_FAILED")}

    def set_percent(self, capability: str, value: int, execute: bool = False) -> dict:
        config = self.config.get(capability, {})
        resource_id = f"system.{capability}.set"
        if not config.get("enabled"):
            return self._failure(resource_id, "CAPABILITY_DISABLED")
        if not isinstance(value, int) or not config.get("min", 0) <= value <= config.get("max", 100):
            return self._failure(resource_id, "VALUE_OUT_OF_RANGE")
        if not execute:
            return self._failure(resource_id, "EXECUTION_DISABLED", dry_run=True)
        try:
            evidence = self.backend(capability, value)
            if evidence.get("observed") != value:
                return self._failure(resource_id, "STATE_EVIDENCE_MISMATCH", evidence=evidence)
            return {"ok": True, "resource_id": resource_id, "evidence": evidence, "error": None}
        except (OSError, RuntimeError) as exc:
            return self._failure(resource_id, str(exc))

    def night_light(self, enabled: bool, execute: bool = False) -> dict:
        config = self.config.get("night_light", {})
        if not config.get("enabled"):
            return self._failure("system.night_light.set", "CAPABILITY_DISABLED")
        if not execute:
            return self._failure("system.night_light.set", "EXECUTION_DISABLED", dry_run=True)
        try:
            evidence = self.backend("night_light", bool(enabled))
            if evidence.get("observed") is not bool(enabled):
                return self._failure("system.night_light.set", "STATE_EVIDENCE_MISMATCH", evidence=evidence)
            return {"ok": True, "resource_id": "system.night_light.set", "evidence": evidence, "error": None}
        except (OSError, RuntimeError) as exc:
            return self._failure("system.night_light.set", str(exc))

    @staticmethod
    def _failure(resource_id: str, error: str, **extra) -> dict:
        return {"ok": False, "resource_id": resource_id, "error": error, **extra}

    @staticmethod
    def _windows_backend(operation: str, value: int) -> dict:
        if operation == "brightness":
            command = (
                "$m=Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods;"
                f"$m|Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=1;Brightness={value}}}|Out-Null;"
                "$b=Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness;"
                "[int]$b.CurrentBrightness"
            )
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=15, check=False)
            if result.returncode:
                raise OSError(result.stderr.strip() or "BRIGHTNESS_FAILED")
            return {"observed": int(result.stdout.strip())}
        if operation == "volume":
            from pycaw.pycaw import AudioUtilities
            endpoint = AudioUtilities.GetSpeakers().EndpointVolume
            endpoint.SetMasterVolumeLevelScalar(value / 100, None)
            return {"observed": round(endpoint.GetMasterVolumeLevelScalar() * 100)}
        if operation == "night_light":
            before = SystemProvider._night_light_state()
            if before is not bool(value):
                SystemProvider._invoke_night_light()
                time.sleep(1)
            return {"observed": SystemProvider._night_light_state()}
        raise RuntimeError("BACKEND_UNAVAILABLE")

    @staticmethod
    def _night_light_control():
        from pywinauto import Desktop
        os.startfile("ms-settings:nightlight")
        deadline = time.time() + 10
        while time.time() < deadline:
            for window in Desktop(backend="uia").windows(class_name="ApplicationFrameWindow"):
                for control in window.descendants(control_type="Button"):
                    if control.element_info.automation_id in (SystemProvider.NIGHT_LIGHT_ON_ID, SystemProvider.NIGHT_LIGHT_OFF_ID):
                        return control
            time.sleep(0.2)
        raise RuntimeError("NIGHT_LIGHT_CONTROL_NOT_FOUND")

    @staticmethod
    def _night_light_state() -> bool:
        return SystemProvider._night_light_control().element_info.automation_id == SystemProvider.NIGHT_LIGHT_OFF_ID

    @staticmethod
    def _invoke_night_light() -> None:
        SystemProvider._night_light_control().invoke()
