from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from Runtime.Providers.Application import ApplicationProvider
from Runtime.Providers.Browser import BrowserProvider
from Runtime.Providers.Media import MediaProvider
from Runtime.Providers.Registry import ProviderRegistry
from Runtime.Providers.System import SystemProvider
from Runtime.Resources.Registry import resolve


def _dependency_available(name: str) -> bool:
    modules = {
        "application.windows": ("win32gui", "win32process", "psutil"),
        "browser.window": ("win32gui", "pywinauto"),
        "system.volume": ("pycaw",),
        "system.night_light": ("pywinauto",),
    }
    if name == "browser.navigation":
        return True
    if name == "system.power":
        return os.name == "nt" and shutil.which("shutdown.exe") is not None
    if name == "system.brightness":
        return os.name == "nt" and shutil.which("powershell.exe") is not None
    return os.name == "nt" and all(importlib.util.find_spec(module) is not None for module in modules.get(name, ()))


def build_builtin_providers(
    registry_dir: Path,
    runner: Callable = subprocess.Popen,
    web_opener: Callable | None = None,
    media_available: Callable[[], bool] = MediaProvider.available,
    dependency_available: Callable[[str], bool] = _dependency_available,
) -> ProviderRegistry:
    application = ApplicationProvider(registry_dir / "applications.json", runner=runner)
    browser = BrowserProvider(
        registry_dir / "browsers.json", runner=runner,
        **({"opener": web_opener} if web_opener else {}),
    )
    media = MediaProvider(registry_dir / "media_providers.json")
    system = SystemProvider(registry_dir / "system.json")

    def playback(command: str):
        return lambda args, execute: media.run(
            [command, args.get("query", "")] if command == "play" else ["pause" if command == "stop" else command],
            args.get("platform", "spotify"), execute,
        )

    implementations = {
        "application.catalog.builtin": ({
            "application.catalog.resolve": lambda args, _execute: _resolve_application(registry_dir, args),
        }, True),
        "application.control.windows": ({
            "application.control.open": lambda args, execute: application.open(args.get("resolved") or args.get("application", ""), execute),
            "application.control.close": lambda args, execute: application.close(args.get("resolved") or args.get("application", ""), execute),
        }, dependency_available("application.windows")),
        "browser.navigation.windows": ({
            "browser.navigation.open": lambda args, execute: browser.open(args.get("url", ""), args.get("browser"), execute),
        }, dependency_available("browser.navigation")),
        "browser.window.windows": ({
            "browser.window.close": lambda args, execute: browser.close_title(args.get("title", ""), execute),
        }, dependency_available("browser.window")),
        "media.spotify": ({resource_id: playback(command) for resource_id, command in {
            "media.playback.play": "play", "media.playback.pause": "pause",
            "media.playback.resume": "resume", "media.playback.stop": "stop",
            "media.playback.next": "next", "media.playback.previous": "previous",
        }.items()}, media_available()),
        "system.power.windows": ({
            "system.power": lambda args, execute: system.power(args.get("action", ""), execute),
        }, dependency_available("system.power")),
        "system.brightness.windows": ({
            "system.brightness.set": lambda args, execute: system.set_percent("brightness", args.get("value"), execute),
        }, dependency_available("system.brightness")),
        "system.volume.windows": ({
            "system.volume.set": lambda args, execute: system.set_percent("volume", args.get("value"), execute),
        }, dependency_available("system.volume")),
        "system.night-light.windows": ({
            "system.night_light.set": lambda args, execute: system.night_light(args.get("enabled"), execute),
        }, dependency_available("system.night_light")),
        "response.builtin": ({
            "response.renderer.social": lambda args, _execute: {"ok": True, "evidence": {"intent": args.get("intent")}},
        }, True),
    }

    manifest = registry_dir / "providers.json"
    declarations = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else []
    providers = ProviderRegistry()
    for declaration in declarations:
        provider_id = declaration.get("provider_id")
        if provider_id not in implementations:
            continue
        implementation, healthy = implementations[provider_id]
        capabilities = {
            resource_id: implementation[resource_id]
            for resource_id in declaration.get("capabilities", []) if resource_id in implementation
        }
        providers.register(
            provider_id, capabilities,
            available=bool(declaration.get("enabled", False) and healthy),
            priority=int(declaration.get("priority", 0)),
        )
    return providers


def _resolve_application(registry_dir: Path, arguments: dict) -> dict:
    item = resolve(registry_dir / "applications.json", arguments.get("application", ""))
    return {"ok": bool(item), "evidence": item, "error": None if item else "APPLICATION_UNSUPPORTED"}
