from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Runtime.Providers.Registry import ProviderRegistry


class ResourceDispatcher:
    def __init__(self, providers: ProviderRegistry, manifest: Path | None = None):
        self.providers = providers
        self.manifest_required = manifest is not None
        self.resources = {
            item["resource_id"]: item
            for item in (json.loads(manifest.read_text(encoding="utf-8")) if manifest and manifest.is_file() else [])
        }

    def dispatch(self, resource_id: str, arguments: dict[str, Any], execute: bool = False) -> dict:
        if self.manifest_required and not self.resources.get(resource_id, {}).get("enabled", False):
            return {"ok": False, "resource_id": resource_id, "error": "RESOURCE_DISABLED"}
        provider = self.providers.resolve(resource_id)
        if not provider:
            return {"ok": False, "resource_id": resource_id, "error": "PROVIDER_UNAVAILABLE"}
        try:
            result = provider.capabilities[resource_id](arguments, execute)
        except (OSError, RuntimeError) as exc:
            result = {"ok": False, "error": str(exc)}
        return {**result, "resource_id": resource_id, "provider_id": provider.provider_id}
