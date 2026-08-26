from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Provider:
    provider_id: str
    capabilities: dict[str, Callable]
    available: bool = True
    priority: int = 0
    health: Callable[[str], bool] | None = None

    def healthy(self, resource_id: str) -> bool:
        if not self.available:
            return False
        try:
            return self.health(resource_id) if self.health else True
        except Exception:
            return False


class ProviderRegistry:
    def __init__(self):
        self.providers: list[Provider] = []

    def register(self, provider_id: str, capabilities: dict[str, Callable],
                 available: bool = True, priority: int = 0,
                 health: Callable[[str], bool] | None = None) -> None:
        self.providers.append(Provider(provider_id, capabilities, available, priority, health))
        self.providers.sort(key=lambda provider: provider.priority, reverse=True)

    def resolve_all(self, resource_id: str) -> list[Provider]:
        return [provider for provider in self.providers
                if resource_id in provider.capabilities and provider.healthy(resource_id)]

    def resolve(self, resource_id: str) -> Provider | None:
        candidates = self.resolve_all(resource_id)
        return candidates[0] if candidates else None

    def available(self, resource_id: str) -> bool:
        return bool(self.resolve_all(resource_id))

    def catalog(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for provider in self.providers:
            for resource_id in provider.capabilities:
                result.setdefault(resource_id, []).append({
                    "provider_id": provider.provider_id,
                    "available": provider.healthy(resource_id),
                    "priority": provider.priority,
                })
        return result
