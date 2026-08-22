from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Provider:
    provider_id: str
    capabilities: dict[str, Callable]
    available: bool = True
    priority: int = 0


class ProviderRegistry:
    def __init__(self):
        self.providers: list[Provider] = []

    def register(self, provider_id: str, capabilities: dict[str, Callable],
                 available: bool = True, priority: int = 0) -> None:
        self.providers.append(Provider(provider_id, capabilities, available, priority))
        self.providers.sort(key=lambda provider: provider.priority, reverse=True)

    def resolve(self, resource_id: str) -> Provider | None:
        return next((provider for provider in self.providers
                     if provider.available and resource_id in provider.capabilities), None)

    def available(self, resource_id: str) -> bool:
        return self.resolve(resource_id) is not None

    def catalog(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for provider in self.providers:
            for resource_id in provider.capabilities:
                result.setdefault(resource_id, []).append({
                    "provider_id": provider.provider_id,
                    "available": provider.available,
                    "priority": provider.priority,
                })
        return result
