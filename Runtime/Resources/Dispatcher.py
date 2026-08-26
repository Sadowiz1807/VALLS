from __future__ import annotations

from pathlib import Path
from typing import Any

from Runtime.Contracts.Validate import load_manifest
from Runtime.Policy.Confirmation import ConfirmationStore
from Runtime.Providers.Registry import ProviderRegistry


class ResourceDispatcher:
    def __init__(self, providers: ProviderRegistry, manifest: Path | None = None,
                 confirmation_store: ConfirmationStore | None = None):
        self.providers = providers
        self.manifest_required = manifest is not None
        self.resources = {
            item["resource_id"]: item
            for item in (load_manifest(manifest) if manifest and manifest.is_file() else [])
        }
        self.confirmation_store = confirmation_store

    @staticmethod
    def _risk(resource: dict | None, arguments: dict[str, Any]) -> str:
        if not resource:
            return "NONE"
        action = str(arguments.get("action", "")).upper()
        return resource.get("risk_by_action", {}).get(action, resource.get("risk", "NONE"))

    @staticmethod
    def _valid(value: Any, contract: dict) -> bool:
        expected = {"string": str, "integer": int, "boolean": bool, "object": dict}.get(contract.get("type"))
        return expected is None or isinstance(value, expected) and not (expected is int and isinstance(value, bool))

    def _validate(self, values: dict[str, Any], contracts: dict[str, dict], success: bool = False) -> bool:
        for name, contract in contracts.items():
            required = contract.get("required") or success and contract.get("required_on_success")
            value = values.get(name)
            if required and value is None:
                return False
            if value is not None and not self._valid(value, contract):
                return False
        return True

    def dispatch(self, resource_id: str, arguments: dict[str, Any], execute: bool = False,
                 *, skill_id: str | None = None, confirmation_grant: Any = None,
                 confirmation_arguments: dict[str, Any] | None = None,
                 confirmation_required: bool = False,
                 request_id: str | None = None) -> dict:
        if confirmation_grant is not None:
            request_id = getattr(confirmation_grant, "request_id", request_id)
        resource = self.resources.get(resource_id)
        if self.manifest_required and (not resource or not resource.get("enabled", True)):
            return {"ok": False, "resource_id": resource_id, "error": "RESOURCE_DISABLED"}
        if resource and not self._validate(arguments, resource.get("arguments", {})):
            return {"ok": False, "resource_id": resource_id, "error": "RESOURCE_CONTRACT_VIOLATION"}
        risk = self._risk(resource, confirmation_arguments or arguments)
        needs_grant = execute and (risk == "HIGH" or confirmation_required)
        if needs_grant:
            if confirmation_grant is None or not self.confirmation_store:
                error = "CONFIRMATION_REQUIRED"
            else:
                error = self.confirmation_store.validate(
                    confirmation_grant, request_id or "", skill_id or "", resource_id,
                    confirmation_arguments or arguments, risk,
                )
            if error:
                result = {"ok": False, "resource_id": resource_id, "error": error}
                if error == "CONFIRMATION_REQUIRED":
                    result.update(request_id=request_id, effective_risk=risk)
                return result

        providers = self.providers.resolve_all(resource_id)
        if not providers:
            return {"ok": False, "resource_id": resource_id, "error": "PROVIDER_UNAVAILABLE"}
        if needs_grant:
            error = self.confirmation_store.validate(
                confirmation_grant, request_id or "", skill_id or "", resource_id,
                confirmation_arguments or arguments, risk, consume=True,
            )
            if error:
                return {"ok": False, "resource_id": resource_id, "error": error}

        from Runtime.Contracts.Errors import ERRORS
        attempts = []
        for provider in providers:
            try:
                result = provider.capabilities[resource_id](arguments, execute)
            except (OSError, RuntimeError) as exc:
                result = {"ok": False, "error": "EXECUTION_FAILED", "message": str(exc),
                          "side_effect_state": "UNKNOWN"}
            if not isinstance(result, dict):
                result = {"ok": False, "error": "RESOURCE_CONTRACT_VIOLATION"}
            if resource and not self._validate(result, resource.get("result", {}), success=bool(result.get("ok"))):
                result = {"ok": False, "error": "RESOURCE_CONTRACT_VIOLATION"}
            if result.get("ok"):
                envelope = {**result, "provider_id": provider.provider_id, "resource_id": resource_id}
                if attempts:
                    envelope["attempts"] = attempts
                if request_id:
                    envelope["request_id"] = request_id
                return envelope
            state = result.get("side_effect_state", "UNKNOWN")
            error = result.get("error", "EXECUTION_FAILED")
            attempt = {"provider_id": provider.provider_id, "error": error, "side_effect_state": state}
            attempts.append(attempt)
            spec = ERRORS.get(error)
            if not spec or not spec.fallbackable or state != "NOT_STARTED":
                return {**result, "provider_id": provider.provider_id, "resource_id": resource_id,
                        "attempts": attempts, **({"request_id": request_id} if request_id else {})}
        return {"ok": False, "resource_id": resource_id, "error": "PROVIDER_UNAVAILABLE",
                "attempts": attempts, **({"request_id": request_id} if request_id else {})}
