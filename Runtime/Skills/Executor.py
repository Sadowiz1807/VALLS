from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Runtime.Resources.Dispatcher import ResourceDispatcher


class SkillExecutor:
    def __init__(self, manifest: Path, dispatcher: ResourceDispatcher):
        self.skills = {
            item["skill_id"]: item
            for item in (json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else [])
        }
        self.dispatcher = dispatcher

    @staticmethod
    def _resource(skill: dict, arguments: dict[str, Any]) -> str | None:
        resources = skill.get("resources", [])
        if len(resources) == 1:
            return resources[0]
        action = str(arguments.get("action", "")).lower()
        return next((resource for resource in resources if resource.rsplit(".", 1)[-1] == action), resources[-1] if resources else None)

    def execute(self, skill_id: str, arguments: dict[str, Any], execute: bool = False,
                confirmed: bool = False) -> dict:
        skill = self.skills.get(skill_id)
        if not skill:
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_NOT_FOUND"}
        if not skill.get("enabled", True):
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_DISABLED"}
        if not skill.get("resources"):
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_NOT_IMPLEMENTED"}

        arguments = dict(arguments)
        if isinstance(arguments.get("action"), str):
            arguments["action"] = arguments["action"].upper()
        action = arguments.get("action")
        confirmation_required = skill.get("confirmation_required", False)
        confirmation_required = skill.get("confirmation_required_by_action", {}).get(
            action, confirmation_required,
        )
        if execute and confirmation_required and not confirmed:
            return {"ok": False, "skill_id": skill_id, "error": "CONFIRMATION_REQUIRED"}

        for name, contract in skill.get("inputs", {}).items():
            value = arguments.get(name)
            if contract.get("required") and (value is None or isinstance(value, str) and not value.strip()):
                return {"ok": False, "skill_id": skill_id, "error": "INPUT_REQUIRED", "input": name}
            expected = {"string": str, "integer": int, "boolean": bool}.get(contract.get("type"))
            if value is not None and expected and (not isinstance(value, expected) or expected is int and isinstance(value, bool)):
                return {"ok": False, "skill_id": skill_id, "error": "INPUT_TYPE_INVALID", "input": name}
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < contract.get("minimum", value) or value > contract.get("maximum", value):
                    return {"ok": False, "skill_id": skill_id, "error": "INPUT_OUT_OF_RANGE", "input": name}
        accepted = skill.get("accepts", {}).get("action")
        accepted = [accepted] if isinstance(accepted, str) else accepted
        if action is not None and accepted and action not in accepted:
            return {"ok": False, "skill_id": skill_id, "error": "ACTION_UNSUPPORTED"}

        steps = skill.get("steps")
        if steps:
            current = dict(arguments)
            result = None
            for resource_id in steps:
                if not self.dispatcher.providers.available(resource_id):
                    return {"ok": False, "skill_id": skill_id, "error": "PROVIDER_UNAVAILABLE", "resources": [resource_id]}
                result = self.dispatcher.dispatch(resource_id, current, execute)
                if not result.get("ok"):
                    return {"skill_id": skill_id, **result}
                current["resolved"] = result.get("evidence")
            return {"skill_id": skill_id, **result}

        resource_id = self._resource(skill, arguments)
        if not resource_id:
            return {"ok": False, "skill_id": skill_id, "error": "RESOURCE_NOT_RESOLVED"}
        if not self.dispatcher.providers.available(resource_id):
            return {"ok": False, "skill_id": skill_id, "error": "PROVIDER_UNAVAILABLE", "resources": [resource_id]}
        return {"skill_id": skill_id, **self.dispatcher.dispatch(resource_id, arguments, execute)}
