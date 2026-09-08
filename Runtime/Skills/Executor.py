from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

from Runtime.Contracts.Validate import load_manifest
from Runtime.Policy.Confirmation import ConfirmationStore
from Runtime.Resources.Dispatcher import ResourceDispatcher


class SkillExecutor:
    def __init__(self, manifest: Path, dispatcher: ResourceDispatcher,
                 confirmation_store: ConfirmationStore | None = None):
        self.skills = {item["skill_id"]: item for item in load_manifest(manifest)}
        self.dispatcher = dispatcher
        self.confirmation_store = confirmation_store

    @staticmethod
    def _binding(value: Any, inputs: dict[str, Any], steps: dict[str, dict]) -> Any:
        if not isinstance(value, str) or not value.startswith("$"):
            return value
        parts = value[1:].split(".")
        if len(parts) == 2 and parts[0] == "inputs":
            source: Any = inputs
            path = parts[1:]
        elif len(parts) >= 4 and parts[0] == "steps" and parts[2] == "result":
            if parts[1] not in steps:
                raise KeyError(value)
            source = steps[parts[1]]["result"]
            path = parts[3:]
        else:
            raise KeyError(value)
        for key in path:
            if not isinstance(source, dict) or key not in source:
                raise KeyError(value)
            source = source[key]
        return source

    @staticmethod
    def _validate_inputs(skill_id: str, skill: dict, arguments: dict[str, Any]) -> dict | None:
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
        return None

    def execute(self, skill_id: str, arguments: dict[str, Any], execute: bool = False,
                confirmation_grant: Any = None, request_id: str | None = None) -> dict:
        skill = self.skills.get(skill_id)
        if not skill:
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_NOT_FOUND"}
        if not skill.get("enabled", True):
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_DISABLED"}
        if not skill.get("resources"):
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_NOT_IMPLEMENTED"}

        arguments = dict(arguments)
        request_id = confirmation_grant.request_id if confirmation_grant else request_id or uuid.uuid4().hex
        if isinstance(arguments.get("action"), str):
            arguments["action"] = arguments["action"].upper()
        invalid = self._validate_inputs(skill_id, skill, arguments)
        if invalid:
            return invalid
        action = arguments.get("action")
        accepted = skill.get("accepts", {}).get("action")
        accepted = [accepted] if isinstance(accepted, str) else accepted
        if action is not None and accepted and action not in accepted:
            return {"ok": False, "skill_id": skill_id, "error": "ACTION_UNSUPPORTED"}
        platform = str(arguments.get("platform", "DEFAULT")).upper()
        unavailable = skill.get("unavailable_actions_by_platform", {}).get(platform, [])
        if action in unavailable:
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_NOT_AVAILABLE"}

        confirmation_required = skill.get("confirmation_required", False)
        confirmation_required = skill.get("confirmation_required_by_action", {}).get(action, confirmation_required)
        if execute and confirmation_required and not confirmation_grant:
            return {"ok": False, "skill_id": skill_id, "error": "CONFIRMATION_REQUIRED"}

        workflow = skill.get("steps")
        if workflow:
            step_results: dict[str, dict] = {}
            result: dict = {}
            for step in workflow:
                if not isinstance(step, dict) or not step.get("id") or not step.get("use"):
                    return {"ok": False, "skill_id": skill_id, "error": "RESOURCE_CONTRACT_VIOLATION"}
                if step["id"] in step_results or step["use"] not in skill["resources"]:
                    return {"ok": False, "skill_id": skill_id, "error": "RESOURCE_CONTRACT_VIOLATION"}
                try:
                    bound = {key: self._binding(value, arguments, step_results)
                             for key, value in step.get("with", {}).items()}
                except KeyError:
                    return {"ok": False, "skill_id": skill_id, "error": "RESOURCE_CONTRACT_VIOLATION"}
                result = self.dispatcher.dispatch(
                    step["use"], bound, execute, skill_id=skill_id,
                    confirmation_grant=confirmation_grant, confirmation_arguments=arguments,
                    confirmation_required=bool(confirmation_required and step is workflow[-1]),
                    request_id=request_id,
                )
                if not result.get("ok"):
                    return {"skill_id": skill_id, **result}
                step_results[step["id"]] = {"result": result}
            return {"skill_id": skill_id, **result}

        mapping = skill.get("resource_by_action")
        if mapping is not None:
            resource_id = mapping.get(action)
            if not resource_id:
                return {"ok": False, "skill_id": skill_id, "error": "ACTION_UNSUPPORTED"}
        elif len(skill["resources"]) == 1:
            resource_id = skill["resources"][0]
        else:
            return {"ok": False, "skill_id": skill_id, "error": "RESOURCE_NOT_RESOLVED"}
        return {"skill_id": skill_id, **self.dispatcher.dispatch(
            resource_id, arguments, execute, skill_id=skill_id,
            confirmation_grant=confirmation_grant, confirmation_arguments=arguments,
            confirmation_required=bool(confirmation_required),
            request_id=request_id,
        )}
