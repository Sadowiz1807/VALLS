from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Runtime.Contracts.Semantics import ROUTES, validate_semantic_parity

BUILTIN_PROVIDER_IDS = frozenset({
    "builtin.web-search",
    "application.catalog.builtin", "application.control.windows",
    "browser.navigation.windows", "media.spotify", "media.spotify-native",
    "system.power.windows", "system.brightness.windows", "system.volume.windows",
    "system.night-light.windows", "response.builtin",
})


class RegistryValidationError(ValueError):
    pass


def load_manifest(path: Path, canonical: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RegistryValidationError(f"MANIFEST_MISSING:{path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if canonical:
            raise RegistryValidationError(f"MANIFEST_ENVELOPE_REQUIRED:{path.name}")
        return data
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise RegistryValidationError(f"MANIFEST_INVALID:{path.name}")
    if data.get("schema_version") != "1.0":
        raise RegistryValidationError(f"SCHEMA_VERSION_MISMATCH:{path.name}")
    if data.get("ontology_contract_version") != "2.0" or data.get("ontology_release_version") != "0.0.4":
        raise RegistryValidationError(f"ONTOLOGY_VERSION_MISMATCH:{path.name}")
    return data["items"]


def validate_registry(registry_dir: Path, ontology_path: Path) -> dict[str, Any]:
    skills = load_manifest(registry_dir / "skills.json", canonical=True)
    resources = load_manifest(registry_dir / "resources.json", canonical=True)
    providers = load_manifest(registry_dir / "providers.json", canonical=True)
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))

    def unique(items: list[dict[str, Any]], key: str) -> set[str]:
        values = [item.get(key) for item in items]
        if None in values or len(values) != len(set(values)):
            raise RegistryValidationError(f"DUPLICATE_OR_MISSING_ID:{key}")
        return set(values)

    skill_ids = unique(skills, "skill_id")
    resource_ids = unique(resources, "resource_id")
    provider_ids = unique(providers, "provider_id")
    unknown_providers = provider_ids - BUILTIN_PROVIDER_IDS
    if unknown_providers:
        raise RegistryValidationError(f"PROVIDER_IMPLEMENTATION_UNKNOWN:{sorted(unknown_providers)}")
    del skill_ids

    missing_resources = {
        resource_id for skill in skills for resource_id in skill.get("resources", [])
        if resource_id not in resource_ids
    }
    if missing_resources:
        raise RegistryValidationError(f"UNKNOWN_SKILL_RESOURCES:{sorted(missing_resources)}")
    declared_capabilities = {
        capability for provider in providers if provider.get("enabled", True)
        for capability in provider.get("capabilities", [])
    }
    enabled_resources = {resource["resource_id"] for resource in resources if resource.get("enabled", True)}
    uncovered = enabled_resources - declared_capabilities
    if uncovered:
        raise RegistryValidationError(f"RESOURCE_WITHOUT_PROVIDER:{sorted(uncovered)}")

    resource_map = {item["resource_id"]: item for item in resources}
    levels = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    for skill in skills:
        declared = set(skill.get("resources", []))
        if (skill.get("enabled", True) and declared
                and all(not resource_map[item].get("enabled", True) for item in declared)):
            raise RegistryValidationError(f"ENABLED_SKILL_USES_DISABLED_RESOURCE:{skill['skill_id']}")
        mapping = skill.get("resource_by_action", {})
        if not set(mapping.values()) <= declared:
            raise RegistryValidationError(f"ACTION_RESOURCE_NOT_DECLARED:{skill['skill_id']}")
        accepted = skill.get("accepts", {}).get("action", [])
        accepted = [accepted] if isinstance(accepted, str) else accepted
        if mapping and set(mapping) != set(accepted):
            raise RegistryValidationError(f"ACTION_MAPPING_INCOMPLETE:{skill['skill_id']}")

        seen: set[str] = set()
        for step in skill.get("steps", []):
            if not isinstance(step, dict) or not step.get("id") or step.get("use") not in declared:
                raise RegistryValidationError(f"WORKFLOW_STEP_INVALID:{skill['skill_id']}")
            if step["id"] in seen:
                raise RegistryValidationError(f"WORKFLOW_STEP_DUPLICATE:{skill['skill_id']}:{step['id']}")
            for value in step.get("with", {}).values():
                if isinstance(value, str) and value.startswith("$steps."):
                    source = value.split(".", 2)[1]
                    if source not in seen:
                        raise RegistryValidationError(f"WORKFLOW_FORWARD_REFERENCE:{skill['skill_id']}:{source}")
            seen.add(step["id"])

        skill_risk = skill.get("risk")
        skill_risk_by_action = skill.get("risk_by_action", {})
        for resource_id in declared:
            resource = resource_map[resource_id]
            resource_risk = resource.get("risk")
            if resource_risk in levels and skill_risk in levels and levels[skill_risk] < levels[resource_risk]:
                raise RegistryValidationError(f"RISK_DOWNGRADE:{skill['skill_id']}:{resource_id}")
            for action, risk in resource.get("risk_by_action", {}).items():
                if levels.get(skill_risk_by_action.get(action, skill_risk), -1) < levels[risk]:
                    raise RegistryValidationError(f"RISK_DOWNGRADE:{skill['skill_id']}:{resource_id}:{action}")
                if risk == "HIGH" and not skill.get("confirmation_required_by_action", {}).get(
                    action, skill.get("confirmation_required", False),
                ):
                    raise RegistryValidationError(f"HIGH_RISK_WITHOUT_CONFIRMATION:{skill['skill_id']}:{action}")

    parity_errors = validate_semantic_parity(ontology, skills, ROUTES)
    if parity_errors:
        raise RegistryValidationError("SEMANTIC_PARITY:" + "|".join(parity_errors))

    return {
        "skills": len(skills), "resources": len(resources), "providers": len(providers),
        "ontology_release": ontology["project"]["release_version"],
    }
