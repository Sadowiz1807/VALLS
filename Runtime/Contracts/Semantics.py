from __future__ import annotations

from typing import Any


ROUTES: dict[tuple[str, str | None], str | None] = {
    ("APPLICATION_CONTROL", "OPEN"): "application.open",
    ("APPLICATION_CONTROL", "CLOSE"): "application.close",
    ("APPLICATION_CONTROL", "FOCUS"): None,
    ("MEDIA_CONTROL", "PLAY"): "media.play",
    ("MEDIA_CONTROL", "PAUSE"): "media.transport",
    ("MEDIA_CONTROL", "RESUME"): "media.transport",
    ("MEDIA_CONTROL", "STOP"): "media.transport",
    ("MEDIA_CONTROL", "NEXT"): "media.transport",
    ("MEDIA_CONTROL", "PREVIOUS"): "media.transport",
    ("MEDIA_CONTROL", "VOLUME_UP"): None,
    ("MEDIA_CONTROL", "VOLUME_DOWN"): None,
    ("MEDIA_CONTROL", "SET_VOLUME"): None,
    ("WEB_OPEN", None): "web.open",
    ("WEB_SEARCH", None): None,
    ("WEB_NAVIGATE", "BACK"): None,
    ("WEB_NAVIGATE", "FORWARD"): None,
    ("WEB_NAVIGATE", "REFRESH"): None,
    ("WEB_NAVIGATE", "SCROLL_UP"): None,
    ("WEB_NAVIGATE", "SCROLL_DOWN"): None,
    ("WEB_NAVIGATE", "GO_HOME"): None,
    ("TAB_CONTROL", "NEW"): None,
    ("TAB_CONTROL", "CLOSE"): None,
    ("TAB_CONTROL", "SWITCH"): None,
    ("TAB_CONTROL", "REOPEN"): None,
    ("RUN_COMMAND", None): None,
    ("TASK_STATUS", None): None,
    ("SOCIAL_RESPONSE", None): "conversation.social",
}


def validate_semantic_parity(
    ontology: dict[str, Any], skills: list[dict[str, Any]],
    routes: dict[tuple[str, str | None], str | None] = ROUTES,
) -> list[str]:
    errors: list[str] = []
    project = ontology.get("project", {})
    if project.get("contract_version") != "2.0":
        errors.append("ONTOLOGY_CONTRACT_VERSION_MISMATCH")
    if project.get("release_version") != "0.0.4":
        errors.append("ONTOLOGY_RELEASE_VERSION_MISMATCH")
    spec = ontology.get("ontology", {})
    goals = set(spec.get("goals", []))
    parameters = spec.get("goal_parameters", {})
    skill_map = {skill["skill_id"]: skill for skill in skills}

    expected_routes: set[tuple[str, str | None]] = set()
    for goal in goals:
        actions = parameters.get(goal, {}).get("properties", {}).get("action", {}).get("enum", [])
        expected_routes.update((goal, action) for action in actions or [None])
    missing = expected_routes - set(routes)
    extra = set(routes) - expected_routes
    errors.extend(f"MODEL_SEMANTIC_UNDECLARED:{goal}:{action}" for goal, action in sorted(missing, key=str))
    errors.extend(f"ROUTE_NOT_IN_MODEL:{goal}:{action}" for goal, action in sorted(extra, key=str))

    for skill in skills:
        accepts = skill.get("accepts", {})
        goal = accepts.get("goal")
        if goal not in goals and skill.get("semantically_reachable", True):
            errors.append(f"RUNTIME_ONLY_GOAL:{goal}:{skill['skill_id']}")
            continue
        accepted = accepts.get("action")
        actions = [accepted] if isinstance(accepted, str) else accepted or []
        model_actions = set(parameters.get(goal, {}).get("properties", {}).get("action", {}).get("enum", []))
        for action in actions:
            if model_actions and action not in model_actions:
                errors.append(f"RUNTIME_ONLY_ACTION:{goal}:{action}:{skill['skill_id']}")

    for (goal, action), skill_id in routes.items():
        if goal not in goals:
            errors.append(f"ROUTE_GOAL_NOT_IN_ONTOLOGY:{goal}")
            continue
        model_actions = set(parameters.get(goal, {}).get("properties", {}).get("action", {}).get("enum", []))
        if action is not None and model_actions and action not in model_actions:
            errors.append(f"ROUTE_ACTION_NOT_IN_ONTOLOGY:{goal}:{action}")
        if skill_id is None:
            continue
        skill = skill_map.get(skill_id)
        if not skill:
            errors.append(f"ROUTE_SKILL_NOT_FOUND:{goal}:{action}:{skill_id}")
            continue
        accepts = skill.get("accepts", {})
        accepted = accepts.get("action")
        accepted = [accepted] if isinstance(accepted, str) else accepted or []
        if accepts.get("goal") != goal or action is not None and action not in accepted:
            errors.append(f"ROUTE_SKILL_SEMANTIC_MISMATCH:{goal}:{action}:{skill_id}")
        if not skill.get("enabled", True) or not skill.get("semantically_reachable", True):
            errors.append(f"ROUTE_SKILL_UNREACHABLE:{goal}:{action}:{skill_id}")

    return errors
