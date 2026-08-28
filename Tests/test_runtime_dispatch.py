import importlib
import json
import shutil
from pathlib import Path

import pytest

from Runtime.Contracts.Errors import error_spec
from Runtime.Contracts.Validate import RegistryValidationError, validate_registry
from Runtime.Providers.Web import build_search_url
from Runtime.engine import AgentHarness


def test_new_input_and_target_errors_are_nonfallbackable_before_side_effect():
    for code, category in (("INPUT_VALUE_INVALID", "INPUT"), ("TARGET_AMBIGUOUS", "TARGET")):
        spec = error_spec(code)
        assert spec.category == category
        assert spec.fallbackable is False
        assert spec.default_side_effect_state == "NOT_STARTED"


def test_legacy_runtime_authorities_stay_absent():
    root = Path(__file__).parents[1] / "Runtime"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    assert 'params.get("route")' not in source
    assert "confirmed=True" not in source
    assert "system.command" not in source


def test_web_search_builds_encoded_allowlisted_url_and_routes_workflow():
    assert build_search_url({"query": "học python", "engine": "DUCKDUCKGO"}) == {
        "ok": True, "url": "https://duckduckgo.com/?q=h%E1%BB%8Dc+python", "engine": "DUCKDUCKGO",
    }
    assert build_search_url({"query": "python", "engine": "YAHOO"})["error"] == "INPUT_VALUE_INVALID"
    root = Path(__file__).parents[1] / "Runtime"
    assert validate_registry(root / "Registry", root / "Model/VSAD/0.0.4/config.json") == {
        "skills": 12, "resources": 26, "providers": 11, "ontology_release": "0.0.4",
    }


def test_spotify_stop_is_blocked_before_provider_attempts():
    root = Path(__file__).parents[1] / "Runtime" / "Registry"
    result = AgentHarness(root, execute=True)._dispatch_turn("dừng Spotify", {
        "act": "EXECUTE", "goal": "MEDIA_CONTROL",
        "parameters": {"action": "STOP", "platform": "SPOTIFY"},
    })

    assert result["status"] == "ERROR"
    assert result["result"] == {
        "ok": False, "skill_id": "media.transport", "error": "SKILL_NOT_AVAILABLE",
    }


def test_title_only_browser_close_skill_and_resource_are_disabled():
    root = Path(__file__).parents[1] / "Runtime" / "Registry"
    skills = json.loads((root / "skills.json").read_text(encoding="utf-8"))["items"]
    resources = json.loads((root / "resources.json").read_text(encoding="utf-8"))["items"]

    assert next(x for x in skills if x["skill_id"] == "web.close")["enabled"] is False
    assert next(x for x in resources if x["resource_id"] == "browser.window.close")["enabled"] is False


def test_media_play_contract_requires_observed_readback_on_success():
    root = Path(__file__).parents[1] / "Runtime" / "Registry"
    resources = json.loads((root / "resources.json").read_text(encoding="utf-8"))["items"]
    play = next(x for x in resources if x["resource_id"] == "media.playback.play")

    assert play["result"]["observed"] == {"type": "object", "required_on_success": True}


def test_validator_rejects_enabled_skill_using_disabled_resource(tmp_path):
    root = Path(__file__).parents[1] / "Runtime"
    registry = tmp_path / "Registry"
    shutil.copytree(root / "Registry", registry)
    skills_path = registry / "skills.json"
    skills = json.loads(skills_path.read_text(encoding="utf-8"))
    next(x for x in skills["items"] if x["skill_id"] == "web.close")["enabled"] = True
    skills_path.write_text(json.dumps(skills), encoding="utf-8")

    with pytest.raises(RegistryValidationError, match="ENABLED_SKILL_USES_DISABLED_RESOURCE:web.close"):
        validate_registry(registry, root / "Model/VSAD/0.0.4/config.json")


def test_browser_close_window_matches_title_without_killing_process(monkeypatch):
    module = importlib.import_module("Runtime.Providers.Browser")
    closed = []
    monkeypatch.setattr(module, "enum_windows", lambda: [(10, "YouTube - Coc Coc"), (11, "Facebook - Chrome")])
    monkeypatch.setattr(module, "close_tabs", lambda handle, title: closed.append((handle, title)) or 2)

    result = module.BrowserProvider.close_title("youtube", execute=True)

    assert result["ok"] is True
    assert result["evidence"]["closed_tabs"] == 2
    assert closed == [(10, "youtube")]


def registry(tmp_path: Path) -> Path:
    (tmp_path / "applications.json").write_text(json.dumps([{
        "app_id": "spotify", "name": "Spotify", "aliases": ["spotify"], "enabled": True,
        "local": {"executable": "spotify.exe"}, "web": {"url": "https://open.spotify.com"},
    }]), encoding="utf-8")
    (tmp_path / "browsers.json").write_text("[]", encoding="utf-8")
    (tmp_path / "skills.json").write_text("[]", encoding="utf-8")
    return tmp_path


class Executor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"ok": True}

    def execute(self, skill_id, args, execute, **kwargs):
        self.calls.append((skill_id, args, execute, kwargs))
        return {"skill_id": skill_id, **self.result}


def test_harness_normalizes_action_and_routes_semantic_to_skill(tmp_path):
    executor = Executor()
    harness = AgentHarness(registry(tmp_path), skill_executor=executor)

    opened = harness._dispatch_turn("mo spotify", {
        "act": "EXECUTE", "goal": "APPLICATION_CONTROL",
        "parameters": {"action": "open", "application": "spotify"},
    })
    played = harness._dispatch_turn("phat nhac", {
        "act": "EXECUTE", "goal": "MEDIA_CONTROL",
        "parameters": {"action": "play", "query": "One More Time"},
    })

    assert opened["skill_id"] == "application.open"
    assert played["skill_id"] == "media.play"
    assert executor.calls[0][1] == {"application": "spotify"}


def test_route_parameter_cannot_change_application_semantic(tmp_path):
    executor = Executor()
    harness = AgentHarness(registry(tmp_path), skill_executor=executor)

    result = harness._dispatch_turn("mo spotify tren web", {
        "act": "EXECUTE", "goal": "APPLICATION_CONTROL",
        "parameters": {"action": "OPEN", "application": "spotify", "route": "WEB"},
    })

    assert result["skill_id"] == "application.open"
    assert executor.calls[0][0] == "application.open"


def test_web_open_routes_directly_without_harness_capability_check(tmp_path):
    executor = Executor({"ok": False, "error": "PROVIDER_UNAVAILABLE"})
    harness = AgentHarness(registry(tmp_path), skill_executor=executor)

    result = harness._dispatch_turn("mo spotify tren web", {
        "act": "EXECUTE", "goal": "WEB_OPEN", "parameters": {"target": "spotify"},
    })

    assert result["status"] == "ERROR"
    assert result["skill_id"] == "web.open"
    assert executor.calls[0][1] == {"target": "spotify", "browser": None}


def test_model_supported_but_unimplemented_semantic_is_not_model_unsupported(tmp_path):
    harness = AgentHarness(registry(tmp_path), skill_executor=Executor())

    result = harness._dispatch_turn("focus spotify", {
        "act": "EXECUTE", "goal": "APPLICATION_CONTROL",
        "parameters": {"action": "FOCUS", "application": "spotify"},
    })

    assert result["status"] == "ERROR"
    assert result["error"] == "SKILL_NOT_AVAILABLE"


def test_system_control_is_not_semantically_reachable(tmp_path):
    harness = AgentHarness(registry(tmp_path), skill_executor=Executor())

    result = harness._dispatch_turn("tat may", {
        "act": "EXECUTE", "goal": "SYSTEM_CONTROL", "parameters": {"action": "SHUTDOWN"},
    })

    assert result["error"] == "SKILL_NOT_AVAILABLE"


def test_confirmation_grant_and_fallback_boundaries(tmp_path):
    from Runtime.Policy.Confirmation import ConfirmationStore
    from Runtime.Providers.Registry import ProviderRegistry
    from Runtime.Resources.Dispatcher import ResourceDispatcher
    from Runtime.Skills.Executor import SkillExecutor

    skills = tmp_path / "skills.json"
    resources = tmp_path / "resources.json"
    skills.write_text(json.dumps([{
        "skill_id": "danger", "enabled": True, "confirmation_required": True,
        "resources": ["danger.run"],
    }]), encoding="utf-8")
    resources.write_text(json.dumps([{
        "resource_id": "danger.run", "enabled": True, "risk": "HIGH",
        "arguments": {"action": {"type": "string", "required": True}},
    }]), encoding="utf-8")
    calls = []
    providers = ProviderRegistry()
    providers.register("first", {"danger.run": lambda *_: calls.append("first") or {
        "ok": False, "error": "PROVIDER_DISCONNECTED", "side_effect_state": "NOT_STARTED",
    }}, priority=20)
    providers.register("second", {"danger.run": lambda *_: calls.append("second") or {"ok": True}}, priority=10)
    store = ConfirmationStore()
    executor = SkillExecutor(skills, ResourceDispatcher(providers, resources, confirmation_store=store))

    assert executor.execute("danger", {"action": "RUN"}, True)["error"] == "CONFIRMATION_REQUIRED"
    grant = store.issue("req", "danger", "danger.run", {"action": "RUN"}, "HIGH")
    result = executor.execute("danger", {"action": "RUN"}, True, confirmation_grant=grant)
    assert result["ok"] is True and calls == ["first", "second"]
    assert result["request_id"] == "req"
    assert executor.execute("danger", {"action": "RUN"}, True, confirmation_grant=grant)["error"] == "POLICY_DENIED"


def test_negative_phrase_containing_ok_never_confirms(tmp_path):
    from datetime import datetime, timedelta
    from Runtime.engine import PendingFrame

    executor = Executor()
    harness = AgentHarness(registry(tmp_path), skill_executor=executor)
    harness.pending_frame = PendingFrame(
        request_id="req", skill_id="application.close",
        resource_id="application.control.close", arguments={"application": "notepad"},
        risk="MEDIUM", created_at=datetime.now(), expires_at=datetime.now() + timedelta(seconds=60),
        description="đóng notepad",
    )

    result = harness._dispatch_turn("không ok", {
        "act": "RESPOND", "goal": "SOCIAL_RESPONSE", "parameters": {"intent": "ACKNOWLEDGEMENT"},
    })

    assert result["status"] != "EXECUTED"
    assert harness.pending_frame is not None
    assert all(call[0] != "application.close" for call in executor.calls)


def test_semantic_parity_is_bidirectional_and_local_app_needs_no_web_url():
    from pathlib import Path
    from Runtime.Contracts.Semantics import ROUTES, validate_semantic_parity
    from Runtime.Contracts.Validate import load_manifest

    root = Path(__file__).parents[1] / "Runtime"
    ontology = json.loads((root / "Model/VSAD/0.0.4/config.json").read_text(encoding="utf-8"))
    skills = load_manifest(root / "Registry/skills.json")
    resources = {item["resource_id"]: item for item in load_manifest(root / "Registry/resources.json")}

    assert validate_semantic_parity(ontology, skills, ROUTES) == []
    missing = dict(ROUTES)
    missing.pop(("APPLICATION_CONTROL", "FOCUS"))
    assert "MODEL_SEMANTIC_UNDECLARED:APPLICATION_CONTROL:FOCUS" in validate_semantic_parity(
        ontology, skills, missing,
    )
    assert resources["application.catalog.resolve"]["result"]["web_url"].get("required_on_success") is not True
