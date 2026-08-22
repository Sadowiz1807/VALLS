import json
from pathlib import Path

import pytest

from Runtime.Resources.Application import Application
from Runtime.Resources.Browser import Browser
from Runtime.Resources.Media import Media
from Runtime.Skills.ApplicationControl import ApplicationControl
from Runtime.Skills.MediaControl import MediaControl
from Runtime.Skills.WebControl import WebControl
from Runtime.engine import AgentHarness


def write_registry(path: Path):
    (path / "applications.json").write_text(json.dumps([
        {"app_id": "notepad", "name": "Notepad", "aliases": ["notepad"], "enabled": True,
         "local": {"executable": "notepad.exe"}},
        {"app_id": "spotify", "name": "Spotify", "aliases": ["spotify"], "enabled": True,
         "web": {"url": "https://open.spotify.com"}},
    ]), encoding="utf-8")
    (path / "browsers.json").write_text(json.dumps([
        {"app_id": "chrome", "name": "Chrome", "aliases": ["chrome"], "enabled": True,
         "local": {"executable": "chrome.exe"}},
    ]), encoding="utf-8")
    (path / "skills.json").write_text(json.dumps([
        {"skill_id": skill_id, "enabled": True}
        for skill_id in ("application.open", "application.close", "web.open", "media.play", "media.transport")
    ]), encoding="utf-8")


def test_application_open_and_close_use_owned_window_only(tmp_path):
    write_registry(tmp_path)
    snapshots = iter([
        [],
        [{"handle": 7, "pid": 42, "title": "Untitled - Notepad"}],
    ])
    closed = []
    resource = Application(
        tmp_path / "applications.json",
        runner=lambda _argv: type("Process", (), {"pid": 1})(),
        observer=lambda: next(snapshots),
        closer=lambda handle: closed.append(handle) or {"handle": handle, "closed": True},
    )
    skill = ApplicationControl(resource)

    opened = skill.open("notepad", execute=True)
    closed_result = skill.close("notepad", execute=True)

    assert opened["ok"] is True and opened["evidence"] == {
        "handle": 7, "pid": 42, "title": "Untitled - Notepad"
    }
    assert closed_result["ok"] is True
    assert closed == [7]


def test_application_close_without_owned_window_fails_closed(tmp_path):
    write_registry(tmp_path)
    result = Application(tmp_path / "applications.json").close("notepad", execute=True)

    assert result["ok"] is False
    assert result["error"] == "NO_OWNED_WINDOW"


def test_web_open_rejects_browser_outside_registry(tmp_path):
    write_registry(tmp_path)
    opened = []
    skill = WebControl(Browser(tmp_path / "browsers.json", opener=lambda url: opened.append(url) or True))

    result = skill.open("https://open.spotify.com", browser="firefox", execute=True)

    assert result["ok"] is False
    assert result["error"] == "BROWSER_UNSUPPORTED"
    assert result["ask_to_add_whitelist"] == "firefox"
    assert opened == []


def test_web_open_uses_whitelisted_default_browser(tmp_path):
    write_registry(tmp_path)
    opened = []
    skill = WebControl(Browser(tmp_path / "browsers.json", opener=lambda url: opened.append(url) or True))

    result = skill.open("https://open.spotify.com", execute=True)

    assert result["ok"] is True
    assert result["evidence"] == {"opened": True}
    assert opened == ["https://open.spotify.com"]


def test_media_play_and_transport_propagate_provider_result():
    calls = []
    resource = Media(command=lambda args: calls.append(args) or {"ok": True, "action": args[0]})
    skill = MediaControl(resource)

    played = skill.play("Daft Punk One More Time", platform="spotify", execute=True)
    paused = skill.transport("PAUSE", platform="spotify", execute=True)

    assert played["ok"] is True and calls[0] == ["play", "Daft Punk One More Time"]
    assert paused["ok"] is True and calls[1] == ["pause"]


@pytest.mark.parametrize(("action", "command"), [
    ("PAUSE", "pause"), ("RESUME", "resume"), ("STOP", "pause"),
    ("NEXT", "next"), ("PREVIOUS", "previous"),
])
def test_media_transport_maps_all_contract_actions(action, command):
    calls = []
    result = MediaControl(Media(command=lambda args: calls.append(args) or {"ok": True})).transport(
        action, platform="spotify", execute=True,
    )
    assert result["ok"] is True
    assert calls == [[command]]


def test_media_play_uses_registry_default_device(tmp_path):
    (tmp_path / "media_providers.json").write_text(json.dumps([{
        "provider_id": "spotify", "name": "Spotify", "aliases": ["spotify"],
        "enabled": True, "device": "Web Player (Chrome)",
    }]), encoding="utf-8")
    calls = []
    skill = MediaControl(Media(tmp_path / "media_providers.json", command=lambda args: calls.append(args) or {"ok": True}))

    result = skill.play("One More Time", platform="spotify", execute=True)

    assert result["ok"] is True
    assert calls == [["play", "One More Time", "--device", "Web Player (Chrome)"]]


def test_application_close_requires_confirmation(tmp_path):
    write_registry(tmp_path)
    calls = []
    harness = AgentHarness(tmp_path, execute=True)
    harness.skill_handlers["application.close"] = lambda args: calls.append(args) or {
        "ok": True, "skill_id": "application.close", "evidence": {"returncode": 0}
    }
    frame = {"act": "EXECUTE", "goal": "APPLICATION_CONTROL", "parameters": {"action": "CLOSE", "application": "notepad"}}

    pending = harness._dispatch_turn("đóng notepad", frame)
    confirmed = harness._dispatch_turn("đồng ý", {"act": "CONFIRM", "goal": "APPLICATION_CONTROL", "parameters": {}})

    assert pending["status"] == "AWAITING_CONFIRMATION"
    assert calls == [{"application": "notepad"}]
    assert confirmed["status"] == "EXECUTED" and confirmed["result"]["ok"] is True


def test_confirmed_skill_failure_returns_error_status(tmp_path):
    write_registry(tmp_path)
    harness = AgentHarness(tmp_path, execute=True)
    harness.skill_handlers["application.close"] = lambda _args: {
        "ok": False, "skill_id": "application.close", "error": "blocked"
    }
    frame = {"act": "EXECUTE", "goal": "APPLICATION_CONTROL", "parameters": {"action": "CLOSE", "application": "notepad"}}

    harness._dispatch_turn("đóng notepad", frame)
    result = harness._dispatch_turn("đồng ý", {"act": "CONFIRM", "goal": "APPLICATION_CONTROL", "parameters": {}})

    assert result["status"] == "ERROR"
    assert result["result"]["ok"] is False


def test_sleep_unimplemented_never_returns_executed(tmp_path):
    write_registry(tmp_path)
    skills = json.loads((tmp_path / "skills.json").read_text(encoding="utf-8"))
    skills.append({"skill_id": "system.command", "enabled": True})
    (tmp_path / "skills.json").write_text(json.dumps(skills), encoding="utf-8")

    result = AgentHarness(tmp_path, execute=True)._dispatch_turn("về chế độ sleep", {
        "act": "EXECUTE", "goal": "RUN_COMMAND", "parameters": {"command_id": "SLEEP_SYSTEM"},
    })

    assert result["status"] == "ERROR"
    assert result["result"]["error"] == "SKILL_NOT_IMPLEMENTED"


def test_harness_unknown_skill_never_returns_success(tmp_path):
    write_registry(tmp_path)
    result = AgentHarness(tmp_path)._execute_skill("missing.skill", {})

    assert result["ok"] is False
    assert result["error"] == "SKILL_NOT_FOUND"


def test_harness_dispatches_all_five_registered_skills(tmp_path):
    write_registry(tmp_path)
    calls = []
    harness = AgentHarness(tmp_path, execute=True)
    harness.skill_handlers = {
        skill_id: lambda args, skill_id=skill_id: calls.append((skill_id, args)) or {
            "ok": True, "skill_id": skill_id, "evidence": {"called": True}
        }
        for skill_id in ("application.open", "application.close", "web.open", "media.play", "media.transport")
    }

    for skill_id in harness.skill_handlers:
        assert harness._execute_skill(skill_id, {"value": 1})["ok"] is True

    assert [skill_id for skill_id, _ in calls] == list(harness.skill_handlers)
