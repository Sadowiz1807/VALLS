import json
import itertools
from pathlib import Path

import pytest

from Runtime.Providers.Application import ApplicationProvider
from Runtime.Providers.Browser import BrowserProvider
from Runtime.Providers.Media import MediaProvider
from Runtime.engine import AgentHarness


class FakeSkillExecutor:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def execute(self, skill_id, args, execute, confirmed=False):
        self.calls.append((skill_id, args, execute, confirmed))
        response = self.responses.get(skill_id, {"ok": True, "evidence": {"called": True}})
        return {"skill_id": skill_id, **response}


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
    resources = {
        "application.open": ["application.catalog.resolve", "application.control.open"],
        "application.close": ["application.catalog.resolve", "application.control.close"],
        "web.open": ["browser.navigation.open"],
        "media.play": ["media.playback.play"],
        "media.transport": ["media.playback.pause", "media.playback.resume", "media.playback.stop", "media.playback.next", "media.playback.previous"],
    }
    (path / "skills.json").write_text(json.dumps([
        {"skill_id": skill_id, "enabled": True, "resources": resource_ids}
        for skill_id, resource_ids in resources.items()
    ]), encoding="utf-8")


def test_application_open_and_close_use_owned_window_only(tmp_path):
    write_registry(tmp_path)
    snapshots = iter([
        [],
        [{"handle": 7, "pid": 42, "title": "Untitled - Notepad"}],
    ])
    closed = []
    resource = ApplicationProvider(
        tmp_path / "applications.json",
        runner=lambda _argv: type("Process", (), {"pid": 42})(),
        observer=lambda: next(snapshots),
        closer=lambda handle: closed.append(handle) or {"handle": handle, "closed": True},
    )
    opened = resource.open("notepad", execute=True)
    closed_result = resource.close("notepad", execute=True)

    assert opened["ok"] is True and opened["evidence"] == {
        "handle": 7, "pid": 42, "title": "Untitled - Notepad"
    }
    assert closed_result["ok"] is True
    assert closed == [7]


def test_application_open_ignores_unrelated_new_window(tmp_path):
    write_registry(tmp_path)
    snapshots = iter([
        [],
        [{"handle": 90, "pid": 900, "title": "Unrelated"}],
        [{"handle": 90, "pid": 900, "title": "Unrelated"},
         {"handle": 101, "pid": 42, "title": "Notepad"}],
    ])
    resource = ApplicationProvider(
        tmp_path / "applications.json",
        runner=lambda _argv: type("Process", (), {"pid": 42})(),
        observer=lambda: next(snapshots),
    )

    result = resource.open("notepad", execute=True)

    assert result["ok"] is True
    assert result["evidence"]["handle"] == 101


def test_application_open_accepts_launcher_child_window(tmp_path):
    write_registry(tmp_path)
    snapshots = iter([
        [],
        [{"handle": 101, "pid": 84, "title": "Notepad"}],
    ])
    resource = ApplicationProvider(
        tmp_path / "applications.json",
        runner=lambda _argv: type("Process", (), {"pid": 42})(),
        observer=lambda: next(snapshots),
        parent_pid=lambda pid: 42 if pid == 84 else None,
    )

    result = resource.open("notepad", execute=True)

    assert result["ok"] is True
    assert result["evidence"]["pid"] == 84


def test_application_open_rejects_matching_title_from_other_pid(tmp_path, monkeypatch):
    write_registry(tmp_path)
    times = iter([0, 1, 6])
    monkeypatch.setattr("Runtime.Providers.Application.time.time", lambda: next(times))
    monkeypatch.setattr("Runtime.Providers.Application.time.sleep", lambda _seconds: None)
    other = [{"handle": 90, "pid": 999, "title": "Notepad"}]
    snapshots = itertools.chain([[], other], itertools.repeat(other))
    resource = ApplicationProvider(
        tmp_path / "applications.json",
        runner=lambda _argv: type("Process", (), {"pid": 42})(),
        observer=lambda: next(snapshots),
    )

    result = resource.open("notepad", execute=True)

    assert result["ok"] is False
    assert result["error"] == "WINDOW_EVIDENCE_MISSING"


def test_application_close_without_owned_window_fails_closed(tmp_path):
    write_registry(tmp_path)
    result = ApplicationProvider(tmp_path / "applications.json").close("notepad", execute=True)

    assert result["ok"] is False
    assert result["error"] == "NO_OWNED_WINDOW"


def test_web_open_rejects_browser_outside_registry(tmp_path):
    write_registry(tmp_path)
    opened = []
    provider = BrowserProvider(tmp_path / "browsers.json", opener=lambda url: opened.append(url) or True)

    result = provider.open("https://open.spotify.com", browser="firefox", execute=True)

    assert result["ok"] is False
    assert result["error"] == "BROWSER_UNSUPPORTED"
    assert result["ask_to_add_whitelist"] == "firefox"
    assert opened == []


def test_web_open_uses_whitelisted_default_browser(tmp_path):
    write_registry(tmp_path)
    opened = []
    provider = BrowserProvider(tmp_path / "browsers.json", opener=lambda url: opened.append(url) or True)

    result = provider.open("https://open.spotify.com", execute=True)

    assert result["ok"] is True
    assert result["evidence"] == {"opened": True}
    assert opened == ["https://open.spotify.com"]


def test_media_play_and_transport_propagate_provider_result():
    calls = []
    provider = MediaProvider(command=lambda args: calls.append(args) or {"ok": True, "action": args[0]})

    played = provider.run(["play", "Daft Punk One More Time"], platform="spotify", execute=True)
    paused = provider.run(["pause"], platform="spotify", execute=True)

    assert played["ok"] is True and calls[0] == ["play", "Daft Punk One More Time"]
    assert paused["ok"] is True and calls[1] == ["pause"]


def test_media_pause_requires_observed_stopped_state():
    states = iter([{"ok": True, "playing": True}, {"ok": True, "playing": False}])
    provider = MediaProvider(
        command=lambda _args: {"ok": True, "action": "pause"},
        observer=lambda: next(states),
        wait=lambda _seconds: None,
    )

    result = provider.run(["pause"], execute=True)

    assert result["ok"] is True
    assert result["observed"]["playing"] is False


@pytest.mark.parametrize(("action", "command"), [
    ("PAUSE", "pause"), ("RESUME", "resume"), ("STOP", "pause"),
    ("NEXT", "next"), ("PREVIOUS", "previous"),
])
def test_media_provider_runs_all_transport_commands(action, command):
    calls = []
    result = MediaProvider(command=lambda args: calls.append(args) or {"ok": True}).run(
        [command], platform="spotify", execute=True,
    )
    assert result["ok"] is True
    assert calls == [[command]]


def test_media_play_uses_registry_default_device(tmp_path):
    (tmp_path / "media_providers.json").write_text(json.dumps([{
        "provider_id": "spotify", "name": "Spotify", "aliases": ["spotify"],
        "enabled": True, "device": "Web Player (Chrome)",
    }]), encoding="utf-8")
    calls = []
    provider = MediaProvider(tmp_path / "media_providers.json", command=lambda args: calls.append(args) or {"ok": True})

    result = provider.run(["play", "One More Time"], platform="spotify", execute=True)

    assert result["ok"] is True
    assert calls == [["play", "One More Time", "--device", "Web Player (Chrome)"]]


def test_application_close_requires_confirmation(tmp_path):
    write_registry(tmp_path)
    executor = FakeSkillExecutor()
    harness = AgentHarness(tmp_path, execute=True, skill_executor=executor)
    frame = {"act": "EXECUTE", "goal": "APPLICATION_CONTROL", "parameters": {"action": "CLOSE", "application": "notepad"}}

    pending = harness._dispatch_turn("đóng notepad", frame)
    confirmed = harness._dispatch_turn("đồng ý", {"act": "CONFIRM", "goal": "APPLICATION_CONTROL", "parameters": {}})

    assert pending["status"] == "AWAITING_CONFIRMATION"
    assert executor.calls == [("application.close", {"application": "notepad"}, True, True)]
    assert confirmed["status"] == "EXECUTED" and confirmed["result"]["ok"] is True


def test_confirmed_skill_failure_returns_error_status(tmp_path):
    write_registry(tmp_path)
    executor = FakeSkillExecutor({"application.close": {"ok": False, "error": "blocked"}})
    harness = AgentHarness(tmp_path, execute=True, skill_executor=executor)
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
    executor = FakeSkillExecutor()
    harness = AgentHarness(tmp_path, execute=True, skill_executor=executor)

    skill_ids = ("application.open", "application.close", "web.open", "media.play", "media.transport")
    for skill_id in skill_ids:
        assert harness._execute_skill(skill_id, {"value": 1})["ok"] is True

    assert [skill_id for skill_id, _, _, _ in executor.calls] == list(skill_ids)
