import json
from pathlib import Path

from Runtime.engine import AgentHarness


def registry(tmp_path: Path) -> Path:
    (tmp_path / "applications.json").write_text(json.dumps([{
        "app_id": "spotify",
        "name": "Spotify",
        "aliases": ["spotify"],
        "enabled": True,
        "local": {"executable": "spotify.exe"},
        "web": {"url": "https://open.spotify.com"},
    }]), encoding="utf-8")
    (tmp_path / "browsers.json").write_text(json.dumps([{
        "browser_id": "chrome",
        "aliases": ["chrome"],
        "executable": "chrome.exe",
        "enabled": True,
    }]), encoding="utf-8")
    (tmp_path / "skills.json").write_text(json.dumps([
        {"skill_id": skill_id, "enabled": True}
        for skill_id in ("application.open", "application.close", "web.open", "media.play", "media.transport")
    ]), encoding="utf-8")
    return tmp_path


class Process:
    pid = 42


def frame(**parameters):
    return {"act": "EXECUTE", "goal": "APPLICATION_CONTROL", "parameters": {"action": "OPEN", "application": "spotify", **parameters}}


def test_local_dispatch_reports_started_process(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda executable: executable)
    harness = AgentHarness(registry(tmp_path), execute=True, runner=lambda argv: calls.append(argv) or Process())

    result = harness._dispatch_turn("mở spotify", frame())

    assert result["status"] == "EXECUTED"
    assert result["result"]["ok"] is True
    assert result["result"]["evidence"] == {"pid": 42}
    assert calls == [["spotify.exe"]]


def test_failed_local_dispatch_never_reports_executed(monkeypatch, tmp_path):
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda executable: executable)

    def fail(_argv):
        raise OSError("blocked")

    result = AgentHarness(registry(tmp_path), execute=True, runner=fail)._dispatch_turn("mở spotify", frame())

    assert result["status"] == "ERROR"
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "blocked"


def test_web_dispatch_uses_registry_url(tmp_path):
    opened = []
    harness = AgentHarness(registry(tmp_path), execute=True, web_opener=lambda url: opened.append(url) or True)

    result = harness._dispatch_turn("mở spotify trên web", frame(route="WEB"))

    assert result["status"] == "EXECUTED"
    assert result["result"]["ok"] is True
    assert result["result"]["evidence"] == {"opened": True}
    assert opened == ["https://open.spotify.com"]


def test_explicit_missing_browser_fails_without_fallback(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda _executable: None)
    harness = AgentHarness(registry(tmp_path), execute=True, web_opener=lambda url: opened.append(url) or True)

    result = harness._dispatch_turn("mở spotify bằng chrome", frame(route="WEB", browser="chrome"))

    assert result["status"] == "ERROR"
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "BROWSER_NOT_FOUND"
    assert opened == []


def test_explicit_browser_starts_with_registry_url(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda executable: f"C:/bin/{executable}")
    harness = AgentHarness(registry(tmp_path), execute=True, runner=lambda argv: calls.append(argv) or Process())

    result = harness._dispatch_turn("mở spotify bằng chrome", frame(route="WEB", browser="chrome"))

    assert result["status"] == "EXECUTED"
    assert result["result"]["ok"] is True
    assert result["result"]["evidence"] == {"pid": 42}
    assert calls == [["C:/bin/chrome.exe", "https://open.spotify.com"]]


def test_dry_run_has_no_side_effect(tmp_path):
    harness = AgentHarness(registry(tmp_path), runner=lambda _argv: (_ for _ in ()).throw(AssertionError()), web_opener=lambda _url: (_ for _ in ()).throw(AssertionError()))

    result = harness._dispatch_turn("mở spotify trên web", frame(route="WEB"))

    assert result["status"] == "ROUTED"
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "EXECUTION_DISABLED"


def test_web_open_goal_uses_application_registry(tmp_path):
    harness = AgentHarness(registry(tmp_path))
    model_frame = {"act": "EXECUTE", "goal": "WEB_OPEN", "parameters": {"target": "spotify"}}

    result = harness._dispatch_turn("mở spotify trên web", model_frame)

    assert result["status"] == "ROUTED"
    assert result["route"] == "WEB"
    assert result["app_id"] == "spotify"
    assert result["result"]["dry_run"] is True


def test_invalid_model_span_falls_back_to_raw_input(tmp_path):
    harness = AgentHarness(registry(tmp_path))
    model_frame = {
        "act": "EXECUTE",
        "goal": "WEB_OPEN",
        "parameters": {"target": {"source": "input_span", "value": "."}},
    }

    result = harness._dispatch_turn("mở spotify trên web", model_frame)

    assert result["status"] == "ROUTED"
    assert result["app_id"] == "spotify"
