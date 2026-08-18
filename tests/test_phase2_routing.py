import json
from pathlib import Path

from Runtime.engine import AgentHarness


class FakeModel:
    def __init__(self, frame):
        self.frame = frame

    def infer(self, raw_input, context=None, state=None):
        return self.frame


def make_registry(tmp_path, *, executable="missing.exe", web_url=None):
    registry = tmp_path / "registry"
    registry.mkdir()
    app = {
        "app_id": "spotify",
        "name": "Spotify",
        "aliases": ["spotify"],
        "enabled": True,
        "local": {"executable": executable},
    }
    if web_url:
        app["web"] = {"url": web_url, "open_with": "browser.open_url"}
    (registry / "applications.json").write_text(json.dumps([app]), encoding="utf-8")
    return registry


def frame(**parameters):
    return {"act": "EXECUTE", "goal": "APPLICATION_CONTROL", "parameters": parameters}


def test_local_route_does_not_execute(tmp_path, monkeypatch):
    registry = make_registry(tmp_path, executable="spotify.exe", web_url="https://open.spotify.com")
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda value: "C:/Spotify/spotify.exe")
    harness = AgentHarness(registry)

    result = harness.step("mở spotify", FakeModel(frame(action="OPEN", application="spotify")))

    assert result["status"] == "ROUTED"
    assert result["route"] == "LOCAL"
    assert result["reason"] == "LOCAL_AVAILABLE"
    assert result["result"] is None


def test_explicit_web_route_wins(tmp_path, monkeypatch):
    registry = make_registry(tmp_path, executable="spotify.exe", web_url="https://open.spotify.com")
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda value: "C:/Spotify/spotify.exe")
    harness = AgentHarness(registry)

    result = harness.step("mở spotify trên web", FakeModel(frame(action="OPEN", application="spotify", route="WEB")))

    assert result["status"] == "ROUTED"
    assert result["route"] == "WEB"
    assert result["reason"] == "WEB_REQUESTED"


def test_local_missing_falls_back_to_web(tmp_path, monkeypatch):
    registry = make_registry(tmp_path, executable="missing.exe", web_url="https://open.spotify.com")
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda value: None)
    harness = AgentHarness(registry)

    result = harness.step("mở spotify", FakeModel(frame(action="OPEN", application="spotify")))

    assert result["route"] == "WEB"
    assert result["reason"] == "LOCAL_UNAVAILABLE_WEB_AVAILABLE"


def test_no_capability_is_unsupported(tmp_path, monkeypatch):
    registry = make_registry(tmp_path, executable="missing.exe")
    monkeypatch.setattr("Runtime.engine.shutil.which", lambda value: None)
    harness = AgentHarness(registry)

    result = harness.step("mở spotify", FakeModel(frame(action="OPEN", application="spotify")))

    assert result["status"] == "UNSUPPORTED"
    assert result["reason"] == "NO_CAPABILITY"
