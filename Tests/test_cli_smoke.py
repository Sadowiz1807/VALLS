import json

from App.cli_runner import run_once


class FakeModel:
    def infer(self, text, context=None, state=None):
        app = "unknown" if "lạ" in text else "spotify"
        return {
            "act": "EXECUTE",
            "goal": "APPLICATION_CONTROL",
            "parameters": {
                "action": "OPEN",
                "application": app,
                "route": "WEB" if "web" in text else None,
            },
        }


def test_cli_dry_run_outputs_grounded_json(capsys):
    result = run_once("mở spotify trên web", model=FakeModel())
    printed = json.loads(capsys.readouterr().out)

    assert result["status"] == "ROUTED"
    assert result["route"] == "WEB"
    assert result["result"]["dry_run"] is True
    assert printed == result


def test_cli_unknown_app_is_not_executed(capsys):
    result = run_once("mở app lạ", model=FakeModel())
    json.loads(capsys.readouterr().out)

    assert result["status"] == "UNSUPPORTED"
    assert result["reason"] == "APP_NOT_FOUND"
