import json

from App.cli_runner import run_once


class FakeModel:
    def infer(self, text, context=None, state=None):
        web = "web" in text
        return {
            "act": "EXECUTE",
            "goal": "WEB_OPEN" if web else "APPLICATION_CONTROL",
            "parameters": ({"target": "spotify"} if web else {
                "action": "OPEN", "application": "unknown" if "la" in text else "spotify",
            }),
        }


def test_cli_dry_run_outputs_grounded_json(capsys):
    result = run_once("mo spotify tren web", model=FakeModel())
    printed = json.loads(capsys.readouterr().out)

    assert result["status"] == "ROUTED"
    assert result["skill_id"] == "web.open"
    assert result["result"]["dry_run"] is True
    assert printed == result


def test_cli_unknown_app_is_runtime_error_not_model_unsupported(capsys):
    result = run_once("mo app la", model=FakeModel())
    json.loads(capsys.readouterr().out)

    assert result["status"] == "ERROR"
    assert result["result"]["error"] == "APPLICATION_UNSUPPORTED"
