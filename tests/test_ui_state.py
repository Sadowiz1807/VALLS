from UI.app import UIAction, UIConfig, UIState


def test_action_timeout_returns_to_normal():
    state = UIState(UIConfig(success_seconds=3.0, error_seconds=5.0))
    state.set_action(UIAction.SUCCESS, now=10.0)
    assert state.tick(now=12.9) is UIAction.SUCCESS
    assert state.tick(now=13.0) is UIAction.NORMAL


def test_error_timeout_is_five_seconds():
    state = UIState(UIConfig(error_seconds=5.0))
    state.set_action(UIAction.ERROR, now=10.0)
    assert state.tick(now=14.9) is UIAction.ERROR
    assert state.tick(now=15.0) is UIAction.NORMAL


def test_processing_action_does_not_auto_reset():
    state = UIState()
    state.set_action(UIAction.PROCESSING, now=10.0)
    assert state.tick(now=100.0) is UIAction.PROCESSING
