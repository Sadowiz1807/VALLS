"""Phase 2 integration tests: Frontend, Voice, Model adapter."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from App.Frontend.app import UIAction, ACTION_STYLES, UIConfig, UIState

# ── UI state tests ──

def test_ui_state_normal():
    state = UIState(UIConfig())
    assert state.tick() is UIAction.NORMAL


def test_ui_state_success_times_out():
    state = UIState(UIConfig(success_seconds=3.0))
    state.set_action(UIAction.SUCCESS, now=10.0)
    assert state.tick(now=12.99) is UIAction.SUCCESS
    assert state.tick(now=13.0) is UIAction.NORMAL


def test_ui_state_error_times_out():
    state = UIState(UIConfig(error_seconds=5.0))
    state.set_action(UIAction.ERROR, now=10.0)
    assert state.tick(now=14.99) is UIAction.ERROR
    assert state.tick(now=15.0) is UIAction.NORMAL


def test_ui_state_non_auto_actions():
    state = UIState()
    state.set_action(UIAction.PROCESSING, now=10.0)
    assert state.tick(now=100.0) is UIAction.PROCESSING


# ── Palette: every action has a distinct color pair ──

def test_palette_covers_all_actions():
    assert set(ACTION_STYLES) == set(UIAction)


def test_palette_colors_distinct():
    accents = {v.accent for v in ACTION_STYLES.values()}
    assert len(accents) == len(UIAction), "accent colors must be unique"


# ── Voice process tests ──

def test_voice_event_fields():
    from Voice import VoiceEvent
    ev = VoiceEvent(event="VOICE_FINAL", text="test", confidence=0.95)
    assert ev.event == "VOICE_FINAL"
    assert ev.text == "test"
    assert ev.confidence == 0.95
    assert ev.source == "microphone"
    assert len(ev.request_id) == 12
    assert ev.timestamp.startswith("202")


def test_voice_event_auto_id():
    from Voice import VoiceEvent
    ev1 = VoiceEvent(event="VOICE_STARTED")
    ev2 = VoiceEvent(event="VOICE_STARTED")
    assert ev1.request_id != ev2.request_id


def test_voice_process_construct():
    from Voice import VoiceProcess
    vp = VoiceProcess(on_event=lambda e: None)
    assert vp.transcriber.model_name == "nvidia/parakeet-ctc-0.6b-vi"
    assert vp.capture.sample_rate == 16000


# ── Model adapter tests ──

def test_model_adapter_load():
    from Runtime.vsad_adapter import ModelAdapter
    adapter = ModelAdapter()
    adapter.load()
    assert adapter._model is not None


def test_model_adapter_infer():
    from Runtime.vsad_adapter import ModelAdapter
    adapter = ModelAdapter().load()
    result = adapter.infer("mở spotify")
    assert "act" in result
    assert "goal" in result
    assert "parameters" in result
    assert result["model_version"] == "VSAD-0.0.4"


# ── Routing smoke test ──

def test_routing_imports():
    from Runtime.engine import AgentHarness
    assert AgentHarness is not None


if __name__ == "__main__":
    tests = [
        test_ui_state_normal,
        test_ui_state_success_times_out,
        test_ui_state_error_times_out,
        test_ui_state_non_auto_actions,
        test_palette_covers_all_actions,
        test_palette_colors_distinct,
        test_voice_event_fields,
        test_voice_event_auto_id,
        test_voice_process_construct,
        test_model_adapter_load,
        test_model_adapter_infer,
        test_routing_imports,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)