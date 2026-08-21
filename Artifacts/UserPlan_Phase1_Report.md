# User Plan — Phase 1: Replace Voice ASR with NVIDIA Parakeet

Date: 2026-08-21
Status: PASS

## Scope

- Removed Faster-Whisper implementation and contract from `Voice/`.
- Kept `VoiceProcess`, `VoiceEvent`, and `MicrophoneCapture` public boundaries.
- Replaced transcription with `nvidia/parakeet-ctc-0.6b-vi` via NeMo 2.6.
- Preserved mono float32 16 kHz capture.
- Added configurable RMS no-speech gate (`min_rms=0.02`).
- Transcription runs only after explicit `VoiceProcess.stop()`.
- Temporary WAV is mono PCM16 16 kHz and always deleted.

## Files

- `Voice/transcriber.py`
- `Voice/__init__.py`
- `Tests/test_transcriber.py`
- `Tests/test_phase2_integration.py`
- `Docs/Baseline.md`
- `Docs/PHASE_1_INPUT_UI_CONTRACT.md`
- `requirements.txt`
- `Artifacts/UserPlan_Phase1_Report.md`

## Verification

```text
compileall: PASS
pytest Tests: 26 passed in 2.05s
uv pip check: No broken requirements found
Faster-Whisper source/docs references: 0
faster_whisper installed in .venv: false
```

Real shared transcriber smoke:

```text
model: nvidia/parakeet-ctc-0.6b-vi
cold load: 13.0 s
1-second inference: 0.15 s
temporary WAV leaks: 0
```

Real production Voice pipeline smoke:

```text
Parakeet load → microphone start → 1-second capture → explicit stop
VOICE_STARTED → VOICE_CANCELLED
elapsed: 14.18 s
result: PASS
```

The one-second microphone sample contained no intentional speech, so it verifies runtime/no-speech behavior only and is not ASR accuracy evidence.

## Known tuning gaps

- `min_rms=0.02` needs calibration on saved, consented benchmark audio.
- Proper nouns such as Facebook still need accuracy evaluation/tuning.
- CPU cold start is approximately 13 seconds; model preloading should be considered only when application startup requirements are fixed.
- NeMo warnings about training/validation config and missing ffmpeg do not block WAV inference.

## Phase Gate

- Faster-Whisper removed from production Voice: PASS
- NVIDIA Parakeet wired into production Voice: PASS
- No-speech does not emit hallucinated `VOICE_FINAL`: PASS
- Explicit stop produces final/cancel event: PASS
- Full current tests: PASS
- Real microphone/model boundary: PASS

Blockers: none for Phase 1. Accuracy calibration remains follow-up tuning, not a wiring blocker.
