# User Plan — Phase 2: Practical Skills

Date: 2026-08-22
Status: PASS — ALL CURRENT SKILLS ACCEPTED

## Acceptance scope

Phase 2 is accepted using generated valid semantic metadata injected into the canonical Runtime. VSAD/model inference is intentionally excluded until retraining.

```text
generated semantic frame
→ Runtime Registry resolution
→ Skill orchestration
→ Resource/provider operation
→ observed result
```

## Current canonical structure

```text
Skills/
├── ApplicationControl/Skill.md
├── WebControl/Skill.md
├── MediaControl/Skill.md
└── SystemControl/Skill.md

Runtime/
├── Registry/
│   ├── applications.json
│   ├── browsers.json
│   ├── media_providers.json
│   ├── skills.json
│   └── system.json
├── Resources/
│   ├── Application.py
│   ├── Browser.py
│   ├── Media.py
│   ├── Registry.py
│   └── System.py
└── Skills/
    ├── ApplicationControl.py
    ├── WebControl.py
    ├── MediaControl.py
    └── SystemControl.py
```

Registry remains the machine-readable capability, configuration, alias, and whitelist authority. `Skill.md` files are human-readable contracts and are not parsed to grant capability.

## Accepted skill matrix

| Skill/workflow | Real evidence | Status |
|---|---|---|
| `application.open` | newly created Notepad window observed by handle/PID/title | PASS |
| `application.close` | confirmation required; only runtime-owned window handle closed; unrelated windows preserved | PASS |
| `web.open` | Facebook/Spotify opened through configured Chrome/Cốc Cốc executable and Registry URL | PASS |
| `web.close` | all matching YouTube tabs closed through native UI Automation; browser and unrelated tabs preserved | PASS |
| `media.play` | Spotify album playback started on observed Connect device | PASS |
| `media.transport/PAUSE` | provider state changed to not playing | PASS |
| `media.transport/RESUME` | provider state changed to playing | PASS |
| `media.transport/NEXT` | track URI changed | PASS |
| `media.transport/PREVIOUS` | track URI returned to previous track | PASS |
| `media.transport/STOP` | provider pause semantic; final playback state false | PASS |
| `system.brightness` | 67→60 read-back; rollback 67 | PASS |
| `system.volume` | 100→60 read-back; rollback 100 | PASS |
| `system.night_light` | Settings UIA OFF→ON read-back; rollback OFF | PASS |
| `system.power/SLEEP` | canonical harness suspended PC and resumed; native evidence `suspended=true` | PASS |
| `system.power/SHUTDOWN` | Windows accepted operation; user accepted shutdown capability | PASS |
| `system.power/RESTART` | fixed Windows operation and confirmation policy verified; accepted with SystemControl family | PASS |
| `conversation.social` | canonical Runtime returned grounded greeting response | PASS |

## Requested generated-metadata cases

| Request | Result |
|---|---|
| `bật chế độ night light` | PASS |
| `chỉnh âm lượng về 60%` | PASS |
| `về chế độ sleep` | PASS |
| `mở facebook qua coccoc` | PASS |
| `tắt youtube` | PASS |

Detailed evidence: `Artifacts/GeneratedMetadata_Case_Report.md` and `Artifacts/SystemControl_Report.md`.

## Safety fixes completed

- Removed false-success skill dispatch.
- Disabled/missing/unknown capabilities fail closed.
- Confirmation failures propagate top-level `ERROR`.
- Removed process-wide `taskkill /IM` from application close.
- Application close requires confirmation and closes only an owned window handle.
- Browser close verifies all matching tabs are gone; it does not kill the browser.
- Explicit browsers use observed absolute executable paths; no silent fallback.
- Night Light uses Windows Settings UI Automation with state read-back; no undocumented CloudStore mutation.
- Sleep uses native `PowrProf.SetSuspendState`; the hanging `rundll32` path was removed.
- Shutdown/restart use fixed commands and require confirmation.
- No raw shell, executable path, URL, PID, credential, or provider result is accepted from model metadata.

## Automated verification

```text
branch: develop
pip check: No broken requirements found.
compileall: COMPILE_OK
pytest: 54 passed in 3.28s
registry/docs/safety verifier: PASS
```

## Deferred work

VSAD 0.0.4 still requires retraining before full voice end-to-end acceptance. This is explicitly deferred and does not invalidate the generated-metadata skill acceptance in this report.

After retraining, rerun:

```text
VOICE_FINAL → VSAD → Runtime → Skill → Resource → observed result → UI
```

## Decision

- All currently registered and enabled skills: PASS.
- Phase 2 skill/runtime scope: PASS.
- Full voice/model integration: DEFERRED UNTIL VSAD RETRAINING.
- Release readiness: not claimed; worktree remains uncommitted/dirty.
