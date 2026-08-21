# User Plan — Phase 2: Practical Skills

Date: 2026-08-21
Status: PARTIAL — BLOCKED BY VSAD SEMANTIC GATE

## Implemented structure

```text
Skills/
├── ApplicationControl/Skill.md
├── WebControl/Skill.md
└── MediaControl/Skill.md

Runtime/
├── Registry/
│   ├── applications.json
│   ├── browsers.json
│   ├── media_providers.json
│   └── skills.json
├── Resources/
│   ├── Application.py
│   ├── Browser.py
│   ├── Media.py
│   └── Registry.py
└── Skills/
    ├── ApplicationControl.py
    ├── WebControl.py
    └── MediaControl.py
```

Registry remains whitelist/config authority. Skill docs/code do not extend its capabilities.

## Automated verification

```text
pip check: No broken requirements found.
compileall: COMPILE_OK
pytest: 41 passed in 2.25s
```

False-success executor was removed. Unknown/disabled/unimplemented skills fail closed. Confirmed resource failures return top-level `ERROR`.

## Real controlled UAT

| Skill | Real evidence | Result |
|---|---|---|
| `application.open` | Notepad process created with PID evidence | PASS |
| `application.close` | MEDIUM confirmation turn, then `taskkill` return code 0 | PASS |
| `web.open` | Registry URL `https://open.spotify.com`; Chrome window title confirmed Spotify Web Player | PASS |
| `media.play` | Spotify played `One More Time` by Daft Punk; provider reported `playing=true` | PASS |
| `media.transport` | Spotify pause returned success; provider reported `playing=false` | PASS |

Firefox outside browser registry returned `BROWSER_UNSUPPORTED`, asked to add whitelist, and did not fallback.

No credentials, tokens, raw audio, or device identifiers are stored in this report.

## Blocking semantic evidence

VSAD 0.0.4 direct inference did not resolve the five representative requests correctly:

| Input | Observed model output | Expected |
|---|---|---|
| `mở notepad` | `CANCEL / APPLICATION_CONTROL` | `EXECUTE / APPLICATION_CONTROL / OPEN` |
| `đóng notepad` | Correct goal/action but missing application parameter | Complete CLOSE frame |
| `mở spotify trên web` | Correct goal but missing parameters | Target/application parameter |
| `phát Daft Punk One More Time trên spotify` | `APPLICATION_CONTROL / FOCUS` | `MEDIA_CONTROL / PLAY` |
| `tạm dừng nhạc` | `CANCEL / MEDIA_CONTROL` | `EXECUTE / MEDIA_CONTROL / PAUSE/STOP` |

The runtime must not add a hard-coded transcript parser that overrides VSAD. Therefore the full pipeline `VOICE_FINAL → VSAD → skill → resource → result → UI` cannot pass yet.

## Phase decision

- Practical implementation of all five MVP skills: PASS.
- Real execution of all five MVP skills: PASS.
- Phase 2 Definition of Done: NOT PASS.
- Blocker: retrain/fix VSAD 0.0.4 semantic outputs, then rerun Gate 5 full-flow acceptance.
