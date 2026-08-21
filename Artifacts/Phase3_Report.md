# Phase 3 — Prototype dispatch adapters

Date: 2026-08-21
Status: PARTIAL — BLOCKED at end-to-end CLI semantic gate

## Files written/modified

- `.gitignore`
- `App/cli_runner.py`
- `Runtime/engine.py`
- `Runtime/Registry/applications.json`
- `Runtime/Registry/browsers.json`
- `Tests/test_runtime_dispatch.py`
- `Tests/test_cli_smoke.py`
- `Tests/test_phase2_integration.py`
- `Artifacts/Phase3_Report.md`

## Phase Gate

- Task 3.1 local adapter with injected runner and grounded result: PASS
- Task 3.1 process-start failure does not report `EXECUTED`: PASS
- Task 3.2 default web adapter uses registry URL: PASS
- Task 3.2 explicit missing browser fails without fallback: PASS
- Task 3.2 explicit browser invokes `[browser_executable, registry_url]`: PASS
- Task 3.3 CLI defaults to dry-run and emits JSON: PASS
- Task 3.3 unknown app fails closed: PASS
- Full current test suite: PASS — `26 passed in 2.00s`
- Compile check: PASS — `compileall` exit 0
- Real VSAD CLI case `mở spotify trên web`: PASS — WEB dry-run, `started=false`
- Real VSAD CLI case unknown app: PASS — `UNSUPPORTED/APP_NOT_FOUND`
- Real VSAD CLI case `mở spotify`: FAIL — VSAD 0.0.4 returns `RUN_COMMAND/RESTART_SYSTEM`; runtime safely returns `AWAITING_CONFIRMATION` and does not execute

## Evidence

Focused Phase 3 tests:

```text
10 passed in 1.34s
```

Full suite:

```text
26 passed in 2.00s
```

Real model output causing blocker:

```text
mở spotify => {
  "act": "EXECUTE",
  "goal": "RUN_COMMAND",
  "parameters": {"command_id": "RESTART_SYSTEM"}
}
```

Runtime response:

```text
AWAITING_CONFIRMATION, risk=HIGH
```

No real app/browser dispatch was performed. All CLI verification used dry-run.

## Blocker

VSAD 0.0.4 semantic classification for `mở spotify` must be corrected or safely normalized at the model/runtime boundary before Phase 3 can be marked complete. No hard-coded phrase parser was added because it would duplicate semantic authority and hide the model defect.
