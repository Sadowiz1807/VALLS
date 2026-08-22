---
name: application-control
description: Use for opening or closing a local application on the current device. Trigger when the semantic frame requests APPLICATION_CONTROL with OPEN or CLOSE. Resolve only applications enabled in Runtime/Registry/applications.json.
version: 1.0.0
platforms: [windows]
metadata:
  family: ApplicationControl
  workflow_ids: [application.open, application.close]
---

# Application Control

## Purpose

Open or close a registry-approved local application and return observed window evidence. This file documents behavior; it never grants a capability. Runtime Registry remains the machine-readable authority.

## Trigger contract

Use this family only when the semantic frame contains:

- `act: EXECUTE`
- `goal: APPLICATION_CONTROL`
- `parameters.action: OPEN | CLOSE`
- `parameters.application`: stable ID, name, or approved alias

Reference phrases: “mở Notepad”, “đóng Spotify”, “thoát Calculator”. Keywords do not override the semantic frame.

## Workflows

### `application.open`

1. Validate required application input.
2. Resolve the application from `applications.json`.
3. Require an enabled local executable.
4. In dry-run, return `EXECUTION_DISABLED` without spawning a process.
5. In execute mode, call `application.control.open` with argv—not raw shell.
6. Compare visible windows before/after launch and own only the newly observed handle.
7. Report success only with handle, PID, and title evidence.

Risk: LOW. Confirmation: no.

### `application.close`

1. Resolve one unambiguous application.
2. Create a MEDIUM-risk confirmation turn.
3. Dispatch `application.control.close` only after explicit confirmation.
4. Close only the window handle owned by this runtime instance.
5. Verify that handle no longer exists before reporting success.

Risk: MEDIUM. Confirmation: required.

## Result contract

Return `skill_id`, `resource_id`, `ok`, `error`, resolved application ID, and evidence. Never render “đã mở/đã đóng” when `ok=false` or evidence is missing.

## Safety boundaries

- Reject applications missing or disabled in Registry.
- Do not accept executable paths, PID values, or shell commands from user/model input.
- Do not use process-name-wide termination; never close unrelated windows of the same application.
- If there is no owned window handle, return `NO_OWNED_WINDOW`.
- Do not silently substitute another application.
- Do not mutate Registry during a request.

## Examples

- `APPLICATION_CONTROL/OPEN/notepad` → `application.open`.
- `APPLICATION_CONTROL/CLOSE/notepad` → confirmation → `application.close`.
- A web entity such as YouTube routes to `web.close`, not process termination.
