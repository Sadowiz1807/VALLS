---
name: system-control
description: Use for controlling the current Windows device: shutdown, restart, sleep, display brightness, system volume, and Night Light. Trigger for SYSTEM_CONTROL frames. Apply strict confirmation, range validation, Registry capability flags, and observed-state evidence.
version: 1.0.0
platforms: [windows]
compatibility: Windows 10/11; CIM brightness; Core Audio volume; Windows Settings UIA Night Light
metadata:
  family: SystemControl
  workflow_ids: [system.power, system.brightness, system.volume, system.night_light]
---

# System Control

## Purpose

Control approved device-level capabilities using fixed Windows operations. `Runtime/Registry/system.json` is the authority for enabled actions and numeric ranges. This document cannot enable a disabled capability.

## Trigger contract

Use only for:

- `act: EXECUTE`
- `goal: SYSTEM_CONTROL`
- action `SHUTDOWN | RESTART | SLEEP | SET_BRIGHTNESS | SET_VOLUME | NIGHT_LIGHT_ON | NIGHT_LIGHT_OFF`
- integer `value` from 0 to 100 for brightness or volume

Reference phrases: “tắt nguồn”, “restart máy”, “sleep”, “độ sáng 60%”, “âm lượng 40%”, “bật Night Light”.

## Workflows

### `system.power`

- `SHUTDOWN`: HIGH risk, explicit confirmation required.
- `RESTART`: HIGH risk, explicit confirmation required.
- `SLEEP`: MEDIUM risk; execute only with explicit real-run permission.
- Use fixed argv from the resource; never accept raw commands.
- Sleep calls native `PowrProf.SetSuspendState`; success requires suspend return evidence and observed resume.
- Shutdown/restart use fixed Windows commands and always require confirmation.

### `system.brightness`

Validate 0–100, call Windows WMI brightness method, read current brightness back, and require exact observed-state evidence. Risk: LOW.

### `system.volume`

Validate 0–100, set the default Windows playback endpoint through Core Audio, read scalar volume back, and require observed-state evidence. Risk: LOW.

### `system.night_light`

Open the native Windows Night Light Settings page, locate the stable manual ON/OFF UI Automation controls, invoke only when requested state differs, and read the automation ID back after the change. Do not modify undocumented CloudStore binary values.

## Result contract

Return `skill_id`, atomic `resource_id`, `ok`, action/value, `error`, and evidence such as return code or observed percentage/state. Dry-run returns `EXECUTION_DISABLED` and performs no state change.

## Safety boundaries

- Reject values outside 0–100 and non-integers.
- Confirm shutdown/restart in a separate turn; CANCEL/expiry must not dispatch.
- Never accept raw PowerShell, shell, DLL, executable, or Registry paths from semantic metadata.
- Do not report success without read-back or command evidence.
- Night Light must fail if UI Automation cannot read the resulting state.
- Do not auto-enable capability flags or expand action lists.

## Examples

- `SYSTEM_CONTROL/SET_VOLUME/value=60` → `system.volume`.
- `SYSTEM_CONTROL/SHUTDOWN` → confirmation → `system.power`.
- `SYSTEM_CONTROL/NIGHT_LIGHT_ON` → Settings UIA toggle → observed `true`.
