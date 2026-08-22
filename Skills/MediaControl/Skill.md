---
name: media-control
description: Use for playing or transporting media through an enabled provider. Trigger for MEDIA_CONTROL frames requesting play, pause, resume, stop, next, or previous. Use only providers and devices configured in Runtime Registry.
version: 1.0.0
platforms: [windows]
metadata:
  family: MediaControl
  workflow_ids: [media.play, media.transport]
---

# Media Control

## Purpose

Play content or control an existing playback session through a Registry-approved provider. Spotify is the current provider. This file documents orchestration; `media_providers.json` owns provider availability and device configuration.

## Trigger contract

Use for semantic frames with:

- `act: EXECUTE`
- `goal: MEDIA_CONTROL`
- action `PLAY | PAUSE | RESUME | STOP | NEXT | PREVIOUS`
- optional query for PLAY
- optional provider/platform

Reference phrases: “phát One More Time”, “tạm dừng nhạc”, “bài tiếp theo”.

## Workflows

### `media.play`

1. Resolve provider from Registry.
2. Validate optional query.
3. Apply Registry-owned default device when configured.
4. In dry-run, return without provider activity.
5. Report success only when provider returns `ok=true` and track/device evidence.

### `media.transport`

Map only approved actions to atomic provider operations: pause, resume, stop, next, previous. Propagate provider failures unchanged.

Risk: LOW. Confirmation: no.

## Result contract

Return `skill_id`, atomic `resource_id`, provider ID, `ok`, `error`, and provider evidence. Do not claim playback from a planned query or inactive device.

## Safety boundaries

- Do not use providers absent or disabled in Registry.
- Do not store credentials, OAuth tokens, or device identifiers in reports.
- Do not download media.
- Do not treat system-volume control as media transport; it belongs to `SystemControl/system.volume`.
- Do not silently select an unconfigured provider.

## Examples

- `MEDIA_CONTROL/PLAY` with query → `media.play`.
- `MEDIA_CONTROL/PAUSE` → `media.transport`.
- No active provider device → grounded provider error, not success.
