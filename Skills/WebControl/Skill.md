---
name: web-control
description: Use for opening a registry-approved web destination in an approved browser. Trigger for WEB_OPEN semantic frames, especially requests naming Chrome, Cốc Cốc, or Edge. Never expand browser or destination support outside Runtime Registry.
version: 1.0.0
platforms: [windows]
metadata:
  family: WebControl
  workflow_ids: [web.open, web.close]
---

# Web Control

## Purpose

Open an approved HTTP(S) destination with an enabled browser and return browser-open evidence. This document defines behavior; Registry defines available destinations, browsers, aliases, URLs, and executables.

## Trigger contract

Use only for:

- `act: EXECUTE`
- `goal: WEB_OPEN`
- required destination/application target
- optional browser preference

Reference phrases: “mở Spotify trên web”, “mở Facebook qua Cốc Cốc”. Reference phrases do not grant missing Registry entries.

## Workflows

### `web.open`

1. Resolve destination through `applications.json` or another approved destination registry.
2. Read its Registry-owned URL; accept only `http://` or `https://`.
3. If the user names a browser, resolve it through `browsers.json`.
4. Reject missing/disabled browser instead of falling back.
5. In dry-run, do not launch anything.
6. In execute mode, call `browser.navigation.open` and report success only after opener/process evidence.

Risk: LOW. Confirmation: no.

### `web.close`

1. Resolve a web entity and its Registry-owned window title.
2. Find one matching browser window.
3. Close every matching browser tab through its native UI Automation `TabCloseButton`.
4. Report success only after no matching tab remains. Never kill the browser process.

Risk: MEDIUM. Confirmation: no for a web tab.

## Result contract

Return `skill_id=web.open`, `resource_id=browser.navigation.open`, `ok`, destination URL/ID, browser ID when explicit, `error`, and evidence.

## Safety boundaries

- Do not accept `javascript:`, file paths, raw browser arguments, or raw shell.
- Do not invent URLs from prose when Registry has no destination.
- Do not add Firefox or another browser automatically.
- If an explicit browser is unsupported, return `BROWSER_UNSUPPORTED` and optionally ask whether the whitelist should be updated.
- Do not mutate Registry during dispatch.

## Examples

- Approved Spotify + Cốc Cốc → open Registry URL with Registry executable.
- Firefox absent from Registry → `BROWSER_UNSUPPORTED`, no fallback.
- Facebook + Cốc Cốc → open `https://www.facebook.com` with the configured Cốc Cốc executable.
- Close YouTube → close all YouTube tabs while preserving Cốc Cốc and unrelated tabs.
