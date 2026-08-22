# Phase 2 — Skill thực chiến: Contract & Execution Checklist

**Ngày:** 2026-08-21  
**Trạng thái:** `PASS — ALL CURRENT SKILLS ACCEPTED WITH GENERATED METADATA; VSAD RETRAIN DEFERRED`  
**Phase 1 evidence:** `Artifacts/UserPlan_Phase1_Report.md` — PASS

> File này là gate duy nhất trước khi code Phase 2. Approval cho phép bắt đầu implementation; checkbox Gate 1–6 chỉ được tick sau khi có evidence thật.

## 0. Cấu trúc và source-of-truth đã chốt

```text
Skills/
├── ApplicationControl/Skill.md
├── WebControl/Skill.md
├── MediaControl/Skill.md
└── SystemControl/Skill.md

Runtime/
├── Registry/                 # whitelist/config authority
├── Resources/                # atomic system/provider operations
└── Skills/                   # workflow implementation classes
```

- `Skills/<Family>/Skill.md` là tài liệu dành cho người đọc: mục đích, trường hợp dùng, keywords tham khảo, workflow IDs, resources, risk và điều cấm.
- `Runtime/Registry/*.json` là machine-readable source of truth cho capability/whitelist/aliases/enabled/config. Skill không được dùng app, browser, provider hoặc resource ngoài registry.
- `Runtime/Skills/<Family>.py` chứa class workflow theo domain. Một class có thể có nhiều method, nhưng mỗi workflow vẫn có `skill_id` độc lập.
- `Runtime/Resources/<Domain>.py` chứa operation nguyên tử. Mỗi public method tương ứng một `resource_id` độc lập.
- Runtime không parse `Skill.md` để quyết định capability. Nếu docs khác registry, registry thắng và docs phải được sửa.

Ví dụ mapping, không phải tên class bắt buộc:

```text
WebControl.open()       → web.open
WebControl.change_tab() → tab.manage
Browser.open()          → browser.navigation.open
Browser.switch_tab()    → browser.tabs.switch
```

Whitelist rule:

- Entity/capability không có entry hoặc `enabled=false` → `UNSUPPORTED`, tuyệt đối không fallback sang target khác.
- Khi user yêu cầu capability chưa whitelist, runtime hỏi có muốn thêm capability đó vào whitelist; việc hỏi không tự ghi registry và không dispatch.
- Registry hiện tại whitelist Chrome, Cốc Cốc và Edge; Firefox là `UNSUPPORTED`.

## 1. Mục tiêu Phase 2

Biến semantic frame đã validate thành tác vụ có kết quả quan sát được:

```text
VOICE_FINAL.text
→ VSAD frame
→ Runtime resolve skill + resource
→ validate input/risk/availability
→ execute hoặc dry-run
→ grounded result
→ UI response
```

Skill là workflow. Resource là operation nguyên tử. Model không được tạo executable, URL, shell command, skill ID hoặc kết quả success.

## 2. Phạm vi MVP đề xuất

| Thứ tự | Skill ID | Input bắt buộc | Resource tối thiểu | Risk | Kết quả thật cần chứng minh |
|---:|---|---|---|---|---|
| 1 | `application.open` | `application` | `application.catalog.resolve`, `application.control.open` | LOW | Process start thành công/thất bại |
| 2 | `application.close` | `application` | `application.catalog.resolve`, `application.control.close` | MEDIUM | Đúng process bị đóng hoặc fail grounded |
| 3 | `web.open` | `target`, optional `browser` | `application.catalog.resolve`, `browser.navigation.open` | LOW | URL registry được mở hoặc fail grounded |
| 4 | `media.play` | optional `query`, `platform` | `media.catalog.resolve`, `media.playback.play` | LOW | Provider xác nhận playback hoặc trả lỗi |
| 5 | `media.transport` | `action` | `media.playback.pause/resume/stop/next/previous` | LOW | Trạng thái playback thay đổi hoặc trả lỗi |

### Phạm vi đã mở rộng và được user duyệt

- `web.close`: đóng đúng tab web theo entity/title Registry.
- `system.power`: shutdown/restart/sleep.
- `system.brightness`, `system.volume`, `system.night_light`.

### Vẫn ngoài phạm vi

- Lock/screenshot, `web.search`, `web.navigate`, tab switching đầy đủ.
- TTS, browser automation đầy đủ, raw shell, plugin marketplace.
- Skill riêng cho từng app/provider.

Chỉ thêm các mục trên sau khi 5 skill MVP có evidence thật và user mở scope.

## 3. Contract semantic đầu vào

Runtime chỉ resolve skill từ frame đã validate:

```json
{
  "act": "EXECUTE",
  "goal": "APPLICATION_CONTROL",
  "parameters": {
    "action": "OPEN",
    "application": {"source": "input_span", "value": "Spotify"}
  }
}
```

Checklist:

- [ ] `act` thuộc allowlist của skill.
- [ ] `goal` và generic `action` khớp `accepts` trong registry.
- [ ] Required input tồn tại và có kiểu đúng.
- [ ] Entity span phải nằm trong raw input; span lỗi thì resolve lại từ raw input.
- [ ] Không resolve duy nhất thì trả `CLARIFICATION_NEEDED` hoặc `UNSUPPORTED`.
- [ ] Không dùng `response_text` để chọn skill/resource.
- [ ] Model không được cung cấp executable, URL, process ID hoặc shell command.
- [ ] Entity/capability phải tồn tại và enabled trong registry; keyword trong `Skill.md` không cấp quyền dispatch.
- [ ] Capability chưa whitelist trả `UNSUPPORTED` kèm câu hỏi opt-in để thêm whitelist; không tự sửa registry.

## 4. Skill manifest contract

Mỗi skill entry bắt buộc có đúng các trường sau:

```json
{
  "skill_id": "application.open",
  "version": "1.0",
  "enabled": true,
  "accepts": {
    "act": ["EXECUTE"],
    "goal": "APPLICATION_CONTROL",
    "action": "OPEN"
  },
  "inputs": {
    "application": {"type": "string", "required": true}
  },
  "resources": [
    "application.catalog.resolve",
    "application.control.open"
  ],
  "risk": "LOW",
  "confirmation_required": false
}
```

Checklist:

- [ ] `skill_id` dùng lowercase dot notation `<domain>.<workflow>`.
- [ ] `version`, `enabled`, `accepts`, `inputs`, `resources`, `risk`, `confirmation_required` tồn tại.
- [ ] Không chứa executable path, URL, credential, secret hoặc raw command.
- [ ] Skill risk không thấp hơn resource risk cao nhất.
- [ ] Skill ID không trùng và không tái sử dụng ID đã disable cho nghĩa khác.
- [ ] Python class/file dùng PascalCase theo family; public `skill_id` vẫn lowercase dot notation.
- [ ] Keywords trong `Skill.md` chỉ là documentation; aliases operational nằm trong registry.

## 5. Resource contract

Mỗi resource biểu diễn đúng một operation:

```json
{
  "resource_id": "application.control.open",
  "version": "1.0",
  "arguments": {
    "application_id": {"type": "string", "required": true}
  },
  "result": {
    "ok": "boolean",
    "target": "string|null",
    "error": "string|null",
    "evidence": "object"
  },
  "risk": "LOW",
  "enabled": true
}
```

Checklist:

- [ ] Resource arguments/result có kiểu rõ ràng.
- [ ] Resource không tự gọi resource khác để tạo workflow ẩn.
- [ ] Availability lấy từ runtime/registry, không lấy từ model.
- [ ] Adapter chỉ nhận ID/URL đã resolve từ registry.
- [ ] Resource disabled/missing làm skill fail closed.
- [ ] Timeout/cancellation được định nghĩa tại adapter gọi side effect.
- [ ] Resource implementation không tự mở rộng whitelist hoặc tự thêm provider/browser/app.

## 6. Execution envelope

Runtime tạo envelope sau resolve/validation; model không tạo:

```json
{
  "request_id": "local-uuid",
  "skill_id": "application.open",
  "inputs": {"application": "Spotify"},
  "resolved": {
    "application_id": "spotify",
    "target": "spotify.exe"
  },
  "risk": "LOW",
  "confirmation_required": false,
  "dry_run": true
}
```

Checklist:

- [ ] Có `request_id`, `skill_id`, `inputs`, `resolved`, `risk`, `confirmation_required`, `dry_run`.
- [ ] `resolved.target` đến từ registry/resource resolver.
- [ ] Mặc định `dry_run=true`.
- [ ] Real execution chỉ khi caller truyền explicit execution permission.
- [ ] Một request chỉ chạy một workflow; không parallel execution trong MVP.

## 7. Result envelope và grounded response

Mọi executor trả cùng shape:

```json
{
  "ok": true,
  "skill_id": "application.open",
  "resource_id": "application.control.open",
  "target": "spotify.exe",
  "started": true,
  "dry_run": false,
  "error": null,
  "evidence": {
    "pid": 1234
  }
}
```

Rules:

- [ ] `ok=true` chỉ sau khi adapter có evidence quan sát được.
- [ ] Dry-run luôn `started=false`, `dry_run=true`; không dùng từ “đã mở/đã phát”.
- [ ] Lỗi adapter giữ nguyên error code/message cần thiết; skill không che lỗi.
- [ ] Response success chỉ render sau `ok=true`.
- [ ] Response failure dùng target/error từ result, không dùng prose model để bịa success.
- [ ] Không lưu credential, token hoặc raw audio trong evidence/log.

## 8. Safety gate

- [ ] Unknown skill/resource/entity → `UNSUPPORTED`, không dispatch.
- [ ] Capability ngoài registry → `UNSUPPORTED`; không silently fallback và không tự thêm whitelist.
- [ ] Malformed frame → `INVALID_FRAME` hoặc `CLARIFICATION_NEEDED`.
- [ ] URL chỉ chấp nhận `http://` hoặc `https://` từ registry/policy.
- [ ] Không dùng `shell=True`; process argument dùng list.
- [ ] Không nhận raw shell từ transcript/model/user.
- [ ] `application.close` không đóng process nếu entity resolve mơ hồ.
- [ ] Medium/high-risk workflow cần confirmation turn riêng.
- [ ] CANCEL/expired confirmation không dispatch.
- [ ] Test không được mở/đóng app, browser hay media thật; inject adapter giả.
- [ ] Real UAT chỉ chạy từng case đã user cho phép.

## 9. Implementation checklist

### Gate 0 — Approval

- [x] User duyệt 5 skill MVP.
- [x] User duyệt contract input/manifest/resource/execution/result.
- [x] User duyệt `application.close` là MEDIUM risk.
- [x] User duyệt cấu trúc `Skills/`, `Runtime/Skills/`, `Runtime/Resources/`.
- [x] User duyệt Registry là whitelist/config authority.
- [x] User duyệt Gate 2–6 về yêu cầu; checkbox evidence bên dưới vẫn chưa tick.
- [x] Spotify là provider media đầu tiên theo registry/project hiện tại.

### Gate 1 — Xóa false-success

- [x] Viết RED chứng minh `_execute_skill()` hiện trả success dù không có executor.
- [x] Thay `_execute_skill()` bằng lookup + dispatch thật tối thiểu.
- [x] Missing/disabled skill/resource fail closed.

### Gate 2 — Application skills

- [x] RED/GREEN `application.open` với fake process runner.
- [x] RED/GREEN `application.close` với fake process inventory/terminator.
- [x] Local unavailable/failure không báo `EXECUTED`.

### Gate 3 — Web skill

- [x] RED/GREEN `web.open` với fake opener.
- [x] Explicit browser missing trả grounded failure, không silent fallback.
- [x] URL ngoài registry/policy bị reject.

### Gate 4 — Media skills

- [x] Chốt provider đầu tiên và adapter native/API có sẵn.
- [x] RED/GREEN `media.play`.
- [x] RED/GREEN `media.transport` cho pause/resume/stop/next/previous.
- [x] Không có active playback session → grounded failure.

### Gate 5 — Integration

- [ ] `VOICE_FINAL → VSAD → skill → resource → result → UI` PASS — deferred đến sau VSAD retrain.
- [x] Generated valid metadata → skill → resource → observed result PASS.
- [x] Unknown/malformed/ambiguous inputs fail closed.
- [x] Response không claim success từ mock/planned result.
- [x] Full test suite PASS.

### Gate 6 — Controlled UAT

- [x] User duyệt từng side effect trước real UAT.
- [x] Application open/close có process evidence.
- [x] Web open có browser/URL evidence.
- [x] Media play/transport có provider playback evidence.
- [x] Web close có tab-level evidence và không kill browser.
- [x] System brightness/volume/Night Light/Sleep có state evidence thật và rollback khi phù hợp.
- [x] Shutdown/restart được user nghiệm thu PASS theo confirmation + Windows operation evidence.
- [x] Lưu report; không lưu secret hoặc raw audio.

## 10. Definition of Done Phase 2

Phase 2 skill/runtime acceptance hiện PASS khi dùng generated valid metadata:

- [x] Mọi skill hiện tại resolve đúng generated semantic contract.
- [x] Không còn executor trả success giả.
- [x] Mọi dependency tồn tại/enabled và schema hợp lệ.
- [x] Dry-run mặc định, real-run explicit.
- [x] Error propagation và grounded response PASS.
- [x] Safety tests PASS.
- [x] Full current test suite PASS.
- [x] Controlled UAT có evidence thật PASS.
- [x] Runtime registry/docs khớp implementation.

Full Voice/VSAD integration là gate riêng được hoãn đến sau retraining; không dùng parser hard-code để ghi đè model.

## 11. User approval

Trạng thái đã chốt:

- [x] **APPROVED WITH CHANGES** — thực thi đúng 5 skill MVP với cấu trúc family class và Registry authority tại mục 0.
- [ ] **REJECTED** — không thực thi Phase 2.

### Thay đổi/yêu cầu bổ sung

```text
Registry là whitelist/config authority. Skill docs và implementation không được mở rộng capability ngoài registry.
Gate 2–6 đã được duyệt về yêu cầu nhưng chỉ PASS sau evidence thực thi.
```

Bắt đầu implementation từ Gate 1. Không tự mở rộng ngoài 5 skill MVP hoặc capability đang được registry cho phép.
