# VSAD 0.0.4 Prototype — Kế hoạch hoàn thiện

> **Phạm vi:** Planning-only. Chưa sửa code, chưa thực thi side effect ngoài việc ghi plan.

**Mục tiêu:** Hoàn thiện prototype voice assistant dùng VSAD 0.0.4 làm semantic router, Runtime Registry quyết định local app/web fallback, và execution layer an toàn có bằng chứng.

**Kiến trúc chốt:** VSAD chỉ trả về act/goal/parameters/spans. Runtime chịu trách nhiệm resolve registry, kiểm tra khả dụng local, chọn web fallback, confirmation/risk, dispatch adapter và grounded response. Không biến VSAD 40M thành general-purpose LLM.

**Source of truth đã kiểm tra:**
- `C:\Users\ASUS\AL_voice_local\Runtime\engine.py`
- `C:\Users\ASUS\AL_voice_local\Runtime\Registry\applications.json`
- `C:\Users\ASUS\AL_voice_local\Runtime\Registry\skills.json`
- `C:\Users\ASUS\AL_voice_local\Runtime\model.py`
- `C:\Users\ASUS\AL_voice_local\Application\cli_runner.py`
- `C:\Users\ASUS\AL_voice_local\Runtime\Model\VSAD\0.0.4\`

---

## Trạng thái và gap hiện tại

| Khu vực | Hiện trạng | Gap prototype |
|---|---|---|
| Model artifact | VSAD 0.0.4 đã có trong `Runtime/Model/VSAD/0.0.4` | Cần smoke-load và contract test với runtime |
| Registry resolve | Có exact/alias/fuzzy resolve | Chưa có schema rõ cho `web_url`, browser và local availability |
| Local app | `engine.py` truyền `executable` vào mock executor | Chưa kiểm tra `Path`/`shutil.which`, chưa phân biệt installed/unavailable |
| Web fallback | Chưa có nhánh `webbrowser`/browser dispatch | Cần thêm tối thiểu cho `WEB_OPEN` và local-missing fallback |
| Execution | `_execute_skill()` luôn trả `ok=True` giả lập | Cần adapter prototype có dry-run/evidence; không được báo đã mở nếu chưa dispatch thật |
| Safety | Confirmation shutdown/restart đã có | Cần giữ fail-closed cho app không tồn tại, URL không hợp lệ và skill chưa đăng ký |
| Tests | Có artifact/manual test cũ, chưa thấy test suite canonical | Cần test runtime deterministic + CLI smoke |
| Docs | Có specification/requirements/skills docs | Cần cập nhật contract 0.0.4 và flow local/web |

---

## Phase 0 — Gate và baseline

### Task 0.1: Chốt prototype contract
**Files:**
- Modify: `Runtime/Model/VSAD/0.0.4/config.json` nếu metadata release còn thiếu
- Create: `Docs/VSAD_0.0.4_PROTOTYPE_CONTRACT.md`

**Contract bắt buộc:**
- `APPLICATION_CONTROL`: `action`, `application`
- `WEB_OPEN`: `target`, optional `browser`
- `UNSUPPORTED`: runtime response cố định, không dispatch
- Local-first chỉ áp dụng khi user không chỉ định web/browser
- User chỉ định “trên web/bằng Chrome/Cốc Cốc” thì đi thẳng web route

**Verify:** parse JSON; load model; assert tokenizer/config/model compatible.

### Task 0.2: Tạo test baseline trước khi sửa Runtime
**Files:**
- Create: `tests/test_vsad_004_contract.py`
- Create: `tests/test_runtime_routing.py`

**Cases bắt buộc:**
- `mở spotify` → local-first decision
- `mở spotify trên web` → `WEB_OPEN`
- `mở spotify bằng cốc cốc` → web + browser
- app không có → unsupported/not-found, không execute
- shutdown/restart → awaiting confirmation; không false execute
- sleep → execute path hiện hành

**Verify:** chạy `python -m pytest -q` và ghi nhận failure hiện tại trước implementation.

---

## Phase 1 — Registry contract và resolver

### Task 1.1: Chuẩn hóa `applications.json`
**Files:**
- Modify: `Runtime/Registry/applications.json`

**Schema tối thiểu mỗi app:**
```json
{
  "app_id": "spotify",
  "name": "Spotify",
  "aliases": ["spotify"],
  "enabled": true,
  "local": {"executable": "spotify.exe"},
  "web": {"url": "https://open.spotify.com", "browsers": ["chrome", "coccoc", "edge"]}
}
```

Không đưa đường dẫn máy cá nhân cố định vào model. Registry là nơi chứa khả năng local/web.

### Task 1.2: Thêm validation registry
**Files:**
- Modify: `Runtime/engine.py` hoặc helper hiện có nếu đã tồn tại
- Test: `tests/test_runtime_routing.py`

**Rules:**
- `app_id`, `name` bắt buộc
- local executable là optional
- web URL phải là `http`/`https`
- app thiếu cả local và web → invalid/unsupported
- không fuzzy-resolve dưới threshold hiện có

**Verify:** invalid registry fixture bị từ chối; valid registry load được.

---

## Phase 2 — Local-first/Web fallback routing

### Task 2.1: Tách quyết định route khỏi execution
**Files:**
- Modify: `Runtime/engine.py`
- Test: `tests/test_runtime_routing.py`

**Decision output chuẩn:**
```json
{
  "status": "ROUTED",
  "route": "LOCAL" | "WEB" | "UNSUPPORTED",
  "app_id": "spotify",
  "browser": null,
  "reason": "LOCAL_AVAILABLE" | "WEB_REQUESTED" | "LOCAL_UNAVAILABLE" | "NO_CAPABILITY"
}
```

`Path.exists()`/`shutil.which()` chỉ dùng ở Runtime. Model không dự đoán availability.

### Task 2.2: Implement user intent precedence
**Files:**
- Modify: `Runtime/engine.py`
- Test: `tests/test_runtime_routing.py`

**Precedence:**
1. Explicit web/browser request → WEB
2. Otherwise local executable available → LOCAL
3. Local unavailable + web URL exists → WEB default browser
4. Không có capability → UNSUPPORTED

**Verify:** 4 nhánh trên pass bằng fake registry/executable, không mở app thật trong unit test.

---

## Phase 3 — Prototype dispatch adapters

### Task 3.1: Local adapter có bằng chứng
**Files:**
- Modify: `Runtime/engine.py` hoặc tạo `Runtime/adapters.py` nếu engine không còn gọn
- Test: `tests/test_runtime_dispatch.py`

**Behavior:**
- Dùng `subprocess.Popen` chỉ khi user chạy prototype thật
- Unit test inject runner giả
- Kết quả phải chứa `route`, `target`, `started`/`error`
- Không trả `EXECUTED` khi process start thất bại

### Task 3.2: Web adapter
**Files:**
- Modify: adapter/runtime
- Test: `tests/test_runtime_dispatch.py`

**Behavior:**
- `webbrowser.open(url)` cho default browser
- browser chỉ là preference/explicit target; nếu browser executable không tồn tại, trả lỗi grounded, không silently đổi route
- URL chỉ lấy từ registry hoặc user-approved URL policy; không cho model sinh executable/path tùy ý

### Task 3.3: CLI integration
**Files:**
- Modify: `Application/cli_runner.py`
- Test: `tests/test_cli_smoke.py`

**Verify:** CLI chạy các lệnh `mở spotify`, `mở spotify trên web`, app lạ ở dry-run; output JSON/grounded response nhất quán.

---

## Phase 4 — Model/runtime integration

### Task 4.1: Load VSAD 0.0.4 chính thức
**Files:**
- Modify: `Runtime/model.py` nếu còn hard-code version/path
- Test: `tests/test_vsad_004_contract.py`

**Verify:**
- load `VASD.safetensors`, `config.json`, `tokenizer.json`
- assert release `0.0.4`
- inference smoke cho local/web/unsupported
- không dùng artifact 0.0.3 fallback âm thầm

### Task 4.2: Normalize model frame
**Files:**
- Modify: `Runtime/model.py` hoặc boundary trong `engine.py`
- Test: `tests/test_vsad_004_contract.py`

Chuẩn hóa span/dict/string về một frame duy nhất trước dispatch; missing required parameters → clarification/unsupported, không execute.

---

## Phase 5 — Safety, memory và regression

### Task 5.1: Safety regression
**Files:**
- Create/modify: `tests/test_safety_regression.py`

Cases:
- shutdown/restart luôn cần confirm
- confirmation hết hạn
- cancel không execute
- app/path/URL không từ input trực tiếp được phép vượt registry
- malformed model frame fail-closed

### Task 5.2: Agentic multi-turn regression
**Files:**
- Create/modify: `tests/test_agentic_harness.py`

Cases:
- mở app → state `current_target_application`
- mở web → state `active_url`/browser
- follow-up có context
- history truncation đúng `max_history_turns`

**Verify:** `pytest -q`; không dùng screenshot-only hoặc mock-only để claim runtime PASS.

---

## Phase 6 — UAT prototype trên Windows

### Task 6.1: Dry-run UAT
**Files:**
- Create: `Artifacts/uat/vsad_0.0.4_dry_run.jsonl`
- Create: `Artifacts/uat/vsad_0.0.4_dry_run_report.md`

Chạy toàn bộ case với dispatch disabled; kiểm tra frame/route/response.

### Task 6.2: Real local/web UAT có kiểm soát
**Files:**
- Create: `Artifacts/uat/vsad_0.0.4_real_windows_report.md`

Chỉ test app an toàn như Calculator/Notepad/browser; Spotify chỉ test nếu registry xác nhận executable thực sự có mặt. Ghi command, status, route và evidence.

**Gate:** không claim PASS nếu không có output thật từ CLI/process/browser.

---

## Phase 7 — Dọn artifact và prototype handoff

### Task 7.1: Chuẩn hóa artifact
- Giữ `Runtime/Model/VSAD/0.0.4/`
- Giữ release package `C:\Users\ASUS\Transformer\releases\VSAD\0.0.4\` và ZIP/SHA256
- Không copy checkpoint training vào app runtime
- Xóa `__pycache__` và manual artifacts chỉ sau import/reference scan

### Task 7.2: Cập nhật docs
- Modify: `Docs/01_Specìication.md`
- Modify: `Docs/02_FunctionRequiment.md`
- Modify: `Docs/Skills.md`
- Create: `Docs/VSAD_0.0.4_PROTOTYPE_RUNBOOK.md`

Runbook phải ghi cách chạy, registry format, dry-run, real-run, rollback về 0.0.3 và known limitations.

---

## Definition of Done

- VSAD 0.0.4 load được từ đúng 4-file release contract.
- Local/web routing đúng 4 nhánh và có test.
- App không tồn tại không bị báo execute.
- Explicit web/browser request không bị local-first override.
- High-risk commands vẫn confirmation-gated.
- Local/web adapters trả evidence-grounded result.
- Unit/integration/UAT có output thật; không dùng synthetic evidence để claim PASS.
- Docs và registry khớp runtime.
- Prototype có rollback rõ ràng về VSAD 0.0.3.

## Out of scope

- Retrain thêm VSAD.
- General QA/LLM fallback.
- Tự động crawl registry từ Internet.
- Browser automation đầy đủ ngoài mở URL.
- Production service, telemetry, distributed execution.

## Thứ tự thực thi đề xuất

`Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7`

Gate chặn quan trọng nhất là Phase 1–3: nếu routing/dispatch chưa grounded thì không mở rộng skill hay UAT.
