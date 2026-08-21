# AL Voice Local — Baseline Architecture

> **Trạng thái:** Baseline chuẩn
> **Model:** VSAD 0.0.4 prototype
> **Nền tảng:** Windows 10
> **Phạm vi:** Kiến trúc nền tảng đã chốt cho prototype; không mô tả kế hoạch phase khác.

## 1. Pipeline

```text
Microphone
  ↓
Voice Process / NVIDIA Parakeet CTC 0.6B Vietnamese
  ↓ VOICE_FINAL.text
VSAD 0.0.4
  ↓ normalized metadata
Routing
  ↓ skill + resource plan
Skill
  ↓ grounded result
UI Pygame
```

## 2. Boundary

| Thành phần | Trách nhiệm | Không làm |
|---|---|---|
| Voice | Thu microphone, lọc no-speech theo RMS, NVIDIA Parakeet ASR, phát voice event | Không resolve app, không gọi skill |
| UI | Pygame window, event loop, render action (màu + decoration) | Không chứa routing/business logic |
| Model | Text → metadata theo VSAD 0.0.4 | Không kiểm tra executable, URL, app availability |
| Routing | Validate metadata, resolve registry, chọn route/skill/resource | Không tự tạo capability ngoài registry |
| Skill docs | Mô tả family/workflow, usage, keywords tham khảo, resources và điều cấm | Không cấp quyền capability, không phải runtime config |
| Runtime Skill | Điều phối workflow trong giới hạn registry | Không parse ngôn ngữ tự nhiên, không gọi system/provider trực tiếp |
| Runtime Resource | Thực thi operation nguyên tử với target đã resolve | Không quyết định intent, không mở rộng whitelist |
| Registry | Config/whitelist authority cho app, browser, provider, skill và resource | Không chứa workflow implementation |
| Result | Trạng thái và evidence grounded | Không báo thành công nếu chưa có evidence |

## 3. Voice và Model contract

### Voice event

```json
{
  "event": "VOICE_FINAL",
  "text": "mở spotify trên web",
  "language": "vi",
  "source": "microphone",
  "confidence": 0.96,
  "request_id": "uuid",
  "timestamp": "ISO-8601"
}
```

Event hỗ trợ:

```text
VOICE_STARTED
VOICE_PARTIAL      optional, không gửi sang model
VOICE_FINAL        text cuối gửi sang model
VOICE_CANCELLED
VOICE_ERROR
```

### VSAD metadata

```json
{
  "act": "EXECUTE",
  "goal": "APPLICATION_CONTROL",
  "parameters": {
    "action": "OPEN",
    "application": "spotify",
    "route": "WEB",
    "browser": "coccoc"
  },
  "response": null,
  "model_version": "VSAD-0.0.4"
}
```

Model chỉ nhận text. Model không:

- kiểm tra app/executable;
- sinh URL hoặc path;
- gọi subprocess/browser/filesystem;
- tự kết luận app không được hỗ trợ vì chưa có trong training data.

## 4. Routing contract

Routing quyết định theo thứ tự:

```text
1. User chỉ định web/browser → WEB
2. Local executable available → LOCAL
3. Local unavailable + web URL hợp lệ → WEB
4. Không có capability → UNSUPPORTED
```

Output mẫu:

```json
{
  "status": "READY",
  "route": "WEB",
  "skill_id": "browser.open_url",
  "app_id": "spotify",
  "browser_id": "coccoc",
  "resource": {
    "url": "https://open.spotify.com",
    "executable": "browser.exe"
  },
  "reason": "WEB_REQUESTED"
}
```

Không dispatch khi metadata malformed, app/browser không resolve được, resource unavailable, URL không hợp lệ hoặc skill không nằm trong allowlist.

## 5. Skill và Resource

Skill prototype:

```text
application.open
browser.open_url
system.sleep
media.play / media.transport / media.volume
```

Resource registry:

```text
Runtime/Registry/applications.json
Runtime/Registry/browsers.json
Runtime/Registry/skills.json
```

`applications.json` chứa app/service. `browsers.json` chứa browser. `skills.json` chứa skill config machine-readable. Registry là whitelist/config authority: entry thiếu hoặc `enabled=false` là `UNSUPPORTED`; skill docs/code không được tự thêm capability. Một request có thể hỏi user có muốn thêm capability chưa hỗ trợ, nhưng không tự ghi registry hoặc dispatch.

Registry hiện tại cho phép Chrome, Cốc Cốc và Edge; Firefox chưa được hỗ trợ.

## 6. Pygame UI contract

Pygame là owner của window, event loop, render và clock. UI không dùng image frames — mỗi action là một tổ hợp màu nền + màu nhấn + decoration vẽ bằng pygame.draw:

```text
Action         Nền              Nhấn             Decoration
normal         (10,14,20)       (60,70,90)       lưới chấm tĩnh
listening      (8,16,30)        (0,150,255)      vòng tròn lan rộng
speech         (6,22,18)        (0,200,100)      cột sóng âm
thinking       (22,20,6)        (200,200,0)      chấm quỹ đạo
processing     (26,16,6)        (255,165,0)      cung quay
executing      (28,12,8)        (255,100,0)      cột tiến trình
success        (6,22,10)        (0,200,0)        vòng nở + dấu tick
error          (26,6,6)         (220,40,40)      chữ X nhấp nháy
```

Action mapping:

```text
IDLE              → normal
LISTENING         → listening
SPEECH_ACTIVITY   → speech
TRANSCRIBING      → thinking
MODEL_INFERENCE   → thinking
ROUTING           → processing
SKILL_EXECUTION   → executing
SUCCESS           → success → normal
ERROR             → error → normal
```

UI hiển thị tên action và subtitle (transcript/model metadata) bằng text. Decoration tính theo thời gian, không đọc file mỗi frame. Thiếu assets không xảy ra vì không còn phụ thuộc file ảnh; lỗi không lường trước fallback về `normal`.

## 7. Threading và request policy

- Pygame chạy trên UI thread.
- Microphone, Parakeet, model, routing và skill không chạy trên UI thread.
- Worker chỉ gửi event/state cho UI, không gọi Pygame trực tiếp.
- Xử lý tuần tự, không chạy nhiều skill song song.
- Queue tối đa 1 request đang chờ.
- Không bỏ qua request cũ để chạy request mới.
- Có thể cancel ở `listening`, `thinking`, `processing`.

## 8. Safety và result

| Tình huống | Kết quả |
|---|---|
| Microphone/Whisper lỗi | `ERROR` |
| Model lỗi | `ERROR` |
| Metadata malformed | `INVALID_FRAME`, fail-closed |
| App không có registry | `APP_NOT_FOUND` |
| Browser explicit không có | `BROWSER_NOT_FOUND` |
| URL invalid/not allowlisted | `INVALID_RESOURCE` |
| Skill chưa đăng ký | `SKILL_NOT_FOUND` |
| Shutdown/restart | `AWAITING_CONFIRMATION` |
| User cancel | `CANCELLED` |

Không lưu audio. Logging mặc định chỉ là structured status; không ghi transcript lâu dài.

## 9. Cấu trúc thư mục

```text
AL_voice_local/
├── App/
│   ├── Frontend/
│   │   └── app.py                     # Pygame window/event loop, color+decoration
│   └── cli_runner.py
├── Voice/                              # microphone + transcription (VoiceProcess)
├── Skills/                             # human-readable skill family docs
│   └── <Family>/Skill.md
├── Runtime/
│   ├── engine.py
│   ├── model.py
│   ├── vsad_adapter.py                 # VSAD adapter boundary
│   ├── Skills/                         # workflow implementation classes
│   ├── Resources/                      # atomic system/provider operations
│   ├── Registry/
│   │   ├── applications.json
│   │   ├── browsers.json
│   │   └── skills.json
│   └── Model/VSAD/0.0.4/
│       ├── VASD.safetensors
│       ├── config.json
│       ├── tokenizer.json
│       └── requirements.txt
├── Docs/
│   ├── baseline.md
│   └── PHASE_1_INPUT_UI_CONTRACT.md
├── tests/
└── Artifacts/
```

Chỉ tạo module khi bắt đầu dùng; ưu tiên reuse `Runtime/engine.py` và `Runtime/model.py`. `__pycache__/` là artifact tự sinh, không thuộc source contract.

## 10. Dependency contract

```text
pygame
nemo_toolkit[asr] 2.6
sounddevice
numpy
Pydantic
```

VSAD runtime artifact giữ contract 4 file: model, config, tokenizer và requirements.
