# Phase 1 — Input/UI Contract

> **Trạng thái:** COMPLETE
> **Phạm vi:** Các quyết định và contract của Phase 1 בלבד. Không mô tả implementation phase khác.

## 1. Input contract

Pipeline của Phase 1:

```text
Microphone → Input Module → Faster-Whisper → INPUT_FINAL.text
```

Event:

```text
INPUT_STARTED
INPUT_PARTIAL       optional, không gửi sang model
INPUT_FINAL         text cuối cùng
INPUT_CANCELLED
INPUT_ERROR
```

`INPUT_FINAL`:

```json
{
  "event": "INPUT_FINAL",
  "text": "mở spotify trên web",
  "language": "vi",
  "source": "microphone",
  "confidence": 0.96,
  "request_id": "uuid",
  "timestamp": "ISO-8601"
}
```

## 2. Faster-Whisper

```text
model size: small
compute type: float16
device: GPU
microphone: sounddevice
sample rate: 16 kHz
VAD: Silero VAD
no-speech timeout: 5 seconds
```

Input Module sở hữu microphone stream và Faster-Whisper. Chỉ `INPUT_FINAL` được gửi tiếp. Không lưu audio.

## 3. Pygame window

```text
runtime: Pygame
mode: windowed
orientation: portrait
ratio: width:height = 6:9
FPS: 30
frame format: PNG sequence
pixel format: RGBA
alpha: enabled
scaling: preserve_aspect_ratio
```

Kích thước pixel cụ thể chưa khóa; implementation giữ đúng tỷ lệ 6:9.

Window chỉ hiển thị animation. Response text do backend speech xử lý.

## 4. Animation contract

```text
UI/assets/animations/
├── normal/
├── listening/
├── speech/
├── thinking/
├── processing/
├── executing/
├── success/
└── error/
```

- Mỗi action có thư mục riêng.
- Frame đặt tên `frame_0001.png`, `frame_0002.png`, ...
- Pygame preload frame khi khởi động/reload.
- Thiếu action/frame/config dùng fallback UI màu đỏ.
- Worker chỉ gửi action/state; Pygame render trên UI thread.

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

## 5. State và request policy

- `success` hiển thị 3 giây rồi về `normal`.
- `error` hiển thị 5 giây rồi về `normal`.
- User có thể cancel ở `listening`, `thinking`, `processing`.
- Pipeline xử lý tuần tự.
- Không chạy nhiều skill song song.
- Queue tối đa 1 request đang chờ.
- Không bỏ qua request cũ để chạy request mới.
- `speech` có thể chuyển sang `listening` khi phát hiện khoảng ngắt.

## 6. Model boundary

- Lỗi model → `ERROR`.
- Model output được validate bằng Pydantic runtime validator.
- JSON Schema là tài liệu contract.
- `request_id` và `model_version` không bắt buộc trong metadata model output.
- Runtime vẫn giữ `request_id`.
- Runtime lấy `model_version` từ model adapter.

## 7. Logging và fallback

- Logging mặc định chỉ ghi structured status.
- Không ghi transcript mặc định.
- Không lưu audio.
- Animation assets thiếu vẫn khởi động được.
- Fallback UI là màu đỏ.

## 8. Definition of Done

- [x] Input event contract.
- [x] Faster-Whisper settings.
- [x] Pygame window settings.
- [x] Animation action folders.
- [x] Frame format and scaling.
- [x] State timeout/cancel policy.
- [x] Sequential request policy.
- [x] Model error policy.
- [x] Validator policy.
- [x] Logging/audio policy.
- [x] Missing-assets fallback policy.
