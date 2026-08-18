# Phase 1 — Voice/UI Contract

> **Trạng thái:** COMPLETE
> **Phạm vi:** Các quyết định và contract của Phase 1. Không mô tả implementation phase khác.

## 1. Voice contract

Pipeline của Phase 1:

```text
Microphone → Voice Process → Faster-Whisper → VOICE_FINAL.text
```

Event:

```text
VOICE_STARTED
VOICE_PARTIAL       optional, không gửi sang model
VOICE_FINAL         text cuối cùng
VOICE_CANCELLED
VOICE_ERROR
```

`VOICE_FINAL`:

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

Voice Process sở hữu microphone stream và Faster-Whisper. Chỉ `VOICE_FINAL` được gửi tiếp. Không lưu audio.

## 3. Pygame window

```text
runtime: Pygame
mode: windowed
orientation: portrait
ratio: width:height = 6:9
FPS: 30
render: màu nền + màu nhấn + decoration (pygame.draw), không dùng image frames
```

Kích thước pixel cụ thể chưa khóa; implementation giữ đúng tỷ lệ 6:9.

UI hiển thị tên action và subtitle (transcript/model metadata) bằng text.

## 4. Action visual contract

Mỗi action là một tổ hợp màu + decoration riêng, vẽ bằng pygame.draw, không cần file ảnh:

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

- Decoration tính theo thời gian, không đọc file mỗi frame.
- Worker chỉ gửi action/state; Pygame render trên UI thread.
- Không còn phụ thuộc animation assets; lỗi không lường trước fallback về `normal`.

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

- Adapter đặt trong Runtime (`Runtime/vsad_adapter.py`).
- Lỗi model → `ERROR`.
- Model output được validate bằng Pydantic runtime validator.
- JSON Schema là tài liệu contract.
- `request_id` và `model_version` không bắt buộc trong metadata model output.
- Runtime vẫn giữ `request_id`.
- Runtime lấy `model_version` từ model adapter.

## 7. Logging

- Logging mặc định chỉ ghi structured status.
- Không ghi transcript mặc định.
- Không lưu audio.

## 8. Definition of Done

- [x] Voice event contract.
- [x] Faster-Whisper settings.
- [x] Pygame window settings.
- [x] Action visual contract (màu + decoration).
- [x] State timeout/cancel policy.
- [x] Sequential request policy.
- [x] Model error policy.
- [x] Validator policy.
- [x] Logging/audio policy.
