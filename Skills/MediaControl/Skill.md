# MediaControl

Dùng để phát và điều khiển playback qua provider được registry cho phép. MVP dùng Spotify.

- Workflow IDs: `media.play`, `media.transport`
- Keywords tham khảo: phát, pause, resume, dừng, bài tiếp, bài trước
- Resources: `media.playback.play`, `media.playback.pause`, `media.playback.resume`, `media.playback.stop`, `media.playback.next`, `media.playback.previous`
- Risk: LOW

Không được dùng provider ngoài registry, tải nội dung, lưu credential hoặc báo thành công khi provider chưa xác nhận.
