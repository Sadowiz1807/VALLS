# ApplicationControl

Dùng để mở hoặc đóng ứng dụng đã được whitelist trong `Runtime/Registry/applications.json`.

- Workflow IDs: `application.open`, `application.close`
- Keywords tham khảo: mở, bật, đóng, thoát ứng dụng
- Resources: `application.catalog.resolve`, `application.control.open`, `application.control.close`
- Risk: OPEN=LOW, CLOSE=MEDIUM

Không được dùng app ngoài registry, nhận raw executable/shell, đóng entity mơ hồ hoặc báo thành công trước process evidence.
