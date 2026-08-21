# WebControl

Dùng để mở URL đã resolve bằng browser được whitelist trong `Runtime/Registry/browsers.json`.

- Workflow ID: `web.open`
- Keywords tham khảo: mở web, mở bằng trình duyệt
- Resources: `browser.navigation.open`
- Risk: LOW

Không được tự thêm browser, fallback khi browser explicit chưa hỗ trợ, nhận URL ngoài policy http/https hoặc báo thành công trước browser evidence.
