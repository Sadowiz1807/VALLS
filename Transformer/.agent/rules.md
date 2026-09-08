# Coding Agent Rules

## Soul of truth
- file "SLU_Model_Architecture.md" là tài liệu chuẩn về kiến trúc model, trước khi build model phải đọc qua tài liệu, **tuyệt đối** không được thay đổi tài liệu nếu chưa có sự cho phép của user 

## Core
- Ưu tiên: **Safety > Contract > Architecture > Correctness > Regression Safety > Maintainability > Performance**.
- Chỉ sửa đúng scope, theo nguyên tắc **minimal sufficient change**. Không tự refactor, thêm feature, dependency, capability hoặc abstraction ngoài yêu cầu.
- Trước khi code phải đọc module liên quan, contract, docs và tests. Không đoán behavior.
- Không tự đổi architecture, ontology, public API, schema, enum hoặc error code nếu task không yêu cầu.

## Architecture
- **SLU Model:** audio → ACT/Goal/Parameters/Confidence → Semantic Frame.
- **Harness:** validate → registry/policy → resolve → execute → Grounded Result.
- **Response Module:** Grounded Result → response text; V0 dùng template.
- **TTS:** response text → audio.
- Model không execute; Harness không tự đoán intent; Response Module không tự claim success.

## Contract & Safety
- Nếu contract đổi, cập nhật đồng bộ: `contract → implementation → validator → tests → docs`.
- Model output luôn là **untrusted input** cho tới khi Harness validate.
- Fail closed với `INVALID`, `UNSUPPORTED`, `AMBIGUOUS`, `LOW_CONFIDENCE`, missing parameter hoặc resource không resolve được.
- Không tự sửa/đoán semantic frame rồi execute.
- Không hard-code intent bằng keyword; semantic decision phải thuộc SLU Model.
- Không hard-code secret, log credential/token, commit `.env`, bypass registry/allowlist/policy.

## SLU Rules
- ACT có thể fixed; Goal phải theo schema/capability resolution.
- Parameter xử lý theo type: `ENUM`, `NUMBER`, `ENTITY`, `STATE_REFERENCE`, `FREE_TEXT`, `BOOLEAN`.
- Lexical/CTC branch chỉ hỗ trợ lexical extraction/debug, không phải semantic authority.
- Confidence/OOD là first-class output; không ép model chọn Goal khi uncertainty cao.
- V0 không thêm response decoder hoặc response-generation loss vào SLU Model.

## Response Module
- V0 dùng **template-based Response Module**.
- Chỉ sinh success response sau khi Harness có Grounded Result xác nhận execution.
- Response Module sinh text; TTS là component riêng.

## Code Quality
- Function/class phải có một responsibility rõ ràng; tránh duplicated logic, magic values, hidden side effects, deep nesting và global mutable state.
- Dùng type hints ở các boundary quan trọng.
- Config, threshold, model version, path, timeout, registry value phải lấy từ config/registry nếu đã có source of truth.
- Không thêm dependency nếu project đã có cách giải quyết tương đương.
- Comment giải thích **why**, không lặp lại code.
- Không dùng `except Exception: pass`; không biến failure thành `SUCCESS`.

## Testing
- Bug fix nên có regression test; feature mới phải cover happy path, invalid input, boundary và failure.
- Không sửa test chỉ để pass; xác định implementation hay expectation mới là bên sai.
- Trước khi hoàn thành: kiểm tra syntax/import, relevant tests, regression phù hợp, contract consistency và Git diff.
- Không claim `fixed`, `pass`, `complete`, `production-ready` nếu chưa có evidence.

## Git & Scope
- Không tự commit, push, merge, rebase, reset, force-push hoặc xóa branch nếu user chưa yêu cầu.
- Không overwrite/revert thay đổi có sẵn của user ngoài scope.
- Không commit `__pycache__`, `*.pyc`, logs, debug dumps, local checkpoints hoặc temporary artifacts.
- Nếu phát hiện vấn đề ngoài scope, ghi nhận nhưng không tự sửa trừ khi nó block task hiện tại.

## Before Editing
Agent phải trả lời được:
1. Module nào sở hữu responsibility này?
2. Contract nào điều khiển behavior?
3. Thay đổi nhỏ nhất nào giải quyết đúng vấn đề?
4. Test/evidence nào chứng minh thay đổi đúng?

Nếu chưa trả lời được, phải inspect project thêm trước khi sửa code.
