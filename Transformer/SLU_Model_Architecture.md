# VALLS SLU Model V0 — Architecture Specification

> **Status:** Draft architecture baseline  
> **Model family:** Voice-native Spoken Language Understanding (SLU)  
> **Version:** V0  
> **Primary input:** Raw user voice/audio  
> **Primary output:** Structured semantic execution frame  
> **Response strategy V0:** Template-based via **Response Module**  
> **Scope:** Model architecture, functional blocks, layer responsibilities, interfaces, and execution flow.  
> **Out of scope:** Detailed training curriculum, dataset implementation, optimizer configuration, deployment packaging.

---

# 1. Mục tiêu kiến trúc

VALLS SLU Model V0 được thiết kế lại hoàn toàn theo hướng **voice-native semantic understanding**.

Model không còn phụ thuộc vào pipeline:

```text
Voice
  ↓
ASR
  ↓
Text
  ↓
NLU / VSAD
```

Thay vào đó, pipeline chính trở thành:

```text
Voice
  ↓
Speech Representation
  ↓
Semantic Representation
  ↓
Capability Resolution
  ↓
Parameter Extraction
  ↓
Semantic Execution Frame
  ↓
Harness
```

Mục tiêu cốt lõi của model:

1. Nhận audio trực tiếp từ user.
2. Học representation âm thanh có khả năng giữ thông tin ngôn ngữ cần thiết.
3. Chuyển speech representation thành semantic representation.
4. Xác định loại hành động tổng quát của user.
5. Xác định capability/goal phù hợp dựa trên schema thay vì fixed classifier.
6. Trích xuất parameter cần thiết cho execution.
7. Đánh giá confidence và Out-of-Distribution (OOD).
8. Trả về một execution frame có cấu trúc cho harness.
9. Không trực tiếp thực thi skill.
10. Không trực tiếp sinh response tự do trong semantic core.

---

# 2. Nguyên tắc thiết kế

## 2.1 Voice là modality gốc

Audio không được xem như một bước trung gian để tạo text trước khi model hiểu semantic.

```text
Audio
  ↓
Semantic understanding
```

Transcript có thể tồn tại như một nhánh phụ để:

- hỗ trợ training;
- debug;
- lexical extraction;
- free-text parameter extraction;
- kiểm tra lỗi model.

Nhưng transcript **không phải dependency bắt buộc của semantic inference**.

---

## 2.2 Semantic model và Response Module tách biệt

Model SLU chịu trách nhiệm:

```text
User voice
   ↓
User intent / capability / parameters
```

Response Module chịu trách nhiệm:

```text
Grounded runtime result
   ↓
Response text
```

Hai phần không dùng chung responsibility.

---

## 2.3 Goal không được đóng cứng thành fixed class head

Không thiết kế:

```text
Linear(d_model → N_goals)
```

với:

```text
APPLICATION_CONTROL
MEDIA_CONTROL
WEB_SEARCH
...
```

là một danh sách class cố định.

Thay vào đó:

```text
Semantic Query
     ↕
Capability Schema
```

Goal được resolve bằng semantic retrieval / matching.

Mục tiêu:

```text
thêm capability mới
≠
thay output dimension toàn model
```

---

## 2.4 Parameter phải có type

Parameter không được giải quyết bằng một hệ thống classification chung.

Các loại parameter chính:

```text
ENUM
NUMBER
ENTITY
STATE_REFERENCE
FREE_TEXT
BOOLEAN
```

Mỗi type có extractor/normalizer phù hợp.

---

## 2.5 Model không có execution authority

Model chỉ trả semantic intent.

Harness mới có quyền:

- validate;
- resolve registry;
- kiểm tra policy;
- kiểm tra resource;
- dispatch skill;
- dispatch resource;
- xác nhận kết quả thực thi.

---

# 3. Kiến trúc tổng thể

```text
                         USER AUDIO
                             │
                             ▼
                 ┌─────────────────────┐
                 │  Input Audio Layer  │
                 │ VAD / segmentation  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Speech Encoder    │
                 │ Acoustic/Linguistic │
                 │ Representation      │
                 └──────────┬──────────┘
                            │
                         H_speech
                            │
             ┌──────────────┴───────────────┐
             │                              │
             ▼                              ▼
┌─────────────────────────┐     ┌────────────────────────┐
│ Semantic Resampler      │     │ Lexical Branch         │
│ / Speech-Semantic       │     │ Auxiliary / Conditional│
│ Adapter                 │     │ CTC / Token Recovery   │
└────────────┬────────────┘     └───────────┬────────────┘
             │                              │
      Z_semantic                         lexical info
             │                              │
             ▼                              │
┌─────────────────────────┐                 │
│ Semantic Core           │                 │
│ Transformer Encoder     │                 │
└────────────┬────────────┘                 │
             │                              │
     ┌───────┼──────────┬───────────┐       │
     │       │          │           │       │
     ▼       ▼          ▼           ▼       │
   ACT    Goal Query   OOD      Operation    │
   Head    Generator   Head      Decoder     │
             │                              │
             ▼                              │
┌─────────────────────────┐                 │
│ Capability Schema       │                 │
│ Retriever / Matcher     │                 │
└────────────┬────────────┘                 │
             │                              │
       Selected Schema                      │
             │                              │
             └──────────────┬───────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Typed Parameter     │
                 │ Extractor           │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Confidence / Risk   │
                 │ Calibration         │
                 └──────────┬──────────┘
                            │
                            ▼
                 SEMANTIC EXECUTION FRAME
                            │
                            ▼
                         HARNESS
                            │
                            ▼
                    SKILL / RESOURCE
                            │
                            ▼
                      GROUNDED RESULT
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Response Module    │
                 │ Template-based V0   │
                 └──────────┬──────────┘
                            │
                            ▼
                      RESPONSE TEXT
                            │
                            ▼
                           TTS
                            │
                            ▼
                    ASSISTANT VOICE
```

---

# 4. Input Audio Layer

## 4.1 Vai trò

Input Audio Layer chịu trách nhiệm chuẩn hóa audio trước khi đưa vào neural model.

Nó không thực hiện semantic reasoning.

Các nhiệm vụ chính:

- nhận microphone stream;
- Voice Activity Detection (VAD);
- xác định utterance boundary;
- loại bỏ no-speech segment;
- resample về sample rate chuẩn;
- chuyển stereo → mono nếu cần;
- normalize biên độ;
- chunking hoặc buffering nếu sử dụng streaming.

---

## 4.2 Input contract dự kiến

```json
{
  "request_id": "uuid",
  "audio": "<waveform>",
  "sample_rate": 16000,
  "channels": 1,
  "source": "microphone"
}
```

---

## 4.3 Output

```text
Normalized waveform
```

Ví dụ:

```text
shape = [num_samples]
sample_rate = 16 kHz
```

---

# 5. Speech Encoder

## 5.1 Vai trò

Speech Encoder chuyển raw/processed audio thành chuỗi hidden representation giàu thông tin acoustic và linguistic.

```text
waveform
   ↓
Speech Encoder
   ↓
H_speech
```

Trong đó:

```text
H_speech ∈ R^(T × D_speech)
```

---

## 5.2 Kiến thức cần học

Speech Encoder tập trung vào:

- phonetic structure;
- phát âm tiếng Việt;
- accent;
- speaker variation;
- speech speed;
- coarticulation;
- acoustic noise;
- Vietnamese-English code switching;
- lexical cues cần thiết cho semantic understanding.

Speech Encoder **không phải nơi lưu ontology VALLS**.

---

## 5.3 Kiến trúc đề xuất

V0 ưu tiên một speech backbone theo hướng:

```text
Audio
  ↓
Log-Mel / acoustic feature frontend
  ↓
Convolutional subsampling
  ↓
Conformer / pretrained speech blocks
```

Ví dụ baseline:

```text
Input:
16 kHz mono

Feature:
80-dim Log-Mel

Subsampling:
Conv ×4

Speech dimension:
512

Speech blocks:
8–12 Conformer layers
```

Các con số trên là baseline kiến trúc, không phải hyperparameter cố định.

---

## 5.4 Boundary

Speech Encoder không output:

```text
goal
intent
skill
execution result
```

Nó chỉ output representation.

---

# 6. Semantic Resampler / Speech-Semantic Adapter

## 6.1 Vấn đề

Speech sequence thường rất dài.

Ví dụ:

```text
4 giây audio
      ↓
hàng trăm speech frames
```

Semantic reasoning không cần xử lý toàn bộ temporal detail đó.

---

## 6.2 Vai trò

Semantic Resampler chuyển:

```text
H_speech
```

thành một representation ngắn hơn:

```text
Z_semantic
```

Ví dụ:

```text
200 speech frames
       ↓
32 semantic latent tokens
```

---

## 6.3 Mục tiêu

Resampler phải:

- giảm sequence length;
- giữ linguistic information;
- giữ entity information;
- giữ number information;
- giữ semantic cues;
- bỏ bớt acoustic redundancy.

---

## 6.4 Kiến trúc dự kiến

Có thể sử dụng:

```text
Learnable latent queries
        +
Cross Attention
```

Ví dụ:

```text
32 learnable latent tokens
          │
          ▼
Cross Attention
          │
          ▼
H_speech
```

Output:

```text
Z_semantic ∈ R^(32 × 512)
```

---

# 7. Semantic Core

## 7.1 Vai trò

Semantic Core là phần chính chịu trách nhiệm hiểu:

> User muốn hệ thống làm gì?

Nó không còn phải tự học toàn bộ acoustic structure.

Input:

```text
Z_semantic
```

Output:

```text
contextual semantic representation
```

---

## 7.2 Kiến trúc đề xuất

Baseline V0:

```text
Transformer Encoder
```

Ví dụ:

```text
layers: 4–6
d_model: 512
attention heads: 8
FFN: 2048
```

---

## 7.3 Semantic Core học

- semantic similarity;
- command structure;
- task distinction;
- operation relationships;
- conversational command form;
- Vietnamese paraphrase;
- Vietnamese-English mixed command;
- compound commands;
- context-sensitive semantics.

---

## 7.4 Semantic Core không chịu trách nhiệm

- execution;
- app availability;
- URL validation;
- operating system state;
- response text generation;
- capability whitelist authority.

---

# 8. ACT Head

## 8.1 Vai trò

ACT biểu diễn hành vi cấp cao của request.

ACT có tính ổn định hơn Goal nên có thể sử dụng fixed classifier.

Baseline ACT ontology:

```text
EXECUTE
RESPOND
ASK_CLARIFICATION
CONFIRM
CANCEL
UNSUPPORTED
```

---

## 8.2 Input

Semantic representation từ Semantic Core.

---

## 8.3 Output

Ví dụ:

```json
{
  "act": "EXECUTE",
  "confidence": 0.97
}
```

---

## 8.4 Tại sao ACT có thể fixed

ACT là abstraction cấp cao.

Số lượng ACT không tăng theo số lượng skill.

Ví dụ thêm:

```text
FILE_CONTROL
EMAIL
CALENDAR
```

không yêu cầu thêm ACT mới.

---

# 9. Goal Query Generator

## 9.1 Vai trò

Goal Query Generator tạo semantic query vector đại diện cho capability mà user đang yêu cầu.

```text
Semantic Core
    ↓
Goal Query
```

Ví dụ:

```text
q_goal ∈ R^512
```

---

## 9.2 Không output class ID

Không dùng:

```text
goal = class_3
```

Mà query được so sánh với capability schemas.

---

# 10. Capability Schema Retriever

## 10.1 Vai trò

Capability Schema Retriever tìm capability phù hợp nhất với semantic query.

```text
Goal Query
     ↕
Capability Schema Embeddings
```

---

## 10.2 Capability schema

Ví dụ:

```yaml
goal: APPLICATION_CONTROL

description:
  Open, close, focus, or otherwise control supported applications.

actions:
  - OPEN
  - CLOSE
  - FOCUS

parameters:
  application:
    type: ENTITY
```

Schema khác:

```yaml
goal: WEB_SEARCH

description:
  Search information on the web using a supported search provider.

actions:
  - SEARCH

parameters:
  query:
    type: FREE_TEXT

  engine:
    type: ENTITY
    optional: true
```

---

## 10.3 Retrieval

Ví dụ user nói:

```text
"đóng spotify"
```

Similarity:

```text
APPLICATION_CONTROL   0.95
MEDIA_CONTROL         0.24
WEB_OPEN              0.08
WEB_SEARCH            0.03
```

Model chọn:

```text
APPLICATION_CONTROL
```

---

## 10.4 Lợi ích

Thêm:

```text
FILE_CONTROL
```

không yêu cầu thay:

```text
Linear(512 → N)
```

Chỉ cần:

- schema mới;
- embedding/schema encoder;
- examples/adaptation khi cần.

---

# 11. Operation Decoder

## 11.1 Mục tiêu

V0 không nên khóa:

```text
1 utterance = 1 operation
```

User có thể nói:

```text
"mở chrome rồi tìm github"
```

Semantic output nên hỗ trợ:

```json
{
  "operations": [
    {
      "goal": "APPLICATION_CONTROL",
      "action": "OPEN",
      "parameters": {
        "application": "chrome"
      }
    },
    {
      "goal": "WEB_SEARCH",
      "action": "SEARCH",
      "parameters": {
        "query": "github"
      }
    }
  ]
}
```

---

## 11.2 Vai trò

Operation Decoder xác định:

- số operation;
- thứ tự operation;
- semantic boundary giữa operations;
- capability schema tương ứng cho từng operation.

---

## 11.3 Giới hạn V0

V0 có thể giới hạn:

```text
max_operations = 2 hoặc 3
```

để giảm độ phức tạp ban đầu.

---

# 12. Lexical Branch

## 12.1 Tại sao vẫn cần lexical information

Semantic model có thể hiểu:

```text
goal = WEB_SEARCH
```

nhưng harness vẫn cần query cụ thể:

```text
"cách cài pytorch cuda"
```

Tương tự:

```text
filename
search query
dynamic entity
URL-like content
user-provided text
```

cần giữ gần nguyên văn.

---

## 12.2 Vai trò

Lexical Branch phục hồi linguistic token/span từ speech representation khi cần.

Nó không phải semantic authority.

---

## 12.3 Baseline V0

Có thể dùng:

```text
Speech Encoder
      ↓
CTC Head
      ↓
tokens
```

CTC branch có thể hoạt động:

- training auxiliary;
- runtime conditional;
- debug.

---

## 12.4 Không bắt buộc full transcript

Model không nhất thiết phải luôn decode toàn bộ utterance.

Ví dụ:

```text
"tìm cách cài pytorch cuda"
```

Semantic branch:

```text
WEB_SEARCH
```

Lexical branch:

```text
query span = "cách cài pytorch cuda"
```

---

# 13. Typed Parameter Extractor

## 13.1 Vai trò

Parameter Extractor nhận:

```text
Selected Capability Schema
+
Semantic Representation
+
Optional Lexical Information
```

và sinh parameters theo schema.

---

# 14. Parameter Type: ENUM

Ví dụ:

```text
OPEN
CLOSE
FOCUS
PLAY
PAUSE
NEXT
PREVIOUS
```

ENUM có candidate set xác định.

Ví dụ:

```text
"đóng chrome"
```

Output:

```json
{
  "action": "CLOSE"
}
```

---

# 15. Parameter Type: NUMBER

Ví dụ:

```text
"đặt âm lượng bảy mươi ba phần trăm"
```

Output:

```json
{
  "volume": 73
}
```

NUMBER không nên dùng 101 class riêng cho:

```text
0
1
2
...
100
```

Mà sử dụng:

```text
spoken number region
      ↓
number normalization
      ↓
numeric value
```

---

# 16. Parameter Type: ENTITY

ENTITY dùng cho các giá trị như:

```text
spotify
chrome
cốc cốc
discord
vscode
```

Model có thể match:

```text
spoken entity representation
          ↕
registry alias embeddings
```

Ví dụ user phát âm:

```text
"spot ti fai"
```

Candidate:

```text
spotify   0.95
discord   0.11
chrome    0.02
```

---

# 17. Parameter Type: STATE_REFERENCE

Dùng cho:

```text
"nó"
"cái đó"
"tab vừa rồi"
"ứng dụng hiện tại"
```

Model không tự resolve final resource.

Nó output:

```json
{
  "application": {
    "type": "STATE_REFERENCE",
    "reference": "current_application"
  }
}
```

Harness chịu trách nhiệm resolve state.

---

# 18. Parameter Type: FREE_TEXT

Dùng cho:

```text
search query
filename
message
dynamic textual value
```

Ví dụ:

```text
"tìm cách cài pytorch cuda"
```

Output:

```json
{
  "query": "cách cài pytorch cuda"
}
```

FREE_TEXT sử dụng Lexical Branch nhiều hơn các type khác.

---

# 19. Parameter Type: BOOLEAN

Ví dụ capability tương lai:

```text
recursive = true
overwrite = false
```

BOOLEAN có thể xử lý bằng:

```text
binary semantic classification
```

theo capability schema.

---

# 20. Confidence / OOD Layer

## 20.1 Vai trò

Model phải có khả năng nói:

> Tôi không đủ chắc chắn để yêu cầu harness execute.

Không sử dụng:

```text
max softmax probability
```

như confidence duy nhất.

---

## 20.2 Các confidence component

Output nên có tối thiểu:

```text
act_confidence
goal_confidence
parameter_confidence
ood_score
overall_confidence
```

---

## 20.3 Ví dụ

```json
{
  "confidence": {
    "act": 0.98,
    "goal": 0.91,
    "parameters": 0.76,
    "ood": 0.04,
    "overall": 0.79
  }
}
```

---

## 20.4 Responsibility

Model:

```text
ước lượng uncertainty
```

Harness:

```text
quyết định policy
```

Ví dụ:

```text
high confidence
→ execute

medium confidence
→ clarification

low confidence / high OOD
→ reject / retry
```

---

# 21. Semantic Execution Frame

Đây là output chính thức của model.

Ví dụ đơn operation:

```json
{
  "model_version": "VALLS-SLU-V0",
  "request_id": "uuid",

  "act": "EXECUTE",

  "operations": [
    {
      "goal": "APPLICATION_CONTROL",
      "action": "OPEN",

      "parameters": {
        "application": {
          "type": "ENTITY",
          "value": "spotify"
        }
      },

      "confidence": 0.95
    }
  ],

  "confidence": {
    "act": 0.98,
    "goal": 0.95,
    "parameters": 0.92,
    "ood": 0.02,
    "overall": 0.94
  }
}
```

---

# 22. Compound Execution Frame

Ví dụ:

```text
"mở chrome rồi tìm github"
```

Output:

```json
{
  "model_version": "VALLS-SLU-V0",

  "act": "EXECUTE",

  "operations": [
    {
      "order": 1,
      "goal": "APPLICATION_CONTROL",
      "action": "OPEN",
      "parameters": {
        "application": {
          "type": "ENTITY",
          "value": "chrome"
        }
      }
    },

    {
      "order": 2,
      "goal": "WEB_SEARCH",
      "action": "SEARCH",
      "parameters": {
        "query": {
          "type": "FREE_TEXT",
          "value": "github"
        }
      }
    }
  ]
}
```

Harness chịu trách nhiệm orchestration.

---

# 23. Harness Boundary

Model output không đồng nghĩa với execution permission.

Pipeline:

```text
Semantic Execution Frame
         ↓
Contract Validation
         ↓
Registry Resolution
         ↓
Policy / Risk Gate
         ↓
Skill Resolution
         ↓
Resource Resolution
         ↓
Execution
         ↓
Grounded Result
```

---

# 24. Harness không được làm semantic guessing

Harness không nên:

```text
audio/text
  ↓
tự đoán intent
```

Semantic authority thuộc model.

Harness chỉ:

- validate;
- resolve;
- authorize;
- dispatch;
- verify.

---

# 25. Grounded Result

Sau execution, harness trả result có evidence.

Ví dụ:

```json
{
  "status": "SUCCESS",
  "goal": "APPLICATION_CONTROL",
  "action": "OPEN",
  "application": "spotify",
  "evidence": {
    "process_started": true
  }
}
```

Model semantic không được tự claim:

```text
"Đã mở Spotify."
```

trước khi harness có grounded result.

---

# 26. Response Module

## 26.1 Định nghĩa

**Response Module** là tên chung cho thành phần chịu trách nhiệm chuyển:

```text
Grounded Result
```

thành:

```text
Response Text
```

Response Module nằm **sau harness execution**.

---

## 26.2 V0 strategy

V0 sử dụng:

```text
Template-based Response Module
```

Không sử dụng neural decoder để sinh response tự do.

---

## 26.3 Ví dụ

Grounded result:

```json
{
  "status": "SUCCESS",
  "action": "OPEN",
  "application": "spotify"
}
```

Response Module:

```text
template:
"Đã mở {application}."
```

Output:

```text
"Đã mở Spotify."
```

---

## 26.4 Ví dụ khác

```text
APPLICATION_OPEN_SUCCESS
→ "Đã mở {application}."

APPLICATION_CLOSE_SUCCESS
→ "Đã đóng {application}."

VOLUME_SET_SUCCESS
→ "Đã đặt âm lượng thành {volume}%."

WEB_SEARCH_STARTED
→ "Đang tìm kiếm {query}."

APP_NOT_FOUND
→ "Không tìm thấy ứng dụng {application}."

REQUEST_UNSUPPORTED
→ "Yêu cầu này hiện chưa được hỗ trợ."
```

---

## 26.5 Lợi ích V0

Template-based response:

- deterministic;
- latency thấp;
- dễ test;
- không hallucination;
- không cần training;
- không cạnh tranh gradient với semantic model;
- chỉ phản hồi từ grounded runtime state.

---

# 27. Response Module không phải TTS

Phân biệt:

```text
Response Module
      ↓
Response Text
      ↓
TTS
      ↓
Assistant Voice
```

Response Module sinh **text**.

TTS chuyển **text → audio**.

---

# 28. TTS Boundary

TTS không thuộc SLU Model.

TTS chỉ nhận:

```text
response_text
language
voice_profile
```

và output:

```text
assistant audio
```

---

# 29. Context Handling

Model cần hỗ trợ runtime context nhưng context không được trộn trực tiếp vào acoustic representation một cách tùy ý.

Context có thể gồm:

```json
{
  "active_application": "chrome",
  "active_browser": "coccoc",
  "active_media": "spotify",
  "last_goal": "WEB_SEARCH"
}
```

Context được encode thành semantic/state representation.

---

# 30. Context Fusion

Có thể dùng:

```text
Speech Semantic Representation
            +
State / Context Embedding
            ↓
Semantic Core
```

hoặc:

```text
Semantic Core
    ↓
Cross Attention
    ↓
Context Encoder
```

V0 ưu tiên giữ state schema nhỏ và machine-readable.

---

# 31. State Reference Example

User nói:

```text
"đóng nó"
```

Model có thể output:

```json
{
  "goal": "APPLICATION_CONTROL",

  "action": "CLOSE",

  "parameters": {
    "application": {
      "type": "STATE_REFERENCE",
      "reference": "active_application"
    }
  }
}
```

Harness mới resolve:

```text
active_application = chrome
```

---

# 32. OOD / Unsupported

Model phải phân biệt:

```text
capability known
```

với:

```text
semantic request không thuộc capability registry
```

Ví dụ:

```text
"pha cho tôi một ly cà phê"
```

nếu không có capability tương ứng:

```json
{
  "act": "UNSUPPORTED",
  "operations": [],
  "confidence": {
    "ood": 0.94
  }
}
```

---

# 33. Clarification

Ví dụ:

```text
"mở nó"
```

nhưng không có state reference.

Output:

```json
{
  "act": "ASK_CLARIFICATION",

  "reason": "MISSING_TARGET"
}
```

Response Module sau đó có thể dùng template:

```text
"Bạn muốn mở ứng dụng nào?"
```

---

# 34. Các layer/module của model

Tóm tắt:

| Block | Input | Output | Vai trò |
|---|---|---|---|
| Input Audio Layer | microphone audio | normalized waveform | segmentation / normalization |
| Speech Encoder | waveform | speech features | acoustic + linguistic representation |
| Semantic Resampler | speech features | semantic latents | temporal compression |
| Semantic Core | semantic latents | semantic features | hiểu intent/meaning |
| ACT Head | semantic features | ACT | hành vi cấp cao |
| Goal Query Generator | semantic features | query vector | biểu diễn capability request |
| Schema Retriever | query + schemas | goal schema | dynamic goal selection |
| Operation Decoder | semantic features | operation sequence | compound command decomposition |
| Lexical Branch | speech features | tokens/spans | lexical recovery |
| Parameter Extractor | schema + semantic + lexical | typed params | extraction |
| OOD/Confidence | semantic state | confidence | uncertainty |
| Frame Builder | model outputs | execution frame | contract serialization |
| Response Module | grounded result | text | template response V0 |
| TTS | response text | audio | assistant speech |

---

# 35. Những gì thuộc neural model

```text
Speech Encoder
Semantic Resampler
Semantic Core
ACT Head
Goal Query Generator
Schema Retriever
Operation Decoder
Lexical Branch
Typed Parameter Extractor
OOD / Confidence
```

---

# 36. Những gì không thuộc neural model

```text
Microphone driver
VAD implementation
Runtime Registry
Skill execution
Resource execution
OS control
Browser control
Policy enforcement
Response Module
TTS
UI
```

---

# 37. Proposed V0 Model Block Diagram

```text
                            ┌─────────────────────┐
                            │      AUDIO          │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Speech Encoder     │
                            │ Conformer / SSL     │
                            └──────────┬──────────┘
                                       │
                       ┌───────────────┴───────────────┐
                       │                               │
                       ▼                               ▼
          ┌────────────────────────┐      ┌──────────────────────┐
          │ Semantic Resampler     │      │ Lexical CTC Branch   │
          │ T → fixed latents      │      │ optional/auxiliary   │
          └───────────┬────────────┘      └──────────┬───────────┘
                      │                              │
                      ▼                              │
          ┌────────────────────────┐                 │
          │ Semantic Core          │                 │
          │ Transformer            │                 │
          └───────────┬────────────┘                 │
                      │                              │
           ┌──────────┼───────────────┐              │
           │          │               │              │
           ▼          ▼               ▼              │
      ┌────────┐ ┌──────────┐ ┌──────────────┐      │
      │ACT Head│ │Goal Query│ │Operation Head│      │
      └───┬────┘ └─────┬────┘ └──────┬───────┘      │
          │            │             │              │
          │            ▼             │              │
          │   ┌─────────────────┐    │              │
          │   │ Schema Retriever │    │              │
          │   └────────┬────────┘    │              │
          │            │             │              │
          └────────────┴──────┬──────┴──────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Typed Parameter     │
                   │ Extractor           │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Confidence / OOD    │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Execution Frame     │
                   └──────────┬──────────┘
                              │
                            HARNESS
                              │
                              ▼
                       Grounded Result
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Response Module     │
                   │ Template-based V0   │
                   └──────────┬──────────┘
                              │
                              ▼
                         Response Text
                              │
                              ▼
                             TTS
```

---

# 38. V0 Responsibility Matrix

| Responsibility | SLU Model | Harness | Response Module |
|---|---:|---:|---:|
| Hiểu audio | ✅ | ❌ | ❌ |
| Xác định ACT | ✅ | ❌ | ❌ |
| Xác định Goal | ✅ | ❌ | ❌ |
| Trích parameter | ✅ | ❌ | ❌ |
| Ước lượng confidence | ✅ | ❌ | ❌ |
| Xác định app có tồn tại | ❌ | ✅ | ❌ |
| Kiểm tra resource | ❌ | ✅ | ❌ |
| Policy/safety execution | ❌ | ✅ | ❌ |
| Chạy skill | ❌ | ✅ | ❌ |
| Xác minh thành công | ❌ | ✅ | ❌ |
| Sinh response text | ❌ | ❌ | ✅ |
| Text → voice | ❌ | ❌ | ❌ |

TTS là component riêng.

---

# 39. Những thay đổi lớn so với VSAD cũ

```text
VSAD cũ
────────────────────────
Input = text
Tokenizer
Fixed Goal classifier
Per-goal parameter heads
Response decoder
Single-model multi-task coupling
```

Model mới:

```text
VALLS SLU V0
────────────────────────
Input = audio
Speech Encoder
Semantic Resampler
Schema-based Goal Retrieval
Typed Parameter Extraction
Optional lexical branch
Dedicated confidence/OOD
Response Module tách riêng
```

---

# 40. Architecture Decision Summary

## AD-01 — Audio-native

Runtime model nhận audio trực tiếp.

---

## AD-02 — Encoder-centric

Semantic understanding là nhiệm vụ chính.

Không xây decoder-only Speech LLM cho V0.

---

## AD-03 — Schema-driven capability

Goal không đóng cứng vào classifier dimension.

---

## AD-04 — Typed parameters

Parameter extraction phụ thuộc parameter type.

---

## AD-05 — Lexical branch là phụ trợ

Text/token recovery không phải semantic authority.

---

## AD-06 — Response Module tách khỏi SLU

V0 dùng template-based response.

---

## AD-07 — Harness giữ execution authority

Model không tự chạy capability.

---

## AD-08 — Grounded response

Response Module chỉ sinh success response sau grounded execution result.

---

## AD-09 — Confidence là first-class output

Model phải biểu diễn uncertainty/OOD.

---

## AD-10 — Multi-operation support

Semantic frame hỗ trợ nhiều operation trong cùng request.

---

# 41. V0 Scope đề xuất

Để giữ model đầu tiên khả thi, V0 nên giới hạn:

```text
Input:
single-user utterance
<= 8–10 seconds

Languages:
Vietnamese primary
English code-switching secondary

Operations:
1–2 operations / utterance

Parameter types:
ENUM
NUMBER
ENTITY
STATE_REFERENCE
FREE_TEXT

Response:
template-based Response Module

Execution:
sequential only
```

---

# 42. Không nằm trong V0

Các chức năng sau nên để V1+:

```text
long-form dialogue generation
speech-to-speech generative response
large decoder language model
arbitrary tool planning
unbounded operation sequence
fully streaming semantic decoding
speaker personalization
emotional voice understanding
end-to-end neural response generation
```

---

# 43. Baseline kiến trúc V0 đề xuất

```yaml
audio:
  sample_rate: 16000
  channels: 1

speech_encoder:
  frontend: log_mel
  feature_dim: 80
  architecture: conformer_or_pretrained_ssl
  d_model: 512
  layers: 8_to_12

semantic_resampler:
  type: learnable_query_cross_attention
  latent_tokens: 32
  d_model: 512

semantic_core:
  type: transformer_encoder
  layers: 4_to_6
  d_model: 512
  heads: 8
  ff_dim: 2048

act:
  type: fixed_classifier

goal:
  type: schema_retrieval
  embedding_dim: 512

lexical:
  type: ctc
  role: auxiliary_and_conditional

parameters:
  types:
    - ENUM
    - NUMBER
    - ENTITY
    - STATE_REFERENCE
    - FREE_TEXT
    - BOOLEAN

confidence:
  enabled: true

multi_operation:
  enabled: true
  max_operations_v0: 2

response:
  module_name: Response Module
  strategy: template_based
```

---

# 44. Final System Flow

```text
User Voice
   ↓
Input Audio Layer
   ↓
Speech Encoder
   ↓
Semantic Resampler
   ↓
Semantic Core
   ↓
ACT
+
Goal Schema Retrieval
+
Operation Decomposition
+
Typed Parameter Extraction
+
Confidence/OOD
   ↓
Semantic Execution Frame
   ↓
Harness Validation
   ↓
Registry / Policy
   ↓
Skill / Resource Execution
   ↓
Grounded Result
   ↓
Response Module
   ↓
Template Response Text
   ↓
TTS
   ↓
Assistant Voice
```

---

# 45. Kết luận

VALLS SLU Model V0 được định hướng thành một **voice-native structured semantic model**, không phải ASR pipeline được đóng gói lại và cũng không phải một Speech LLM tổng quát.

Ba abstraction chính của kiến trúc:

```text
Speech Encoder
= nghe

Semantic Core + Schema Retrieval
= hiểu

Harness
= hành động

Response Module
= phản hồi bằng text

TTS
= nói
```

Boundary quan trọng nhất:

```text
Audio
  ↓
SLU Model
  ↓
Semantic Execution Frame
```

là neural understanding boundary.

```text
Semantic Execution Frame
  ↓
Harness
```

là execution boundary.

```text
Grounded Result
  ↓
Response Module
```

là response-generation boundary.

Ở V0, **Response Module sử dụng template**, nhằm giữ response deterministic, grounded, dễ kiểm thử và không làm semantic model chịu thêm nhiệm vụ language generation.
