"""
Runtime Engine & Agentic Execution Harness.
Bao gồm:
1. Application & Skill Registry
2. Dialogue Working Memory (Context & State Management)
3. Step-by-step Execution Loop (Agentic harness)
4. State Machine (Confirmation / Clarification)
"""
from __future__ import annotations
import difflib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

from Runtime.Providers.Builtin import build_builtin_providers
from Runtime.Contracts.Validate import load_manifest, validate_registry
from Runtime.Policy.Confirmation import ConfirmationStore
from Runtime.Resources.Dispatcher import ResourceDispatcher
from Runtime.Skills.Executor import SkillExecutor

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"\s+", " ", text)
    return text

@dataclass
class PendingFrame:
    request_id: str
    skill_id: str
    resource_id: str
    arguments: Dict[str, Any]
    risk: str
    created_at: datetime
    expires_at: datetime
    description: str

@dataclass
class DialogueMemory:
    """Lưu trữ ngữ cảnh làm việc và trạng thái thời gian thực."""
    history: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    max_history_turns: int = 5

    def add_turn(self, user_text: str, assistant_response: str, frame: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> None:
        turn = {
            "turn_id": len(self.history),
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "assistant_response": assistant_response,
            "act": frame.get("act"),
            "goal": frame.get("goal"),
            "parameters": frame.get("parameters"),
            "execution_result": result
        }
        self.history.append(turn)
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_context_for_model(self) -> List[Dict[str, Any]]:
        # Chuyển đổi sang format schema mà MultiTaskDataset hiểu được
        return [{"role": "user" if i % 2 == 0 else "assistant", "text": t.get("user_text") or t.get("assistant_response")} for i, t in enumerate(self.history[-3:])]


class ApplicationRegistry:
    def __init__(self, config_path: Optional[Path] = None):
        self.apps: List[Dict[str, Any]] = []
        if config_path and config_path.exists():
            self.apps = json.loads(config_path.read_text(encoding="utf-8"))

    @staticmethod
    def _local_executable(app: Dict[str, Any]) -> Optional[str]:
        local = app.get("local")
        return (local or {}).get("executable") or app.get("executable")

    @staticmethod
    def _web_url(app: Dict[str, Any]) -> Optional[str]:
        web = app.get("web") or {}
        url = web.get("url")
        return url if isinstance(url, str) and url.startswith(("http://", "https://")) else None

    def resolve(self, query: str) -> Tuple[Optional[Dict[str, Any]], float]:
        q = normalize_text(query)
        if not q:
            return None, 0.0

        for app in self.apps:
            if not app.get("enabled", True):
                continue
            entity_id = app.get("app_id") or app.get("browser_id", "")
            if q == normalize_text(entity_id) or q == normalize_text(app.get("name", entity_id)):
                return app, 1.0

        for app in self.apps:
            if not app.get("enabled", True):
                continue
            for alias in app.get("aliases", []):
                if q == normalize_text(alias):
                    return app, 1.0

        best_app = None
        best_score = 0.0
        for app in self.apps:
            if not app.get("enabled", True):
                continue
            all_names = [app["app_id"], app["name"]] + app.get("aliases", [])
            for name in all_names:
                norm_name = normalize_text(name)
                if re.search(rf"(?<!\w){re.escape(norm_name)}(?!\w)", q):
                    score = 0.95
                    if score > best_score:
                        best_score = score
                        best_app = app
                elif q in norm_name:
                    score = len(q) / len(norm_name) * 0.95
                    if score > best_score:
                        best_score = score
                        best_app = app
                ratio = difflib.SequenceMatcher(None, q, norm_name).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_app = app

        if best_score >= 0.7:
            return best_app, best_score
        return None, best_score


class AgentHarness:
    """Agentic Execution Harness quản lý vòng lặp xử lý, memory và tool dispatching."""
    def __init__(self, registry_dir: Path, execute: bool = False, runner: Any = None,
                 web_opener: Any = None, skill_executor: SkillExecutor | None = None):
        self.registry_dir = registry_dir
        self.app_registry = ApplicationRegistry(registry_dir / "applications.json")
        self.browser_registry = ApplicationRegistry(registry_dir / "browsers.json")
        self.skills_path = registry_dir / "skills.json"
        self.skills: List[Dict[str, Any]] = []
        if self.skills_path.exists():
            self.skills = load_manifest(self.skills_path)
        self.pending_frame: Optional[PendingFrame] = None
        self.memory = DialogueMemory()
        self.execute = execute
        self.runner = runner or subprocess.Popen
        self.confirmations = ConfirmationStore()
        manifest_data = json.loads(self.skills_path.read_text(encoding="utf-8")) if self.skills_path.is_file() else None
        if skill_executor is None and isinstance(manifest_data, dict):
            ontology = registry_dir.parent / "Model" / "VSAD" / "0.0.4" / "config.json"
            validate_registry(registry_dir, ontology)
        providers = build_builtin_providers(registry_dir, self.runner, web_opener) if skill_executor is None else None
        self.skill_executor = skill_executor or SkillExecutor(
            self.skills_path,
            ResourceDispatcher(providers, registry_dir / "resources.json", confirmation_store=self.confirmations),
            confirmation_store=self.confirmations,
        )

    def step(self, raw_input: str, vsad_model: Any) -> Dict[str, Any]:
        """Thực hiện một bước agentic: nạp context -> infer model -> dispatch tool -> update state & memory."""
        ctx = self.memory.get_context_for_model()
        st = self.memory.state
        
        # 1. Model Inference có kèm working context & system state
        frame = vsad_model.infer(raw_input, context=ctx, state=st)
        
        # 2. Dispatching & Tool Execution
        res = self._dispatch_turn(raw_input, frame)
        
        # 3. Cập nhật Working Memory
        self.memory.add_turn(raw_input, res.get("response", ""), frame, res.get("result"))
        return res

    def _dispatch_turn(self, raw_input: str, model_frame: Dict[str, Any]) -> Dict[str, Any]:
        act = model_frame.get("act")
        goal = model_frame.get("goal")
        params = model_frame.get("parameters", {})
        now = datetime.now()
        norm_in = normalize_text(raw_input)

        # Confirmation is a security boundary: exact phrases only, cancellation wins.
        if self.pending_frame:
            cancel = {"huy", "thoi", "khong", "cancel", "no", "dung"}
            confirm = {"dong y", "xac nhan", "chac chan", "ok", "yes", "tiep tuc"}
            if norm_in in cancel:
                act = "CANCEL"
            elif norm_in in confirm:
                act = "CONFIRM"

        if act == "CONFIRM":
            if not self.pending_frame:
                return {"status": "REJECTED", "reason": "NO_PENDING_ACTION", "response": "Không có yêu cầu nào đang chờ xác nhận."}
            if now > self.pending_frame.expires_at:
                self.pending_frame = None
                return {"status": "EXPIRED", "reason": "CONFIRMATION_EXPIRED", "response": "Yêu cầu trước đó đã hết hạn xác nhận."}
            target_frame = self.pending_frame
            self.pending_frame = None
            grant = self.confirmations.issue(
                target_frame.request_id, target_frame.skill_id, target_frame.resource_id,
                target_frame.arguments, target_frame.risk,
            )
            exec_result = self._execute_skill(target_frame.skill_id, target_frame.arguments, confirmation_grant=grant)
            return {
                "status": "EXECUTED" if exec_result["ok"] else "ERROR",
                "skill_id": target_frame.skill_id, "result": exec_result,
                "response": f"Đã xác nhận và thực thi: {target_frame.description}." if exec_result["ok"] else f"Thực thi thất bại: {exec_result.get('error')}"
            }

        if act == "CANCEL":
            if self.pending_frame:
                desc = self.pending_frame.description
                self.pending_frame = None
                return {"status": "CANCELLED", "response": f"Đã hủy yêu cầu: {desc}."}
            return {"status": "CANCELLED", "response": "Đã hủy thao tác."}

        if act == "RESPOND":
            intent = params.get("intent", "GREETING")
            result = self._execute_skill("conversation.social", {"intent": intent})
            responses = {
                "GREETING": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?",
                "THANKS": "Không có chi, rất vui được hỗ trợ bạn!",
                "GOODBYE": "Tạm biệt bạn, hẹn gặp lại!",
                "ACKNOWLEDGEMENT": "Tôi đã hiểu."
            }
            return {"status": "RESPONDED" if result.get("ok") else "ERROR", "result": result,
                    "response": responses.get(intent, "Tôi đã ghi nhận.") if result.get("ok") else f"Không thể phản hồi: {result.get('error')}."}

        if act in ("UNSUPPORTED", None):
            return {"status": "UNSUPPORTED", "response": "Xin lỗi, tôi chưa hỗ trợ yêu cầu này."}

        if act == "ASK_CLARIFICATION":
            return {"status": "CLARIFICATION_NEEDED", "response": model_frame.get("response") or "Bạn có thể nói rõ hơn yêu cầu được không?"}

        if act == "EXECUTE":
            return self._handle_execute(raw_input, goal, params)

        return {"status": "INVALID_FRAME", "response": "Không thể xử lý định dạng yêu cầu."}

    def _handle_execute(self, raw_input: str, goal: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
        from Runtime.Contracts.Semantics import ROUTES

        params = dict(params)
        if isinstance(params.get("action"), str):
            params["action"] = params["action"].upper()
        action = params.get("action")
        skill_id = ROUTES.get((goal, action), ROUTES.get((goal, None)))
        if skill_id is None:
            return {
                "status": "ERROR", "error": "SKILL_NOT_AVAILABLE",
                "response": f"Runtime chưa có skill khả dụng cho {goal}/{action or '-'}."
            }

        arguments: Dict[str, Any]
        if skill_id.startswith("application."):
            application = params.get("application", "")
            if isinstance(application, dict):
                application = application.get("value", "")
            arguments = {"application": application or raw_input}
        elif skill_id == "web.open":
            target = params.get("target", "")
            if isinstance(target, dict):
                target = target.get("value", "")
            arguments = {"target": target or raw_input, "browser": params.get("browser")}
        elif skill_id == "media.play":
            query = params.get("query", "")
            if isinstance(query, dict):
                query = query.get("value", "")
            arguments = {"query": query, "platform": params.get("platform", "DEFAULT")}
        elif skill_id == "media.transport":
            arguments = {"action": action, "platform": params.get("platform", "DEFAULT")}
        else:
            arguments = params

        result = self._execute_skill(skill_id, arguments)
        if result.get("error") == "CONFIRMATION_REQUIRED":
            risk = result["effective_risk"]
            description = f"thực hiện {skill_id}"
            self.pending_frame = PendingFrame(
                request_id=result["request_id"], skill_id=skill_id,
                resource_id=result["resource_id"], arguments=arguments, risk=risk,
                created_at=datetime.now(), expires_at=datetime.now() + timedelta(seconds=60),
                description=description,
            )
            return {"status": "AWAITING_CONFIRMATION", "skill_id": skill_id, "risk": risk,
                    "response": f"Bạn có chắc chắn muốn {description} không?"}
        status = "EXECUTED" if result.get("ok") else ("ROUTED" if result.get("error") == "EXECUTION_DISABLED" else "ERROR")
        return {"status": status, "skill_id": skill_id, "result": result,
                "response": "Đã thực hiện yêu cầu." if result.get("ok") else f"Không thể thực hiện: {result.get('error')}."}

    def _execute_skill(self, skill_id: str, args: Dict[str, Any], confirmation_grant: Any = None) -> Dict[str, Any]:
        if confirmation_grant is None:
            return self.skill_executor.execute(skill_id, args, self.execute)
        return self.skill_executor.execute(skill_id, args, self.execute, confirmation_grant=confirmation_grant)


# Alias for backward compatibility
RuntimeEngine = AgentHarness
