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
import shutil
import subprocess
import unicodedata
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

from Runtime.Resources import Application, Browser, Media, System
from Runtime.Skills import ApplicationControl, MediaControl, SystemControl, WebControl

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.strip().lower())
    text = re.sub(r"\s+", " ", text)
    return text

@dataclass
class PendingFrame:
    skill_id: str
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
    def __init__(self, registry_dir: Path, execute: bool = False, runner: Any = None, web_opener: Any = None):
        self.registry_dir = registry_dir
        self.app_registry = ApplicationRegistry(registry_dir / "applications.json")
        self.browser_registry = ApplicationRegistry(registry_dir / "browsers.json")
        self.skills_path = registry_dir / "skills.json"
        self.skills: List[Dict[str, Any]] = []
        if self.skills_path.exists():
            self.skills = json.loads(self.skills_path.read_text(encoding="utf-8"))
        self.pending_frame: Optional[PendingFrame] = None
        self.memory = DialogueMemory()
        self.execute = execute
        self.runner = runner or subprocess.Popen
        self.web_opener = web_opener or webbrowser.open
        application = Application(registry_dir / "applications.json", runner=self.runner)
        web = Browser(registry_dir / "browsers.json", opener=self.web_opener, runner=self.runner)
        media = Media(registry_dir / "media_providers.json")
        system = System(registry_dir / "system.json")
        application_skill = ApplicationControl(application)
        web_skill = WebControl(web)
        media_skill = MediaControl(media)
        system_skill = SystemControl(system)
        self.skill_handlers = {
            "application.open": lambda args: application_skill.open(args.get("application", ""), self.execute),
            "application.close": lambda args: application_skill.close(args.get("application", ""), self.execute),
            "web.open": lambda args: web_skill.open(args.get("url", ""), args.get("browser"), self.execute),
            "web.close": lambda args: web_skill.close(args.get("title", ""), self.execute),
            "media.play": lambda args: media_skill.play(args.get("query", ""), args.get("platform", "spotify"), self.execute),
            "media.transport": lambda args: media_skill.transport(args.get("action", ""), args.get("platform", "spotify"), self.execute),
            "system.power": lambda args: system_skill.power(args.get("action", ""), self.execute),
            "system.brightness": lambda args: system_skill.brightness(args.get("value"), self.execute),
            "system.volume": lambda args: system_skill.volume(args.get("value"), self.execute),
            "system.night_light": lambda args: system_skill.night_light(args.get("enabled"), self.execute),
        }

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

        # Context-aware Confirmation override
        if self.pending_frame:
            if any(k in norm_in for k in ("đồng ý", "dong y", "xác nhận", "xac nhan", "chắc chắn", "ok", "yes", "tiếp tục")):
                act = "CONFIRM"
            elif any(k in norm_in for k in ("hủy", "huy", "thôi", "khong", "không", "cancel", "no", "dừng")):
                act = "CANCEL"

        if act == "CONFIRM":
            if not self.pending_frame:
                return {"status": "REJECTED", "reason": "NO_PENDING_ACTION", "response": "Không có yêu cầu nào đang chờ xác nhận."}
            if now > self.pending_frame.expires_at:
                self.pending_frame = None
                return {"status": "EXPIRED", "reason": "CONFIRMATION_EXPIRED", "response": "Yêu cầu trước đó đã hết hạn xác nhận."}
            target_frame = self.pending_frame
            self.pending_frame = None
            exec_result = self._execute_skill(target_frame.skill_id, target_frame.arguments)
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
            responses = {
                "GREETING": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?",
                "THANKS": "Không có chi, rất vui được hỗ trợ bạn!",
                "GOODBYE": "Tạm biệt bạn, hẹn gặp lại!",
                "ACKNOWLEDGEMENT": "Tôi đã hiểu."
            }
            return {"status": "RESPONDED", "response": responses.get(intent, "Tôi đã ghi nhận.")}

        if act in ("UNSUPPORTED", None):
            return {"status": "UNSUPPORTED", "response": "Xin lỗi, tôi chưa hỗ trợ yêu cầu này."}

        if act == "ASK_CLARIFICATION":
            cmd_id = params.get("command_id")
            if goal == "RUN_COMMAND" and cmd_id in ("SHUTDOWN_SYSTEM", "RESTART_SYSTEM"):
                desc = "tắt máy tính" if cmd_id == "SHUTDOWN_SYSTEM" else "khởi động lại máy tính"
                self.pending_frame = PendingFrame(
                    skill_id="system.command", arguments={"command_id": cmd_id}, risk="HIGH",
                    created_at=datetime.now(), expires_at=datetime.now() + timedelta(seconds=60), description=desc
                )
                return {"status": "AWAITING_CONFIRMATION", "risk": "HIGH", "response": f"Bạn có chắc chắn muốn {desc} không?"}
            return {"status": "CLARIFICATION_NEEDED", "response": model_frame.get("response") or "Bạn có thể nói rõ hơn yêu cầu được không?"}

        if act == "EXECUTE":
            return self._handle_execute(raw_input, goal, params)

        return {"status": "INVALID_FRAME", "response": "Không thể xử lý định dạng yêu cầu."}

    def _handle_execute(self, raw_input: str, goal: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
        if goal in ("APPLICATION_CONTROL", "WEB_OPEN"):
            action = params.get("action", "OPEN")
            raw_app = params.get("target" if goal == "WEB_OPEN" else "application", "")
            if isinstance(raw_app, dict):
                raw_app = raw_app.get("value", "")
            resolved_app, score = self.app_registry.resolve(raw_app)
            if not resolved_app:
                resolved_app, score = self.app_registry.resolve(raw_input)

            if not resolved_app:
                return {"status": "UNSUPPORTED", "reason": "APP_NOT_FOUND", "response": f"Không tìm thấy ứng dụng '{raw_app or raw_input}' trong hệ thống."}

            if action not in {"OPEN", "CLOSE"}:
                return {"status": "UNSUPPORTED", "reason": "ACTION_NOT_SUPPORTED", "response": f"Chưa hỗ trợ tác vụ {action} với ứng dụng."}

            if action == "CLOSE":
                if resolved_app.get("type") == "web":
                    result = self._execute_skill("web.close", {"title": resolved_app.get("window_title", resolved_app["name"])})
                    return {"status": "EXECUTED" if result["ok"] else "ERROR", "skill_id": "web.close", "result": result,
                            "response": "Đã đóng trang web." if result["ok"] else f"Không thể đóng trang web: {result.get('error')}."}
                self.pending_frame = PendingFrame(
                    skill_id="application.close", arguments={"application": resolved_app["app_id"]}, risk="MEDIUM",
                    created_at=datetime.now(), expires_at=datetime.now() + timedelta(seconds=60),
                    description=f"đóng {resolved_app['name']}",
                )
                return {"status": "AWAITING_CONFIRMATION", "skill_id": "application.close", "risk": "MEDIUM",
                        "response": f"Bạn có chắc chắn muốn đóng {resolved_app['name']} không?"}

            local_executable = self.app_registry._local_executable(resolved_app)
            web_url = self.app_registry._web_url(resolved_app)
            explicit_web = goal == "WEB_OPEN" or params.get("route") == "WEB" or params.get("browser")
            local_available = bool(local_executable and shutil.which(local_executable))
            if explicit_web:
                route, reason = ("WEB", "WEB_REQUESTED") if web_url else ("UNSUPPORTED", "NO_CAPABILITY")
            elif local_available:
                route, reason = "LOCAL", "LOCAL_AVAILABLE"
            elif web_url:
                route, reason = "WEB", "LOCAL_UNAVAILABLE_WEB_AVAILABLE"
            else:
                route, reason = "UNSUPPORTED", "NO_CAPABILITY"

            if route == "UNSUPPORTED":
                return {"status": "UNSUPPORTED", "reason": reason, "resolved": resolved_app,
                        "response": f"Không có capability phù hợp cho {resolved_app['name']}."}
            self.memory.update_state("current_target_application", resolved_app["app_id"])
            if route == "LOCAL":
                result = self._execute_skill("application.open", {"application": resolved_app["app_id"]})
                status = "EXECUTED" if result["ok"] else ("ROUTED" if result.get("error") == "EXECUTION_DISABLED" else "ERROR")
                return {"status": status, "route": route, "reason": reason,
                        "app_id": resolved_app["app_id"], "app_name": resolved_app["name"],
                        "score": score, "local_executable": local_executable, "web_url": web_url,
                        "browser": params.get("browser"), "result": result,
                        "response": "Đã mở ứng dụng." if result["ok"] else f"Không thể mở ứng dụng: {result.get('error')}."}
            result = self._execute_skill("web.open", {"url": web_url, "browser": params.get("browser")})
            status = "EXECUTED" if result["ok"] else ("ROUTED" if result.get("error") == "EXECUTION_DISABLED" else "ERROR")
            return {"status": status, "route": route, "reason": reason,
                    "app_id": resolved_app["app_id"], "app_name": resolved_app["name"],
                    "score": score, "local_executable": local_executable, "web_url": web_url,
                    "browser": params.get("browser"), "result": result,
                    "response": "Đã mở trang web." if result["ok"] else f"Không thể mở trang web: {result.get('error')}."}

        if goal == "MEDIA_CONTROL":
            action = params.get("action", "PLAY")
            raw_query = params.get("query", "")
            if isinstance(raw_query, dict):
                raw_query = raw_query.get("value", "")
            skill_id = "media.play" if action == "PLAY" else ("media.volume" if "VOLUME" in action else "media.transport")
            exec_result = self._execute_skill(skill_id, {"action": action, "query": raw_query, "platform": params.get("platform", "DEFAULT")})
            if raw_query:
                self.memory.update_state("active_media", raw_query)
            status = "EXECUTED" if exec_result["ok"] else ("ROUTED" if exec_result.get("error") == "EXECUTION_DISABLED" else "ERROR")
            return {"status": status, "skill_id": skill_id, "result": exec_result,
                    "response": f"Đã thực hiện media: {raw_query or action}." if exec_result["ok"] else f"Không thể thực hiện media: {exec_result.get('error')}."}

        if goal == "SYSTEM_CONTROL":
            action = params.get("action", "").upper()
            if action in ("SHUTDOWN", "RESTART"):
                description = "tắt nguồn" if action == "SHUTDOWN" else "khởi động lại"
                self.pending_frame = PendingFrame(
                    skill_id="system.power", arguments={"action": action}, risk="HIGH",
                    created_at=datetime.now(), expires_at=datetime.now() + timedelta(seconds=60),
                    description=description,
                )
                return {"status": "AWAITING_CONFIRMATION", "skill_id": "system.power", "risk": "HIGH",
                        "response": f"Bạn có chắc chắn muốn {description} không?"}
            mapping = {
                "SLEEP": ("system.power", {"action": "SLEEP"}),
                "SET_BRIGHTNESS": ("system.brightness", {"value": params.get("value")}),
                "SET_VOLUME": ("system.volume", {"value": params.get("value")}),
                "NIGHT_LIGHT_ON": ("system.night_light", {"enabled": True}),
                "NIGHT_LIGHT_OFF": ("system.night_light", {"enabled": False}),
            }
            if action not in mapping:
                return {"status": "UNSUPPORTED", "reason": "ACTION_NOT_SUPPORTED", "response": f"Chưa hỗ trợ system action {action}."}
            skill_id, arguments = mapping[action]
            exec_result = self._execute_skill(skill_id, arguments)
            status = ("EXECUTED" if exec_result["ok"] else "ROUTED" if exec_result.get("error") == "EXECUTION_DISABLED"
                      else "UNSUPPORTED" if exec_result.get("error") in ("SKILL_DISABLED", "CAPABILITY_DISABLED") else "ERROR")
            return {"status": status, "skill_id": skill_id, "result": exec_result,
                    "response": "Đã thực hiện điều khiển hệ thống." if exec_result["ok"] else f"Không thể điều khiển hệ thống: {exec_result.get('error')}."}

        if goal == "RUN_COMMAND":
            cmd_id = params.get("command_id")
            if cmd_id == "SLEEP_SYSTEM":
                exec_result = self._execute_skill("system.command", {"command_id": "SLEEP_SYSTEM"})
                return {"status": "EXECUTED" if exec_result["ok"] else "ERROR", "skill_id": "system.command", "result": exec_result,
                        "response": "Đang đưa máy tính vào chế độ ngủ." if exec_result["ok"] else f"Không thể đưa máy vào chế độ ngủ: {exec_result.get('error')}."}
            if cmd_id in ("SHUTDOWN_SYSTEM", "RESTART_SYSTEM"):
                desc = "tắt máy tính" if cmd_id == "SHUTDOWN_SYSTEM" else "khởi động lại máy tính"
                self.pending_frame = PendingFrame(
                    skill_id="system.command", arguments={"command_id": cmd_id}, risk="HIGH",
                    created_at=datetime.now(), expires_at=datetime.now() + timedelta(seconds=60), description=desc
                )
                return {"status": "AWAITING_CONFIRMATION", "risk": "HIGH", "response": f"Bạn có chắc chắn muốn {desc} không?"}

        return {"status": "UNSUPPORTED", "response": f"Chưa hỗ trợ tác vụ {goal}."}

    def _dispatch_route(self, route: str, app: Dict[str, Any], browser: Optional[str]) -> Dict[str, Any]:
        target = self.app_registry._local_executable(app) if route == "LOCAL" else self.app_registry._web_url(app)
        result = {"route": route, "target": target, "started": False, "error": None}
        if not self.execute:
            return {**result, "dry_run": True}
        try:
            if route == "LOCAL":
                self.runner([target])
            elif browser:
                resolved, _ = self.browser_registry.resolve(browser)
                executable = self.browser_registry._local_executable(resolved or {})
                executable = shutil.which(executable) if executable else None
                if not executable:
                    raise RuntimeError("BROWSER_NOT_FOUND")
                self.runner([executable, target])
            elif not self.web_opener(target):
                raise RuntimeError("WEB_OPEN_FAILED")
            result["started"] = True
        except (OSError, RuntimeError) as exc:
            result["error"] = str(exc)
        return result

    @staticmethod
    def _dispatch_response(app_name: str, result: Dict[str, Any]) -> str:
        if result["started"]:
            return f"Đã mở {app_name}."
        if result["error"]:
            return f"Không thể mở {app_name}: {result['error']}."
        return f"Dry-run: sẵn sàng mở {app_name} qua {result['route']}."

    def _execute_skill(self, skill_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        skill = next((item for item in self.skills if item.get("skill_id") == skill_id), None)
        if not skill:
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_NOT_FOUND"}
        if not skill.get("enabled", True):
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_DISABLED"}
        handler = self.skill_handlers.get(skill_id)
        if not handler:
            return {"ok": False, "skill_id": skill_id, "error": "SKILL_NOT_IMPLEMENTED"}
        return handler(args)


# Alias for backward compatibility
RuntimeEngine = AgentHarness
