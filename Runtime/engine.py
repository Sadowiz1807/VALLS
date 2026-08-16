"""
Runtime Core Prototype — Local Voice AI Assistant (Phase 1)
Bao gồm:
1. Application / Skill Registry & Resolver (Exact, Normalized, Fuzzy)
2. Schema & Safety Policy Validator (Fail-closed, Allowlist)
3. Confirmation State Machine (Pending frame, expiry, CONFIRM/CANCEL)
4. Application & System Adapters (Mock/Real execution boundaries)
5. Grounded Response Renderer (Pre/Post-execution, Clarify, Confirm)
"""
from __future__ import annotations
import difflib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

class ApplicationRegistry:
    def __init__(self, config_path: Optional[Path] = None):
        self.apps: List[Dict[str, Any]] = []
        if config_path and config_path.exists():
            self.apps = json.loads(config_path.read_text(encoding="utf-8"))

    def resolve(self, query: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """Resolve tên ứng dụng từ raw query hoặc dynamic span qua exact, normalized, fuzzy."""
        q = normalize_text(query)
        if not q:
            return None, 0.0

        # 1. Exact match trên app_id hoặc name
        for app in self.apps:
            if not app.get("enabled", True):
                continue
            if q == normalize_text(app["app_id"]) or q == normalize_text(app["name"]):
                return app, 1.0

        # 2. Exact match trên aliases
        for app in self.apps:
            if not app.get("enabled", True):
                continue
            for alias in app.get("aliases", []):
                if q == normalize_text(alias):
                    return app, 1.0

        # 3. Token containment / substring
        best_app = None
        best_score = 0.0
        for app in self.apps:
            if not app.get("enabled", True):
                continue
            all_names = [app["app_id"], app["name"]] + app.get("aliases", [])
            for name in all_names:
                norm_name = normalize_text(name)
                # Containment check
                if q in norm_name or norm_name in q:
                    score = min(len(q), len(norm_name)) / max(len(q), len(norm_name)) * 0.95
                    if score > best_score:
                        best_score = score
                        best_app = app
                # Fuzzy ratio
                ratio = difflib.SequenceMatcher(None, q, norm_name).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_app = app

        if best_score >= 0.6:
            return best_app, best_score
        return None, best_score

class RuntimeEngine:
    def __init__(self, registry_dir: Path):
        self.app_registry = ApplicationRegistry(registry_dir / "applications.json")
        self.skills_path = registry_dir / "skills.json"
        self.skills: List[Dict[str, Any]] = []
        if self.skills_path.exists():
            self.skills = json.loads(self.skills_path.read_text(encoding="utf-8"))
        self.pending_frame: Optional[PendingFrame] = None
        self.execution_log: List[Dict[str, Any]] = []

    def dispatch_turn(self, raw_input: str, model_frame: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nhận semantic frame từ VSAD (hoặc ASR/Model layer),
        Validate -> Resolve -> Policy/Confirmation -> Execute -> Grounded Response.
        """
        act = model_frame.get("act")
        goal = model_frame.get("goal")
        params = model_frame.get("parameters", {})
        now = datetime.now()

        # 1. Xử lý CONFIRM
        if act == "CONFIRM":
            if not self.pending_frame:
                return {
                    "status": "REJECTED",
                    "reason": "NO_PENDING_ACTION",
                    "response": "Không có yêu cầu nào đang chờ xác nhận."
                }
            if now > self.pending_frame.expires_at:
                self.pending_frame = None
                return {
                    "status": "EXPIRED",
                    "reason": "CONFIRMATION_EXPIRED",
                    "response": "Yêu cầu trước đó đã hết hạn xác nhận."
                }
            # Thực thi pending action
            target_frame = self.pending_frame
            self.pending_frame = None
            exec_result = self._execute_skill(target_frame.skill_id, target_frame.arguments)
            return {
                "status": "EXECUTED",
                "skill_id": target_frame.skill_id,
                "result": exec_result,
                "response": f"Đã xác nhận và thực thi: {target_frame.description}." if exec_result["ok"] else f"Thực thi thất bại: {exec_result.get('error')}"
            }

        # 2. Xử lý CANCEL
        if act == "CANCEL":
            if self.pending_frame:
                desc = self.pending_frame.description
                self.pending_frame = None
                return {
                    "status": "CANCELLED",
                    "response": f"Đã hủy yêu cầu: {desc}."
                }
            return {
                "status": "CANCELLED",
                "response": "Đã hủy thao tác."
            }

        # 3. Xử lý RESPOND (Social response)
        if act == "RESPOND":
            intent = params.get("intent", "GREETING")
            responses = {
                "GREETING": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?",
                "THANKS": "Không có chi, rất vui được hỗ trợ bạn!",
                "GOODBYE": "Tạm biệt bạn, hẹn gặp lại!",
                "ACKNOWLEDGEMENT": "Tôi đã hiểu."
            }
            return {
                "status": "RESPONDED",
                "response": responses.get(intent, "Tôi đã ghi nhận.")
            }

        # 4. Fail-closed với UNSUPPORTED hoặc invalid act
        if act in ("UNSUPPORTED", None):
            return {
                "status": "UNSUPPORTED",
                "response": "Xin lỗi, tôi chưa hỗ trợ yêu cầu này."
            }

        if act == "ASK_CLARIFICATION":
            # Nếu model ask confirmation cho lệnh shutdown/restart
            cmd_id = params.get("command_id")
            if goal == "RUN_COMMAND" and cmd_id in ("SHUTDOWN_SYSTEM", "RESTART_SYSTEM"):
                desc = "tắt máy tính" if cmd_id == "SHUTDOWN_SYSTEM" else "khởi động lại máy tính"
                self.pending_frame = PendingFrame(
                    skill_id="system.command",
                    arguments={"command_id": cmd_id},
                    risk="HIGH",
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=60),
                    description=desc
                )
                return {
                    "status": "AWAITING_CONFIRMATION",
                    "risk": "HIGH",
                    "response": f"Bạn có chắc chắn muốn {desc} không?"
                }
            return {
                "status": "CLARIFICATION_NEEDED",
                "response": model_frame.get("response_text") or "Bạn có thể nói rõ hơn yêu cầu được không?"
            }

        # 5. Xử lý EXECUTE
        if act == "EXECUTE":
            return self._handle_execute(raw_input, goal, params)

        return {"status": "INVALID_FRAME", "response": "Không thể xử lý định dạng yêu cầu."}

    def _handle_execute(self, raw_input: str, goal: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
        # A. APPLICATION_CONTROL
        if goal == "APPLICATION_CONTROL":
            action = params.get("action", "OPEN")
            raw_app = params.get("application", "")
            if isinstance(raw_app, dict):
                raw_app = raw_app.get("value", "")
            
            # Nếu span rỗng, thử extract từ raw_input
            search_query = raw_app if raw_app else raw_input
            resolved_app, score = self.app_registry.resolve(search_query)

            if not resolved_app:
                return {
                    "status": "UNSUPPORTED",
                    "reason": "APP_NOT_FOUND",
                    "response": f"Không tìm thấy ứng dụng '{raw_app or raw_input}' trong hệ thống."
                }

            skill_id = "application.open" if action == "OPEN" else "application.close"
            exec_result = self._execute_skill(skill_id, {"app_id": resolved_app["app_id"], "name": resolved_app["name"], "executable": resolved_app["executable"]})
            action_vn = "mở" if action == "OPEN" else "đóng"
            return {
                "status": "EXECUTED",
                "skill_id": skill_id,
                "resolved": resolved_app,
                "result": exec_result,
                "response": f"Tôi sẽ {action_vn} {resolved_app['name']}." if exec_result["ok"] else f"Không thể {action_vn} {resolved_app['name']}: {exec_result.get('error')}"
            }

        # B. RUN_COMMAND (System commands)
        if goal == "RUN_COMMAND":
            command_id = params.get("command_id")
            allowlist = ["SHUTDOWN_SYSTEM", "RESTART_SYSTEM", "SLEEP_SYSTEM", "LOCK_SCREEN", "TAKE_SCREENSHOT"]
            if command_id not in allowlist:
                return {
                    "status": "REJECTED",
                    "reason": "COMMAND_NOT_ALLOWLISTED",
                    "response": "Lệnh hệ thống không nằm trong danh mục cho phép."
                }

            # Kiểm tra policy confirmation
            if command_id in ("SHUTDOWN_SYSTEM", "RESTART_SYSTEM"):
                desc = "tắt máy tính" if command_id == "SHUTDOWN_SYSTEM" else "khởi động lại máy tính"
                self.pending_frame = PendingFrame(
                    skill_id="system.command",
                    arguments={"command_id": command_id},
                    risk="HIGH",
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=60),
                    description=desc
                )
                return {
                    "status": "AWAITING_CONFIRMATION",
                    "risk": "HIGH",
                    "response": f"Bạn có chắc chắn muốn {desc} không?"
                }

            # Lệnh không cần confirm: SLEEP, LOCK, SCREENSHOT
            exec_result = self._execute_skill("system.command", {"command_id": command_id})
            desc_map = {
                "SLEEP_SYSTEM": "Đã đưa máy vào chế độ ngủ.",
                "LOCK_SCREEN": "Đã khóa màn hình máy tính.",
                "TAKE_SCREENSHOT": "Đã chụp ảnh màn hình."
            }
            return {
                "status": "EXECUTED",
                "skill_id": "system.command",
                "result": exec_result,
                "response": desc_map.get(command_id, "Đã thực thi lệnh hệ thống.")
            }

        # C. MEDIA_CONTROL
        if goal == "MEDIA_CONTROL":
            action = params.get("action", "PLAY")
            query = params.get("query", "")
            if isinstance(query, dict):
                query = query.get("value", "")
            exec_result = self._execute_skill("media.play", {"action": action, "query": query})
            return {
                "status": "EXECUTED",
                "skill_id": "media.play",
                "result": exec_result,
                "response": f"Đang phát '{query}'." if query else "Đang điều khiển phát nhạc."
            }

        return {
            "status": "UNSUPPORTED",
            "reason": f"GOAL_{goal}_NOT_IMPLEMENTED",
            "response": "Tác vụ này chưa được cấu hình bộ xử lý."
        }

    def _execute_skill(self, skill_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter mock/safe execution cho prototype Phase 1."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "args": args,
            "ok": True,
            "error": None
        }
        self.execution_log.append(record)
        return {"ok": True, "executed_at": record["timestamp"], "skill_id": skill_id, "args": args}
