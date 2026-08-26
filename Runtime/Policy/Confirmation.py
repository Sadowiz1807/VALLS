from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any


def arguments_digest(arguments: dict[str, Any]) -> str:
    canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConfirmationGrant:
    grant_id: str
    request_id: str
    skill_id: str
    resource_id: str
    arguments_digest: str
    effective_risk: str
    issued_at: datetime
    expires_at: datetime


class ConfirmationStore:
    def __init__(self):
        self._grants: dict[str, ConfirmationGrant] = {}
        self._consumed: set[str] = set()
        self._lock = Lock()

    def issue(self, request_id: str, skill_id: str, resource_id: str,
              arguments: dict[str, Any], effective_risk: str,
              ttl: timedelta = timedelta(seconds=60)) -> ConfirmationGrant:
        now = datetime.now(timezone.utc)
        grant = ConfirmationGrant(
            grant_id=uuid.uuid4().hex, request_id=request_id, skill_id=skill_id,
            resource_id=resource_id, arguments_digest=arguments_digest(arguments),
            effective_risk=effective_risk, issued_at=now, expires_at=now + ttl,
        )
        with self._lock:
            self._grants[grant.grant_id] = grant
        return grant

    def validate(self, grant: Any, request_id: str, skill_id: str, resource_id: str,
                 arguments: dict[str, Any], effective_risk: str,
                 *, consume: bool = False) -> str | None:
        with self._lock:
            if not isinstance(grant, ConfirmationGrant) or self._grants.get(grant.grant_id) is not grant:
                return "CONFIRMATION_MISMATCH"
            if grant.grant_id in self._consumed:
                return "POLICY_DENIED"
            if datetime.now(timezone.utc) > grant.expires_at:
                return "CONFIRMATION_EXPIRED"
            if (grant.request_id, grant.skill_id, grant.resource_id, grant.arguments_digest, grant.effective_risk) != (
                request_id, skill_id, resource_id, arguments_digest(arguments), effective_risk,
            ):
                return "CONFIRMATION_MISMATCH"
            if consume:
                self._consumed.add(grant.grant_id)
            return None
