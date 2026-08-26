from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSpec:
    category: str
    fallbackable: bool = False
    default_side_effect_state: str = "NOT_STARTED"


ERRORS = {
    "UNSUPPORTED_SEMANTIC": ErrorSpec("SEMANTIC"),
    "SKILL_NOT_FOUND": ErrorSpec("SKILL"),
    "SKILL_DISABLED": ErrorSpec("SKILL"),
    "SKILL_NOT_AVAILABLE": ErrorSpec("SKILL"),
    "INPUT_REQUIRED": ErrorSpec("INPUT"),
    "INPUT_TYPE_INVALID": ErrorSpec("INPUT"),
    "INPUT_OUT_OF_RANGE": ErrorSpec("INPUT"),
    "ACTION_UNSUPPORTED": ErrorSpec("INPUT"),
    "RESOURCE_NOT_FOUND": ErrorSpec("RESOURCE"),
    "RESOURCE_DISABLED": ErrorSpec("RESOURCE"),
    "RESOURCE_CONTRACT_VIOLATION": ErrorSpec("RESOURCE"),
    "PROVIDER_UNAVAILABLE": ErrorSpec("PROVIDER", True),
    "PROVIDER_DISCONNECTED": ErrorSpec("PROVIDER", True),
    "DEPENDENCY_UNAVAILABLE": ErrorSpec("PROVIDER", True),
    "BACKEND_UNAVAILABLE": ErrorSpec("PROVIDER", True),
    "TARGET_UNAVAILABLE": ErrorSpec("TARGET"),
    "PERMISSION_DENIED": ErrorSpec("POLICY"),
    "POLICY_DENIED": ErrorSpec("POLICY"),
    "CONFIRMATION_REQUIRED": ErrorSpec("POLICY"),
    "CONFIRMATION_EXPIRED": ErrorSpec("POLICY"),
    "CONFIRMATION_MISMATCH": ErrorSpec("POLICY"),
    "EXECUTION_FAILED": ErrorSpec("EXECUTION", default_side_effect_state="UNKNOWN"),
    "EVIDENCE_MISMATCH": ErrorSpec("EVIDENCE", default_side_effect_state="UNKNOWN"),
}


def error_spec(code: str) -> ErrorSpec:
    return ERRORS[code]
