from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any


def _normal(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value.strip().lower()).split())


def resolve(path: Path, query: str) -> dict[str, Any] | None:
    needle = _normal(query)
    if not needle or not path.exists():
        return None
    for item in json.loads(path.read_text(encoding="utf-8")):
        if not item.get("enabled", True):
            continue
        names = [item.get("app_id", ""), item.get("browser_id", ""), item.get("provider_id", ""), item.get("name", ""), *item.get("aliases", [])]
        if needle in {_normal(str(name)) for name in names if name}:
            return item
    return None
