from __future__ import annotations

from urllib.parse import urlencode


_SEARCH_ENDPOINTS = {
    "GOOGLE": "https://www.google.com/search",
    "BING": "https://www.bing.com/search",
    "DUCKDUCKGO": "https://duckduckgo.com/",
}


def build_search_url(arguments: dict, _execute: bool = False) -> dict:
    query = arguments.get("query")
    engine = str(arguments.get("engine") or "GOOGLE").upper()
    if engine not in _SEARCH_ENDPOINTS:
        return {
            "ok": False,
            "error": "INPUT_VALUE_INVALID",
            "side_effect_state": "NOT_STARTED",
        }
    return {
        "ok": True,
        "url": f"{_SEARCH_ENDPOINTS[engine]}?{urlencode({'q': query})}",
        "engine": engine,
    }
