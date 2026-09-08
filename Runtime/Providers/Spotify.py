from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable


_SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
_CACHE = Path.home() / "AppData/Local/hermes/skills/media/spotify-local-control/.spotify_token_cache"


class SpotifyProvider:
    def __init__(self, client_factory: Callable | None = None):
        self.client_factory = client_factory or self._client

    @staticmethod
    def available() -> bool:
        return bool(
            _CACHE.is_file()
            and os.getenv("SPOTIFY_CLIENT_ID")
            and os.getenv("SPOTIFY_CLIENT_SECRET")
        )

    @staticmethod
    def _client():
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        auth = SpotifyOAuth(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
            redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            scope=_SCOPE,
            cache_path=str(_CACHE),
            open_browser=False,
        )
        return spotipy.Spotify(auth_manager=auth)

    def resolve(self, arguments: dict, _execute: bool = False) -> dict:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "INPUT_REQUIRED", "side_effect_state": "NOT_STARTED"}
        target = query.split("?", 1)[0]
        exact = re.fullmatch(r"spotify:track:([A-Za-z0-9]+)", target)
        if not exact:
            exact = re.fullmatch(r"https://open\.spotify\.com/track/([A-Za-z0-9]+)", target)
        if exact:
            return {
                "ok": True, "uri": f"spotify:track:{exact.group(1)}", "media_type": "track",
                "item": {"name": None, "artists": []},
            }
        items = (((self.client_factory().search(q=query, type="track", limit=2) or {})
                  .get("tracks") or {}).get("items") or [])
        if not items:
            return {"ok": False, "error": "TARGET_UNAVAILABLE", "side_effect_state": "NOT_STARTED"}
        if len(items) != 1:
            return {"ok": False, "error": "TARGET_AMBIGUOUS", "side_effect_state": "NOT_STARTED"}
        item = items[0]
        return {
            "ok": True,
            "uri": item["uri"],
            "media_type": "track",
            "item": {
                "name": item.get("name"),
                "artists": [artist.get("name") for artist in item.get("artists", [])],
            },
        }

    def play(self, arguments: dict, execute: bool = False) -> dict:
        uri = arguments.get("uri")
        if not isinstance(uri, str) or not uri.startswith("spotify:track:"):
            return {"ok": False, "error": "RESOURCE_CONTRACT_VIOLATION", "side_effect_state": "NOT_STARTED"}
        if not execute:
            return {"ok": False, "error": "EXECUTION_DISABLED", "side_effect_state": "NOT_STARTED"}
        client = self.client_factory()
        devices = (client.devices() or {}).get("devices") or []
        active = [device for device in devices if device.get("is_active") and not device.get("is_restricted")]
        if len(active) != 1 or not active[0].get("id"):
            error = "TARGET_AMBIGUOUS" if len(active) > 1 else "TARGET_UNAVAILABLE"
            return {"ok": False, "error": error, "side_effect_state": "NOT_STARTED"}
        device = active[0]
        client.start_playback(device_id=device["id"], uris=[uri])
        try:
            observed = client.current_playback() or {}
        except Exception as exc:
            return {"ok": False, "error": "EXECUTION_FAILED", "side_effect_state": "STARTED",
                    "message": str(exc)}
        item = observed.get("item") or {}
        observed_device = observed.get("device") or {}
        evidence = {
            "uri": item.get("uri"),
            "playing": bool(observed.get("is_playing")),
            "device_id": observed_device.get("id"),
            "device_name": observed_device.get("name"),
        }
        if evidence["uri"] != uri or not evidence["playing"] or evidence["device_id"] != device["id"]:
            return {"ok": False, "error": "EVIDENCE_MISMATCH", "side_effect_state": "STARTED", "observed": evidence}
        return {"ok": True, "observed": evidence}
