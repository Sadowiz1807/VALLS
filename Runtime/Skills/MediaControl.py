from Runtime.Resources.Media import Media


class MediaControl:
    ACTIONS = {"PAUSE": "pause", "RESUME": "resume", "STOP": "pause", "NEXT": "next", "PREVIOUS": "previous"}

    def __init__(self, resource: Media):
        self.resource = resource

    def play(self, query: str, platform: str = "spotify", execute: bool = False) -> dict:
        if not query.strip():
            return {"ok": False, "skill_id": "media.play", "error": "QUERY_REQUIRED"}
        return {"skill_id": "media.play", **self.resource.run(["play", query], platform, execute)}

    def transport(self, action: str, platform: str = "spotify", execute: bool = False) -> dict:
        command = self.ACTIONS.get(action.upper())
        if not command:
            return {"ok": False, "skill_id": "media.transport", "error": "ACTION_UNSUPPORTED"}
        return {"skill_id": "media.transport", **self.resource.run([command], platform, execute)}
