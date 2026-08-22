from Runtime.Resources.System import System


class SystemControl:
    def __init__(self, resource: System):
        self.resource = resource

    def power(self, action: str, execute: bool = False) -> dict:
        return {"skill_id": "system.power", **self.resource.power(action, execute)}

    def brightness(self, value: int, execute: bool = False) -> dict:
        return {"skill_id": "system.brightness", **self.resource.set_percent("brightness", value, execute)}

    def volume(self, value: int, execute: bool = False) -> dict:
        return {"skill_id": "system.volume", **self.resource.set_percent("volume", value, execute)}

    def night_light(self, enabled: bool, execute: bool = False) -> dict:
        return {"skill_id": "system.night_light", **self.resource.night_light(enabled, execute)}
