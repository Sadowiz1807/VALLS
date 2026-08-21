from Runtime.Resources.Application import Application


class ApplicationControl:
    def __init__(self, resource: Application):
        self.resource = resource

    def open(self, application: str, execute: bool = False) -> dict:
        return {"skill_id": "application.open", **self.resource.open(application, execute)}

    def close(self, application: str, execute: bool = False) -> dict:
        return {"skill_id": "application.close", **self.resource.close(application, execute)}
