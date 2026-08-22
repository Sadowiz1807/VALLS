from Runtime.Resources.Browser import Browser


class WebControl:
    def __init__(self, resource: Browser):
        self.resource = resource

    def open(self, url: str, browser: str | None = None, execute: bool = False) -> dict:
        return {"skill_id": "web.open", **self.resource.open(url, browser, execute)}

    def close(self, title: str, execute: bool = False) -> dict:
        return {"skill_id": "web.close", **self.resource.close_title(title, execute)}
