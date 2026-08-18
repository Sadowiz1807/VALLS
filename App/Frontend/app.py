"""Frontend: Pygame UI theo Phase 1 contract.

Mỗi action = 1 màu nền + màu nhấn + decoration vẽ bằng pygame.draw (không image frames).
Worker chỉ gửi action/state; Pygame render trên process riêng.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import Optional

import pygame


class UIAction(str, Enum):
    NORMAL = "normal"
    LISTENING = "listening"
    SPEECH = "speech"
    THINKING = "thinking"
    PROCESSING = "processing"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"


ACTION_PALETTE = {
    UIAction.NORMAL: ((10, 14, 20), (60, 70, 90)),
    UIAction.LISTENING: ((8, 16, 30), (0, 150, 255)),
    UIAction.SPEECH: ((6, 22, 18), (0, 200, 100)),
    UIAction.THINKING: ((22, 20, 6), (200, 200, 0)),
    UIAction.PROCESSING: ((26, 16, 6), (255, 165, 0)),
    UIAction.EXECUTING: ((28, 12, 8), (255, 100, 0)),
    UIAction.SUCCESS: ((6, 22, 10), (0, 200, 0)),
    UIAction.ERROR: ((26, 6, 6), (220, 40, 40)),
}
AUTO_RESET = {UIAction.SUCCESS, UIAction.ERROR}


@dataclass
class UIConfig:
    width: int = 600          # 6:9 portrait
    height: int = 900
    fps: int = 30
    title: str = "AL Voice Local"
    success_seconds: float = 3.0
    error_seconds: float = 5.0


class UIState:
    """State machine: success/error tự về normal sau timeout, action khác giữ nguyên."""

    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        self.action = UIAction.NORMAL
        self.since = monotonic()
        self.text = ""
        self.subtitle = ""

    def set_action(self, action: UIAction, now: Optional[float] = None,
                   text: str = "", subtitle: str = ""):
        self.action = action
        self.since = now if now is not None else monotonic()
        if text:
            self.text = text
        if subtitle:
            self.subtitle = subtitle

    def tick(self, now: Optional[float] = None) -> UIAction:
        now = now if now is not None else monotonic()
        if self.action in AUTO_RESET:
            limit = (self.config.success_seconds if self.action is UIAction.SUCCESS
                     else self.config.error_seconds)
            if now - self.since >= limit:
                self.action = UIAction.NORMAL
        return self.action


def _draw_template(surface, action: UIAction, accent, t: float, w: int, h: int):
    """MỘT layout cố định cho mọi action: lưới chấm nhạt + vòng lõi giữa màn hình.
    Action chỉ đổi màu (palette) + motif nhỏ ở trung tâm — không vẽ khác layout."""
    cx, cy = w // 2, int(h * 0.42)
    # lưới chấm nhạt, cố định, không đổi theo action
    for x in range(50, w, 70):
        for y in range(50, h, 70):
            pygame.draw.circle(surface, accent, (x, y), 1)
    # lõi: vòng tròn cố định
    pygame.draw.circle(surface, accent, (cx, cy), 80, 3)
    _draw_motif(surface, action, accent, t, cx, cy)


def _draw_motif(surface, action: UIAction, accent, t: float, cx: int, cy: int):
    """Họa tiết nhỏ (~60px) ở trung tâm vòng lõi, khác nhau theo action."""
    if action is UIAction.NORMAL:
        pygame.draw.circle(surface, accent, (cx, cy), 10)
    elif action is UIAction.LISTENING:  # vòng thở
        r = 28 + int(8 * (0.5 + 0.5 * math.sin(t * 2)))
        pygame.draw.circle(surface, accent, (cx, cy), r, 3)
    elif action is UIAction.SPEECH:  # 3 cột sóng âm nhỏ
        for i in range(3):
            hgt = 20 + int(24 * ((t * 3 + i * 0.5) % 1.0))
            x = cx + (i - 1) * 20
            pygame.draw.rect(surface, accent, (x - 5, cy - hgt, 10, hgt))
    elif action is UIAction.THINKING:  # 2 chấm quỹ đạo nhỏ
        for i in range(2):
            ang = t * 2.5 + i * math.pi
            x = cx + int(30 * math.cos(ang))
            y = cy + int(30 * math.sin(ang))
            pygame.draw.circle(surface, accent, (x, y), 6)
    elif action is UIAction.PROCESSING:  # cung quay
        start = (t * 3) % (2 * math.pi)
        pygame.draw.arc(surface, accent, (cx - 30, cy - 30, 60, 60), start, start + 4.0, 5)
    elif action is UIAction.EXECUTING:  # thanh tiến trình nhỏ
        frac = (t * 0.4) % 1.0
        pygame.draw.rect(surface, accent, (cx - 40, cy - 5, int(80 * frac), 10), border_radius=5)
    elif action is UIAction.SUCCESS:  # dấu tick
        pygame.draw.lines(surface, accent, False, [(cx - 18, cy), (cx - 6, cy + 12), (cx + 20, cy - 12)], 6)
    elif action is UIAction.ERROR:  # chữ X
        pygame.draw.line(surface, accent, (cx - 14, cy - 14), (cx + 14, cy + 14), 6)
        pygame.draw.line(surface, accent, (cx + 14, cy - 14), (cx - 14, cy + 14), 6)


def _render(surface, config: UIConfig, state: UIState, t: float):
    bg, accent = ACTION_PALETTE[state.action]
    surface.fill(bg)
    _draw_template(surface, state.action, accent, t, config.width, config.height)
    label = pygame.font.SysFont("segoeui", 34).render(
        state.action.value.upper(), True, accent)
    surface.blit(label, label.get_rect(center=(config.width // 2, int(config.height * 0.82))))
    subtitle = state.subtitle or state.text
    if subtitle:
        sub = pygame.font.SysFont("segoeui", 22).render(
            subtitle, True, (220, 225, 235))
        surface.blit(sub, sub.get_rect(center=(config.width // 2, int(config.height * 0.86))))


def _run_pygame(in_q: mp.Queue, out_q: mp.Queue, config: UIConfig):
    """Child process: sở hữu window + event loop + render."""
    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption(config.title)
    clock = pygame.time.Clock()
    state = UIState(config)
    out_q.put({"event": "UI_READY", "timestamp": time.time()})
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                out_q.put({"event": "UI_CANCEL_REQUEST",
                           "action": state.action.value, "timestamp": time.time()})
        try:
            msg = in_q.get_nowait()
        except queue.Empty:
            msg = None
        if msg is not None:
            if msg.get("kind") == "stop":
                running = False
            elif msg["kind"] == "action":
                state.set_action(UIAction(msg["action"]), text=msg.get("text", ""),
                                 subtitle=msg.get("subtitle", ""))
                out_q.put({"event": "UI_ACTION", "action": state.action.value,
                           "timestamp": time.time()})
            elif msg["kind"] == "text":
                state.text = msg.get("text", "")
                state.subtitle = msg.get("subtitle", "")
        state.tick()
        _render(screen, config, state, time.time())
        pygame.display.flip()
        clock.tick(config.fps)
    pygame.quit()


class UIProcess:
    """Giao diện cho worker: start/stop/set_action/set_text/poll; window chạy process riêng."""

    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        self._in: mp.Queue = mp.Queue()
        self._out: mp.Queue = mp.Queue()
        self._proc: Optional[mp.Process] = None

    def start(self):
        if self._proc is not None:
            return
        self._proc = mp.Process(target=_run_pygame, args=(self._in, self._out, self.config),
                                daemon=True)
        self._proc.start()
        try:
            ready = self._out.get(timeout=10)
            if ready.get("event") != "UI_READY":
                raise RuntimeError("UI process did not become ready")
        except queue.Empty:
            raise RuntimeError("UI process did not become ready (window failed to open)") from None

    def set_action(self, action: UIAction, text: str = "", subtitle: str = ""):
        self._in.put({"kind": "action", "action": action.value, "text": text, "subtitle": subtitle})

    def set_text(self, text: str, subtitle: str = ""):
        self._in.put({"kind": "text", "text": text, "subtitle": subtitle})

    def poll(self) -> list:
        events = []
        while True:
            try:
                events.append(self._out.get_nowait())
            except queue.Empty:
                return events

    def stop(self):
        if self._proc is None:
            return
        self._in.put({"kind": "stop"})
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
        self._proc = None


if __name__ == "__main__":
    # Tự chạy: `py -3.13 -m App.Frontend.app` — duyệt qua các action, ESC để thoát
    ui = UIProcess()
    ui.start()
    for a in [UIAction.LISTENING, UIAction.SPEECH, UIAction.THINKING,
              UIAction.PROCESSING, UIAction.EXECUTING, UIAction.SUCCESS,
              UIAction.ERROR, UIAction.NORMAL]:
        ui.set_action(a, text=a.value, subtitle="demo")
        time.sleep(1.5)
    ui.stop()
    print("Frontend demo done.")
