"""Frontend Pygame UI cho VALLS.

Thiết kế:
- Một animation system thống nhất cho mọi UIAction.
- Action chủ yếu khác nhau bằng accent color và một số tham số chuyển động nhỏ.
- Transition giữa action được nội suy mượt, không reset animation phase.
- Windows dùng layered color-key window để loại bỏ background hình chữ nhật.
- Worker chỉ gửi action/state; Pygame render trong process riêng.

Public API được giữ nguyên:
    UIProcess.start()
    UIProcess.stop()
    UIProcess.set_action()
    UIProcess.set_text()
    UIProcess.poll()
"""
from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
import queue
import sys
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


# Color duy nhất dùng để "đục" nền trên Windows.
# Nên là màu gần như không bao giờ xuất hiện trong UI.
TRANSPARENT_KEY = (1, 2, 3)

AUTO_RESET = {UIAction.SUCCESS, UIAction.ERROR}


@dataclass(frozen=True)
class AnimationStyle:
    """Các tham số animation cho một state.

    Geometry/layout không đổi giữa các state; chỉ màu và mức năng lượng/tốc độ thay đổi nhẹ.
    """

    accent: tuple[int, int, int]
    pulse_strength: float = 1.0
    rotation_speed: float = 1.0
    orbit_speed: float = 1.0
    ripple_speed: float = 1.0
    energy: float = 1.0


ACTION_STYLES = {
    UIAction.NORMAL: AnimationStyle((105, 120, 145), 0.45, 0.45, 0.45, 0.45, 0.45),
    UIAction.LISTENING: AnimationStyle((0, 155, 255), 1.00, 0.75, 0.80, 0.95, 0.90),
    UIAction.SPEECH: AnimationStyle((0, 220, 155), 1.15, 0.95, 1.00, 1.10, 1.00),
    UIAction.THINKING: AnimationStyle((235, 205, 45), 0.75, 1.20, 1.35, 0.75, 0.90),
    UIAction.PROCESSING: AnimationStyle((255, 160, 30), 0.90, 1.55, 1.25, 1.00, 1.00),
    UIAction.EXECUTING: AnimationStyle((255, 95, 35), 1.05, 1.85, 1.45, 1.10, 1.10),
    UIAction.SUCCESS: AnimationStyle((35, 215, 90), 1.30, 0.70, 0.85, 1.25, 1.00),
    UIAction.ERROR: AnimationStyle((235, 55, 65), 1.35, 1.10, 1.20, 1.35, 1.10),
}


@dataclass
class UIConfig:
    width: int = 600
    height: int = 900
    fps: int = 30
    title: str = "AL Voice Local"

    success_seconds: float = 3.0
    error_seconds: float = 5.0

    # Transition action -> action.
    transition_seconds: float = 0.35

    # Transparent overlay settings.
    transparent: bool = True
    borderless: bool = True

    # Animation tuning.
    core_y_ratio: float = 0.42
    base_radius: int = 82
    particle_count: int = 10


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ease_in_out(t: float) -> float:
    """Smoothstep: 0..1 -> 0..1."""
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return (
        int(_lerp(a[0], b[0], t)),
        int(_lerp(a[1], b[1], t)),
        int(_lerp(a[2], b[2], t)),
    )


def _lerp_style(a: AnimationStyle, b: AnimationStyle, t: float) -> AnimationStyle:
    return AnimationStyle(
        accent=_lerp_color(a.accent, b.accent, t),
        pulse_strength=_lerp(a.pulse_strength, b.pulse_strength, t),
        rotation_speed=_lerp(a.rotation_speed, b.rotation_speed, t),
        orbit_speed=_lerp(a.orbit_speed, b.orbit_speed, t),
        ripple_speed=_lerp(a.ripple_speed, b.ripple_speed, t),
        energy=_lerp(a.energy, b.energy, t),
    )


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale RGB brightness mà không dùng alpha."""
    factor = max(0.0, factor)
    return tuple(min(255, max(0, int(c * factor))) for c in color)


class UIState:
    """State machine + visual transition.

    SUCCESS/ERROR tự về NORMAL sau timeout.
    Animation phase không nằm trong state nên không restart khi action đổi.
    """

    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()

        now = monotonic()
        self.action = UIAction.NORMAL
        self.action_since = now

        self.text = ""
        self.subtitle = ""

        self._transition_from = ACTION_STYLES[self.action]
        self._transition_since = now

    def visual_style(self, now: Optional[float] = None) -> AnimationStyle:
        now = now if now is not None else monotonic()
        target = ACTION_STYLES[self.action]

        duration = max(0.001, self.config.transition_seconds)
        progress = _ease_in_out((now - self._transition_since) / duration)

        return _lerp_style(self._transition_from, target, progress)

    def set_action(
        self,
        action: UIAction,
        now: Optional[float] = None,
        text: str = "",
        subtitle: str = "",
    ):
        now = now if now is not None else monotonic()

        # Lấy đúng visual hiện tại trước khi thay target để transition không bị giật.
        current_visual = self.visual_style(now)

        self.action = action
        self.action_since = now
        self._transition_from = current_visual
        self._transition_since = now

        if text:
            self.text = text
        if subtitle:
            self.subtitle = subtitle

    def tick(self, now: Optional[float] = None) -> UIAction:
        now = now if now is not None else monotonic()

        if self.action in AUTO_RESET:
            limit = (
                self.config.success_seconds
                if self.action is UIAction.SUCCESS
                else self.config.error_seconds
            )
            if now - self.action_since >= limit:
                self.set_action(UIAction.NORMAL, now=now)

        return self.action


@dataclass
class RenderResources:
    label_font: pygame.font.Font
    subtitle_font: pygame.font.Font


def _create_resources() -> RenderResources:
    return RenderResources(
        label_font=pygame.font.SysFont("segoeui", 34),
        subtitle_font=pygame.font.SysFont("segoeui", 22),
    )


def _draw_glow(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    accent: tuple[int, int, int],
    energy: float,
):
    """Pseudo-glow phù hợp color-key transparency.

    Không dùng alpha thật vì Windows color-key transparency chỉ đục đúng key color.
    Các vòng tối dần tạo cảm giác glow nhưng không tạo rectangular background.
    """
    layers = (
        (radius + 34, 0.13),
        (radius + 24, 0.18),
        (radius + 15, 0.24),
        (radius + 8, 0.34),
    )
    for r, intensity in layers:
        pygame.draw.circle(
            surface,
            _shade(accent, intensity * energy),
            center,
            max(1, int(r)),
            1,
        )


def _draw_core(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    base_radius: int,
    accent: tuple[int, int, int],
    t: float,
    style: AnimationStyle,
):
    pulse = 0.5 + 0.5 * math.sin(t * 2.6)
    core_radius = int(16 + 5.5 * style.pulse_strength * pulse)
    ring_radius = int(base_radius + 4.0 * style.pulse_strength * pulse)

    _draw_glow(surface, (cx, cy), ring_radius, accent, style.energy)

    # Inner core.
    pygame.draw.circle(surface, _shade(accent, 0.42), (cx, cy), core_radius + 8, 2)
    pygame.draw.circle(surface, accent, (cx, cy), core_radius, 3)
    pygame.draw.circle(surface, _shade(accent, 0.70), (cx, cy), max(2, core_radius // 3))

    # Main breathing ring.
    pygame.draw.circle(surface, _shade(accent, 0.92), (cx, cy), ring_radius, 3)


def _draw_rotating_arcs(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    base_radius: int,
    accent: tuple[int, int, int],
    t: float,
    style: AnimationStyle,
):
    """Hai arc quay đối xứng; geometry luôn giống nhau cho mọi state."""
    speed = style.rotation_speed
    angle = t * speed

    rect1 = pygame.Rect(
        cx - (base_radius + 23),
        cy - (base_radius + 23),
        (base_radius + 23) * 2,
        (base_radius + 23) * 2,
    )
    rect2 = pygame.Rect(
        cx - (base_radius + 42),
        cy - (base_radius + 42),
        (base_radius + 42) * 2,
        (base_radius + 42) * 2,
    )

    pygame.draw.arc(
        surface,
        accent,
        rect1,
        angle,
        angle + math.radians(108),
        4,
    )
    pygame.draw.arc(
        surface,
        _shade(accent, 0.62),
        rect1,
        angle + math.pi,
        angle + math.pi + math.radians(72),
        2,
    )

    pygame.draw.arc(
        surface,
        _shade(accent, 0.48),
        rect2,
        -angle * 0.72,
        -angle * 0.72 + math.radians(88),
        2,
    )
    pygame.draw.arc(
        surface,
        _shade(accent, 0.34),
        rect2,
        -angle * 0.72 + math.pi,
        -angle * 0.72 + math.pi + math.radians(58),
        2,
    )


def _draw_ripples(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    base_radius: int,
    accent: tuple[int, int, int],
    t: float,
    style: AnimationStyle,
):
    """Các ripple vòng tròn chạy liên tục, không reset khi state đổi."""
    for i in range(3):
        phase = (t * 0.38 * style.ripple_speed + i / 3.0) % 1.0
        radius = int(base_radius + 52 + phase * 72)

        # Color giảm dần theo khoảng cách; vẫn là opaque để tương thích color key.
        brightness = max(0.10, (1.0 - phase) * 0.42 * style.energy)
        pygame.draw.circle(
            surface,
            _shade(accent, brightness),
            (cx, cy),
            radius,
            1,
        )


def _draw_particles(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    base_radius: int,
    accent: tuple[int, int, int],
    t: float,
    style: AnimationStyle,
    count: int,
):
    """Orbit particles thống nhất cho mọi action."""
    count = max(1, count)
    orbit_radius = base_radius + 64

    for i in range(count):
        phase = i / count
        direction = -1.0 if i % 2 else 1.0
        angle = (
            phase * math.tau
            + direction * t * (0.34 + 0.08 * (i % 3)) * style.orbit_speed
        )

        wobble = math.sin(t * 1.6 + i * 1.7) * (5.0 + 2.0 * style.energy)
        radius = orbit_radius + wobble

        x = cx + int(math.cos(angle) * radius)
        y = cy + int(math.sin(angle) * radius)

        sparkle = 0.5 + 0.5 * math.sin(t * 3.0 + i * 1.4)
        dot_r = 2 + int(2 * sparkle * style.energy)
        dot_color = _shade(accent, 0.52 + 0.36 * sparkle)

        pygame.draw.circle(surface, dot_color, (x, y), dot_r)


def _draw_energy_wave(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    base_radius: int,
    accent: tuple[int, int, int],
    t: float,
    style: AnimationStyle,
):
    """Một vòng năng lượng biến dạng nhẹ, tạo cảm giác sống động nhưng giữ cùng layout."""
    points: list[tuple[int, int]] = []
    point_count = 48
    base = base_radius + 12

    for i in range(point_count + 1):
        ang = (i / point_count) * math.tau
        wave = (
            math.sin(ang * 4.0 + t * 2.7)
            + 0.45 * math.sin(ang * 7.0 - t * 1.9)
        )
        r = base + wave * 2.8 * style.energy
        x = cx + int(math.cos(ang) * r)
        y = cy + int(math.sin(ang) * r)
        points.append((x, y))

    pygame.draw.lines(
        surface,
        _shade(accent, 0.56),
        False,
        points,
        1,
    )


def _draw_text(
    surface: pygame.Surface,
    config: UIConfig,
    state: UIState,
    resources: RenderResources,
    accent: tuple[int, int, int],
):
    label = resources.label_font.render(
        state.action.value.upper(),
        True,
        accent,
    )
    surface.blit(
        label,
        label.get_rect(
            center=(config.width // 2, int(config.height * 0.82))
        ),
    )

    subtitle = state.subtitle or state.text
    if subtitle:
        sub = resources.subtitle_font.render(
            subtitle,
            True,
            _shade(accent, 0.78),
        )
        surface.blit(
            sub,
            sub.get_rect(
                center=(config.width // 2, int(config.height * 0.86))
            ),
        )


def _draw_template(
    surface: pygame.Surface,
    config: UIConfig,
    style: AnimationStyle,
    t: float,
):
    """Một render pipeline duy nhất cho toàn bộ action."""
    cx = config.width // 2
    cy = int(config.height * config.core_y_ratio)
    accent = style.accent
    r = config.base_radius

    _draw_ripples(surface, cx, cy, r, accent, t, style)
    _draw_particles(
        surface,
        cx,
        cy,
        r,
        accent,
        t,
        style,
        config.particle_count,
    )
    _draw_rotating_arcs(surface, cx, cy, r, accent, t, style)
    _draw_energy_wave(surface, cx, cy, r, accent, t, style)
    _draw_core(surface, cx, cy, r, accent, t, style)


def _render(
    surface: pygame.Surface,
    config: UIConfig,
    state: UIState,
    resources: RenderResources,
    animation_t: float,
    now: float,
):
    # Không còn action background. Toàn bộ nền là transparent key.
    surface.fill(TRANSPARENT_KEY)

    style = state.visual_style(now)
    _draw_template(surface, config, style, animation_t)
    _draw_text(surface, config, state, resources, style.accent)


def _colorref(rgb: tuple[int, int, int]) -> int:
    """RGB -> Win32 COLORREF (0x00BBGGRR)."""
    r, g, b = rgb
    return r | (g << 8) | (b << 16)


def _enable_windows_transparency(
    color_key: tuple[int, int, int],
) -> bool:
    """Enable color-key transparency cho Pygame window trên Windows.

    Trả về False nếu platform/Win32 API không hỗ trợ.
    """
    if sys.platform != "win32":
        return False

    try:
        wm_info = pygame.display.get_wm_info()
        hwnd = wm_info.get("window")
        if not hwnd:
            return False

        user32 = ctypes.windll.user32

        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        LWA_COLORKEY = 0x00000001

        get_window_long = user32.GetWindowLongW
        set_window_long = user32.SetWindowLongW
        set_layered_attributes = user32.SetLayeredWindowAttributes

        get_window_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
        get_window_long.restype = ctypes.c_long

        set_window_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        set_window_long.restype = ctypes.c_long

        set_layered_attributes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_ubyte,
            ctypes.c_uint,
        ]
        set_layered_attributes.restype = ctypes.c_int

        current_style = get_window_long(hwnd, GWL_EXSTYLE)
        set_window_long(hwnd, GWL_EXSTYLE, current_style | WS_EX_LAYERED)

        ok = set_layered_attributes(
            hwnd,
            _colorref(color_key),
            0,
            LWA_COLORKEY,
        )
        return bool(ok)

    except Exception:
        return False


def _create_window(config: UIConfig) -> tuple[pygame.Surface, bool]:
    flags = pygame.NOFRAME if config.borderless else 0
    screen = pygame.display.set_mode(
        (config.width, config.height),
        flags,
    )
    pygame.display.set_caption(config.title)

    transparency_enabled = False
    if config.transparent:
        transparency_enabled = _enable_windows_transparency(TRANSPARENT_KEY)

    return screen, transparency_enabled


def _run_pygame(in_q: mp.Queue, out_q: mp.Queue, config: UIConfig):
    """Child process: sở hữu window + event loop + render."""
    pygame.init()
    pygame.font.init()

    screen, transparency_enabled = _create_window(config)
    clock = pygame.time.Clock()
    resources = _create_resources()
    state = UIState(config)

    animation_started = monotonic()

    out_q.put(
        {
            "event": "UI_READY",
            "timestamp": time.time(),
            "transparent": transparency_enabled,
        }
    )

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                out_q.put(
                    {
                        "event": "UI_CANCEL_REQUEST",
                        "action": state.action.value,
                        "timestamp": time.time(),
                    }
                )

        # Drain queue để tránh UI bị trễ nếu worker gửi nhiều update liên tiếp.
        while True:
            try:
                msg = in_q.get_nowait()
            except queue.Empty:
                break

            kind = msg.get("kind")

            if kind == "stop":
                running = False
                break

            if kind == "action":
                try:
                    action = UIAction(msg["action"])
                except (KeyError, ValueError):
                    continue

                now = monotonic()
                state.set_action(
                    action,
                    now=now,
                    text=msg.get("text", ""),
                    subtitle=msg.get("subtitle", ""),
                )
                out_q.put(
                    {
                        "event": "UI_ACTION",
                        "action": state.action.value,
                        "timestamp": time.time(),
                    }
                )

            elif kind == "text":
                state.text = msg.get("text", "")
                state.subtitle = msg.get("subtitle", "")

        now = monotonic()
        state.tick(now)

        animation_t = now - animation_started
        _render(
            screen,
            config,
            state,
            resources,
            animation_t,
            now,
        )

        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()


class UIProcess:
    """API cho worker: start/stop/set_action/set_text/poll."""

    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        self._in: mp.Queue = mp.Queue()
        self._out: mp.Queue = mp.Queue()
        self._proc: Optional[mp.Process] = None

    def start(self):
        if self._proc is not None:
            return

        self._proc = mp.Process(
            target=_run_pygame,
            args=(self._in, self._out, self.config),
            daemon=True,
        )
        self._proc.start()

        try:
            ready = self._out.get(timeout=10)
            if ready.get("event") != "UI_READY":
                raise RuntimeError("UI process did not become ready")
        except queue.Empty:
            raise RuntimeError(
                "UI process did not become ready (window failed to open)"
            ) from None

    def set_action(
        self,
        action: UIAction,
        text: str = "",
        subtitle: str = "",
    ):
        self._in.put(
            {
                "kind": "action",
                "action": action.value,
                "text": text,
                "subtitle": subtitle,
            }
        )

    def set_text(self, text: str, subtitle: str = ""):
        self._in.put(
            {
                "kind": "text",
                "text": text,
                "subtitle": subtitle,
            }
        )

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
            self._proc.join(timeout=1)

        self._proc = None


if __name__ == "__main__":
    # Demo toàn bộ state:
    #   python -m App.Frontend.app
    #
    # ESC gửi UI_CANCEL_REQUEST; Alt+F4 đóng window.
    ui = UIProcess()
    ui.start()

    try:
        for action in [
            UIAction.LISTENING,
            UIAction.SPEECH,
            UIAction.THINKING,
            UIAction.PROCESSING,
            UIAction.EXECUTING,
            UIAction.SUCCESS,
            UIAction.ERROR,
            UIAction.NORMAL,
        ]:
            ui.set_action(
                action,
                text=action.value,
                subtitle="demo",
            )
            time.sleep(1.5)
    finally:
        ui.stop()

    print("Frontend demo done.")
