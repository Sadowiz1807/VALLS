"""Phase 2 UI prototype: Pygame renderer and action state machine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Iterable


class UIAction(str, Enum):
    NORMAL = "normal"
    LISTENING = "listening"
    SPEECH = "speech"
    THINKING = "thinking"
    PROCESSING = "processing"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True)
class UIConfig:
    width: int = 600
    height: int = 900
    fps: int = 30
    success_seconds: float = 3.0
    error_seconds: float = 5.0


class UIState:
    def __init__(self, config: UIConfig | None = None) -> None:
        self.config = config or UIConfig()
        self.action = UIAction.NORMAL
        self._reset_at: float | None = None

    def set_action(self, action: UIAction, now: float | None = None) -> None:
        self.action = action
        if action in (UIAction.SUCCESS, UIAction.ERROR):
            seconds = self.config.success_seconds if action is UIAction.SUCCESS else self.config.error_seconds
            self._reset_at = (monotonic() if now is None else now) + seconds
        else:
            self._reset_at = None

    def tick(self, now: float | None = None) -> UIAction:
        if self._reset_at is not None and (monotonic() if now is None else now) >= self._reset_at:
            self.set_action(UIAction.NORMAL, now)
        return self.action


def load_frames(asset_root: Path, action: UIAction, *, pygame_module) -> list:
    """Load PNG frames once; missing assets fall back to a red placeholder."""
    folder = asset_root / action.value
    paths = sorted(folder.glob("frame_*.png"))
    frames = [pygame_module.image.load(str(path)).convert_alpha() for path in paths]
    return frames


def run_ui(asset_root: Path, *, pygame_module=None) -> None:
    """Run the windowed 6:9 prototype. Import is delayed for headless tests."""
    if pygame_module is None:
        import pygame as pygame_module

    pygame_module.init()
    config = UIConfig()
    screen = pygame_module.display.set_mode((config.width, config.height))
    pygame_module.display.set_caption("AL Voice Local")
    clock = pygame_module.time.Clock()
    state = UIState(config)
    frame_index = 0
    frames: dict[UIAction, list] = {action: load_frames(asset_root, action, pygame_module=pygame_module) for action in UIAction}

    running = True
    while running:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                running = False
        action = state.tick()
        action_frames = frames[action] or frames[UIAction.NORMAL]
        if action_frames:
            frame = action_frames[frame_index % len(action_frames)]
            frame_index += 1
            screen.fill((0, 0, 0))
            rect = frame.get_rect(center=screen.get_rect().center)
            screen.blit(frame, rect)
        else:
            screen.fill((180, 0, 0))
        pygame_module.display.flip()
        clock.tick(config.fps)

    pygame_module.quit()


if __name__ == "__main__":
    run_ui(Path(__file__).parent / "assets" / "animations")
