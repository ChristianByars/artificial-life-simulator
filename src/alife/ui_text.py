"""HUD text and UI copy (SPEC 4, 5). Headless: no Ursina import, so it is tested without a window.

``hud_lines`` builds the top-left HUD; ``alife.ui.hud`` only draws the lines it returns.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

# Copy (SPEC 4 toasts and confirmations).
SAVED = "Saved"
SAVE_FAILED = "Save failed: {reason}"
CONFIRM_RESET = "Reset to the start? Same seed, same settings."
CONFIRM_NEW = "Start over with a new seed?"
CONFIRM_YES = "Yes"
CONFIRM_CANCEL = "Cancel"
CONFIRM_HINT = "Enter: yes    Esc: cancel"
PAUSED = "PAUSED"

FPS_WINDOW_SECONDS = 0.5  # our assumption: FPS is averaged and refreshed twice a second
_WINDOW_EPS = 1e-9  # float sums of frame times fall a hair short of the window


def format_speed(speed: float) -> str:
    """0.25 -> '0.25×', 1.0 -> '1×', 8.0 -> '8×'."""
    return f"{float(speed):g}×"


def format_sim_time(seconds: float) -> str:
    """Simulated time as 'mm:ss' (minutes do not wrap into hours); '--:--' if not a time."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    whole = math.floor(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


def format_fps(fps: float | None) -> str:
    if fps is None or not math.isfinite(fps) or fps <= 0:
        return "--"
    return f"{fps:.0f}"


def format_stat(value: str | float) -> str:
    """A ``stats()`` value: text as is, whole numbers whole, others to 3 significant digits."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    v = float(value)
    if not math.isfinite(v):
        return str(v)
    if v.is_integer() or abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def hud_lines(
    name: str,
    tick: int,
    sim_time: float,
    speed: float,
    fps: float | None,
    stats: Mapping[str, str | float],
    paused: bool,
    selected: int | None,
) -> list[str]:
    """The HUD, one string per line. Line 1 is the life form's name; 'PAUSED' follows it."""
    lines = [name]
    if paused:
        lines.append(PAUSED)
    lines.append(f"Tick {tick}   Time {format_sim_time(sim_time)}")
    lines.append(f"Speed {format_speed(speed)}   FPS {format_fps(fps)}")
    lines.extend(f"{key}: {format_stat(value)}" for key, value in stats.items())
    if selected is not None:
        lines.append(f"Selected #{selected}")
    return lines


class FpsMeter:
    """Frames per second, averaged over windows of ``window`` seconds of frame time.

    ``fps`` is None until the first window fills. Non-finite or negative frame times are skipped.
    """

    def __init__(self, window: float = FPS_WINDOW_SECONDS) -> None:
        self.window = window
        self.fps: float | None = None
        self._frames = 0
        self._elapsed = 0.0

    def add(self, dt: float) -> None:
        if not math.isfinite(dt) or dt < 0:
            return
        self._frames += 1
        self._elapsed += dt
        if self._elapsed >= self.window - _WINDOW_EPS:
            self.fps = self._frames / self._elapsed
            self._frames = 0
            self._elapsed = 0.0
