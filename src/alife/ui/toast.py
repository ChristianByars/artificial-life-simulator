"""Toast: a short message at the bottom centre that fades out (SPEC 4: "Saved"). GUI-only."""

from __future__ import annotations

import math

from ursina import Entity, Text, color, time
from ursina.models.procedural.quad import Quad

# Layout, colours and timing: our assumption (style, not biology). Ursina UI units.
TOAST_Y = -0.44  # below the dish at the default pose
TEXT_SCALE = 1.2
PADDING_X = 0.03
PADDING_Y = 0.016
PANEL_RADIUS = 0.5  # Quad corner radius, relative to the shorter side: a pill
TEXT_RGB = (255, 255, 255)
PANEL_RGB = (30, 34, 44)
PANEL_ALPHA = 230
SHOW_SECONDS = 2.0  # how long a toast stays, fade included
FADE_SECONDS = 0.4  # the last part of SHOW_SECONDS, fading out


class Toast(Entity):
    """``show(message)`` displays ``message``; the toast hides itself after ``seconds``."""

    def __init__(self, parent: Entity) -> None:
        super().__init__(parent=parent, name="toast", y=TOAST_Y)
        self.panel = Entity(parent=self, z=0.01)
        self.label = Text("", parent=self, origin=(0, 0), scale=TEXT_SCALE, use_tags=False)
        self.message = ""
        self._remaining = 0.0
        self.enabled = False

    def show(self, message: str, seconds: float = SHOW_SECONDS) -> None:
        self.message = message
        self.label.text = message
        w = self.label.width * TEXT_SCALE + 2 * PADDING_X
        h = Text.size * TEXT_SCALE + 2 * PADDING_Y
        self.panel.model = Quad(radius=PANEL_RADIUS, aspect=w / h)
        self.panel.scale = (w, h)
        self._remaining = seconds if math.isfinite(seconds) and seconds > 0 else SHOW_SECONDS
        self._set_alpha(1.0)
        self.enabled = True

    def update(self) -> None:  # Ursina calls this only while the toast is enabled
        dt = time.dt if math.isfinite(time.dt) and time.dt > 0 else 0.0
        self._remaining -= dt
        if self._remaining <= 0:
            self.enabled = False
            return
        self._set_alpha(min(1.0, self._remaining / FADE_SECONDS))

    def _set_alpha(self, alpha: float) -> None:
        self.label.color = color.rgba32(*TEXT_RGB, round(255 * alpha))
        self.panel.color = color.rgba32(*PANEL_RGB, round(PANEL_ALPHA * alpha))
