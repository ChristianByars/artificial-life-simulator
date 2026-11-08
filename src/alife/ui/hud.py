"""Top-left HUD (SPEC 5). GUI-only: draws the lines ``alife.ui_text.hud_lines`` returns.

One ``Text`` per line, on a translucent panel. Line 1 (the life form's name) is larger; the
``PAUSED`` line is drawn in amber so the paused state stands out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ursina import Entity, Text, color, destroy, window
from ursina.models.procedural.quad import Quad

from alife.ui_text import PAUSED

# Layout and colours: our assumption (style, not biology). Ursina UI units: the window is 1 unit
# tall and (0, 0) is its centre; a lower z is nearer the viewer.
MARGIN = 0.02  # from the window's top-left corner
PADDING = 0.016  # inside the panel
NAME_SCALE = 1.2
LINE_SCALE = 1.0
LINE_PITCH = 1.45  # distance between line tops, in text heights
MIN_WIDTH = 0.3
WIDTH_STEP = 0.04  # the panel widens in steps, so it does not twitch as digits change
PANEL_RADIUS = 0.08  # Quad corner radius, relative to the panel's shorter side
NAME_RGB = (255, 255, 255)
TEXT_RGB = (205, 210, 222)
PAUSED_RGB = (255, 196, 64)
PANEL_RGBA = (10, 12, 18, 170)


class Hud(Entity):
    """``show(lines)`` once per frame; Texts are rebuilt only for lines that changed."""

    def __init__(self, parent: Entity) -> None:
        super().__init__(parent=parent, name="hud")
        self.panel = Entity(parent=self, z=0.01, color=color.rgba32(*PANEL_RGBA))
        self.texts: list[Text] = []
        self.lines: list[str] = []
        self._panel_size: tuple[float, float] | None = None

    def show(self, lines: Sequence[str]) -> None:
        # Re-anchored every frame: the left edge moves when the window's aspect ratio changes.
        self.position = (-window.aspect_ratio / 2 + MARGIN, 0.5 - MARGIN, 0)
        lines = list(lines)
        if lines == self.lines:
            return
        while len(self.texts) < len(lines):
            self.texts.append(Text("", parent=self, origin=(-0.5, 0.5), use_tags=False))
        while len(self.texts) > len(lines):
            destroy(self.texts.pop())
        y = -PADDING
        width = 0.0
        for i, (text, line) in enumerate(zip(self.texts, lines, strict=True)):
            old = self.lines[i] if i < len(self.lines) else None
            scale = NAME_SCALE if i == 0 else LINE_SCALE
            if line != old:
                text.text = line
                rgb = NAME_RGB if i == 0 else PAUSED_RGB if line == PAUSED else TEXT_RGB
                text.color = color.rgba32(*rgb)
            text.scale = scale
            text.position = (PADDING, y, 0)
            width = max(width, text.width * scale)
            y -= Text.size * scale * LINE_PITCH
        self.lines = lines
        # The last line's pitch includes the gap below it; keep only its text height.
        height = -y + PADDING - Text.size * LINE_SCALE * (LINE_PITCH - 1)
        self._fit_panel(width + 2 * PADDING, height)

    def _fit_panel(self, width: float, height: float) -> None:
        w = max(MIN_WIDTH, math.ceil(width / WIDTH_STEP) * WIDTH_STEP)
        h = round(height, 4)
        if self._panel_size == (w, h):
            return
        self._panel_size = (w, h)
        self.panel.model = Quad(radius=PANEL_RADIUS, aspect=w / h)
        self.panel.scale = (w, h)
        self.panel.position = (w / 2, -h / 2, 0.01)  # the Quad is centred; anchor its top-left
