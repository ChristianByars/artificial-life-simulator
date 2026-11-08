"""Confirm dialog: a question with Yes and Cancel (SPEC 4: Reset and New ask first). GUI-only.

The dialog draws itself and handles its buttons; keys reach it through :meth:`handle_key`, called
by the screen that owns it. The screen keeps the dialog modal: while one is open it passes every
key here and lets nothing else (camera, sim) see it. Closing destroys the dialog, then runs the
callback, so the callback may open another dialog.
"""

from __future__ import annotations

from collections.abc import Callable

from ursina import Button, Entity, Text, color
from ursina.models.procedural.quad import Quad

from alife.keys import dialog_action
from alife.ui import destroy_tree
from alife.ui_text import CONFIRM_CANCEL, CONFIRM_HINT, CONFIRM_YES

# Layout and colours: our assumption (style, not biology). Ursina UI units: the window is 1 unit
# tall and (0, 0) is its centre; a lower z is nearer the viewer.
DIALOG_Z = -1.0  # in front of the HUD and the toast
BACKDROP_SIZE = 100  # covers the window at any aspect ratio
MIN_WIDTH = 0.6
HEIGHT = 0.24
PADDING = 0.06
PANEL_RADIUS = 0.1  # Quad corner radius, relative to the panel height
MESSAGE_Y = 0.055
MESSAGE_SCALE = 1.15
BUTTON_Y = -0.025
BUTTON_WIDTH = 0.16
BUTTON_HEIGHT = 0.055
BUTTON_GAP = 0.03
HINT_Y = -0.083
HINT_SCALE = 0.85
BACKDROP_RGBA = (0, 0, 0, 130)
PANEL_RGB = (38, 44, 58)
MESSAGE_RGB = (255, 255, 255)
HINT_RGB = (150, 156, 172)
YES_RGB = (40, 110, 220)
CANCEL_RGB = (72, 78, 94)


class ConfirmDialog(Entity):
    """``message`` with Yes / Cancel. Exactly one of the callbacks runs, once."""

    def __init__(
        self,
        message: str,
        on_confirm: Callable[[], None],
        on_cancel: Callable[[], None],
        parent: Entity,
    ) -> None:
        super().__init__(parent=parent, name="confirm_dialog", z=DIALOG_Z)
        self.message = message
        self.closed = False
        self._on_confirm = on_confirm
        self._on_cancel = on_cancel
        # Dims the scene; no collider, so it does not take clicks (the screen ignores them).
        self.backdrop = Entity(
            parent=self,
            model="quad",
            scale=BACKDROP_SIZE,
            z=0.02,
            color=color.rgba32(*BACKDROP_RGBA),
        )
        self.text = Text(
            message,
            parent=self,
            origin=(0, 0),
            y=MESSAGE_Y,
            scale=MESSAGE_SCALE,
            color=color.rgba32(*MESSAGE_RGB),
            use_tags=False,
        )
        width = max(MIN_WIDTH, self.text.width * MESSAGE_SCALE + 2 * PADDING)
        self.panel = Entity(
            parent=self,
            model=Quad(radius=PANEL_RADIUS, aspect=width / HEIGHT),
            scale=(width, HEIGHT),
            z=0.01,
            color=color.rgba32(*PANEL_RGB),
        )
        offset = (BUTTON_WIDTH + BUTTON_GAP) / 2
        self.yes_button = Button(
            text=CONFIRM_YES,
            parent=self,
            position=(-offset, BUTTON_Y),
            scale=(BUTTON_WIDTH, BUTTON_HEIGHT),
            color=color.rgba32(*YES_RGB),
            on_click=self.confirm,
        )
        self.cancel_button = Button(
            text=CONFIRM_CANCEL,
            parent=self,
            position=(offset, BUTTON_Y),
            scale=(BUTTON_WIDTH, BUTTON_HEIGHT),
            color=color.rgba32(*CANCEL_RGB),
            on_click=self.cancel,
        )
        self.hint = Text(
            CONFIRM_HINT,
            parent=self,
            origin=(0, 0),
            y=HINT_Y,
            scale=HINT_SCALE,
            color=color.rgba32(*HINT_RGB),
            use_tags=False,
        )

    def handle_key(self, key: str) -> None:
        """Enter confirms, Esc cancels; any other key does nothing."""
        action = dialog_action(key)
        if action == "confirm":
            self.confirm()
        elif action == "cancel":
            self.cancel()

    def confirm(self) -> None:
        self._close(self._on_confirm)

    def cancel(self) -> None:
        self._close(self._on_cancel)

    def _close(self, then: Callable[[], None]) -> None:
        if self.closed:  # a key and a click in the same frame must not run two callbacks
            return
        self.closed = True
        destroy_tree(self)
        then()
