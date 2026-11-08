"""Menu screen: the title and one card per registered life form (SPEC 3.1). GUI-only.

Each card shows the life form's name, its one-line summary and a Start button. F6 adds Continue
and the save-error state.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ursina import Button, Entity, Text, camera, color
from ursina.models.procedural.quad import Quad

from alife.lifeforms import Lifeform
from alife.ui import destroy_tree

if TYPE_CHECKING:
    from alife.app import App

TITLE = "Artificial Life Simulator"

# Layout and colours: our assumption (style, not biology). Positions are Ursina UI units: the
# window is 1 unit tall, (0, 0) is its centre, and a lower z is nearer the viewer.
TITLE_Y = 0.34
TITLE_SCALE = 2.6
FIRST_CARD_Y = 0.16
CARD_WIDTH = 0.9
CARD_HEIGHT = 0.15
CARD_GAP = 0.03
CARD_PADDING = 0.03
CARD_RADIUS = 0.12  # Quad corner radius, relative to the card height
NAME_SCALE = 1.4
SUMMARY_SCALE = 1.0
BUTTON_WIDTH = 0.16
BUTTON_HEIGHT = 0.06
CARD_RGB = (34, 38, 48)
SUMMARY_RGB = (170, 176, 190)
START_RGB = (40, 110, 220)


class MenuScreen(Entity):
    """The menu UI under ``camera.ui``. ``cards`` maps each life form id to its card."""

    def __init__(self, app: App) -> None:
        super().__init__(parent=camera.ui, name="menu_screen")
        self.app = app
        self.title = Text(
            TITLE, parent=self, origin=(0, 0), y=TITLE_Y, scale=TITLE_SCALE, color=color.white
        )
        self.cards: dict[str, Entity] = {}
        for i, (lifeform_id, cls) in enumerate(app.lifeforms.items()):
            y = FIRST_CARD_Y - i * (CARD_HEIGHT + CARD_GAP)
            self.cards[lifeform_id] = self._make_card(lifeform_id, cls, y)

    def _make_card(self, lifeform_id: str, cls: type[Lifeform], y: float) -> Entity:
        card = Entity(parent=self, y=y, name=f"card_{lifeform_id}")
        card.background = Entity(
            parent=card,
            model=Quad(radius=CARD_RADIUS, aspect=CARD_WIDTH / CARD_HEIGHT),
            scale=(CARD_WIDTH, CARD_HEIGHT),
            z=0.01,  # behind the card's text and button
            color=color.rgba32(*CARD_RGB),
        )
        left = -CARD_WIDTH / 2 + CARD_PADDING
        card.name_text = Text(
            cls.name,
            parent=card,
            origin=(-0.5, 0),
            position=(left, CARD_HEIGHT * 0.17),
            scale=NAME_SCALE,
            color=color.white,
        )
        card.summary_text = Text(
            cls.summary,
            parent=card,
            origin=(-0.5, 0),
            position=(left, -CARD_HEIGHT * 0.2),
            scale=SUMMARY_SCALE,
            color=color.rgba32(*SUMMARY_RGB),
        )
        card.start_button = Button(
            text="Start",
            parent=card,
            x=CARD_WIDTH / 2 - CARD_PADDING - BUTTON_WIDTH / 2,
            scale=(BUTTON_WIDTH, BUTTON_HEIGHT),
            color=color.rgba32(*START_RGB),
            on_click=partial(self.app.start, lifeform_id),
        )
        return card

    def close(self) -> None:
        destroy_tree(self)
