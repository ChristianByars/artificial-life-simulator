"""Menu screen: the title and one card per registered life form (SPEC 3.1). GUI-only.

Each card shows the life form's name and its one-line summary, then, from ``alife.menu_model``:
- no save: Start;
- a save that reads and restores: Continue (restores it) and New, which asks first (R20): the next
  save would overwrite the old one;
- a save that does not: its error, in amber, and New. New asks first over a save from a newer
  version (M5), which that version could still open, and starts at once over a damaged one. The
  bad file stays until the next save.
New always starts with a new seed; ``--seed`` is only for the start the command line makes (#31).
``--lifeform ID`` over a protected save opens this menu with that card's dialog up (R25).
While the confirm dialog is open the menu passes every key to it, and the cards take no clicks.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from ursina import Button, Entity, Text, camera, color
from ursina.models.procedural.quad import Quad

from alife.menu_model import CardModel, card_models, new_needs_confirm
from alife.session import new_seed
from alife.ui import destroy_tree
from alife.ui.confirm import ConfirmDialog
from alife.ui_text import CONFIRM_NEW

if TYPE_CHECKING:
    from alife.app import App

TITLE = "Artificial Life Simulator"

# Layout and colours: our assumption (style, not biology). Positions are Ursina UI units: the
# window is 1 unit tall, (0, 0) is its centre, and a lower z is nearer the viewer.
TITLE_Y = 0.34
TITLE_SCALE = 2.6
FIRST_CARD_Y = 0.16
CARD_WIDTH = 1.05  # fits the summary beside two buttons (Continue, New)
CARD_HEIGHT = 0.15
CARD_GAP = 0.03
CARD_PADDING = 0.03
CARD_RADIUS = 0.12  # Quad corner radius, relative to the card height
NAME_SCALE = 1.4
SUMMARY_SCALE = 1.0
BUTTON_WIDTH = 0.16
BUTTON_HEIGHT = 0.06
BUTTON_GAP = 0.02
ERROR_SCALE = 0.95
CARD_RGB = (34, 38, 48)
SUMMARY_RGB = (170, 176, 190)
START_RGB = (40, 110, 220)  # the card's main action: Start, Continue, or New after an error
SECONDARY_RGB = (70, 76, 92)  # New next to Continue
ERROR_RGB = (255, 196, 64)  # amber, as the HUD's PAUSED

START_LABEL = "Start"
NEW_LABEL = "New"
CONTINUE_LABEL = "Continue"


class MenuScreen(Entity):
    """The menu UI under ``camera.ui``. ``cards`` maps each life form id to its card.

    A card has ``name_text``, ``summary_text``, ``start_button`` (a fresh start: "Start", or
    "New" once a save exists), ``continue_button`` (None unless the save restores) and
    ``error_text`` (None unless the save cannot be continued).
    """

    def __init__(self, app: App) -> None:
        super().__init__(parent=camera.ui, name="menu_screen")
        self.app = app
        self.title = Text(
            TITLE, parent=self, origin=(0, 0), y=TITLE_Y, scale=TITLE_SCALE, color=color.white
        )
        self.cards: dict[str, Entity] = {}
        self.dialog: ConfirmDialog | None = None
        for i, model in enumerate(card_models(app.lifeforms, app.store)):
            y = FIRST_CARD_Y - i * (CARD_HEIGHT + CARD_GAP)
            self.cards[model.lifeform_id] = self._make_card(model, y)

    def _make_card(self, model: CardModel, y: float) -> Entity:
        lifeform_id = model.lifeform_id
        has_save = model.can_continue or model.error is not None
        card = Entity(parent=self, y=y, name=f"card_{lifeform_id}")
        card.background = Entity(
            parent=card,
            model=Quad(radius=CARD_RADIUS, aspect=CARD_WIDTH / CARD_HEIGHT),
            scale=(CARD_WIDTH, CARD_HEIGHT),
            z=0.01,  # behind the card's text and button
            color=color.rgba32(*CARD_RGB),
        )
        left = -CARD_WIDTH / 2 + CARD_PADDING
        # Text rows as fractions of the card height: two rows, or three with an error line.
        rows = (0.17, -0.2) if model.error is None else (0.28, -0.02, -0.29)
        card.name_text = Text(
            model.name,
            parent=card,
            origin=(-0.5, 0),
            position=(left, CARD_HEIGHT * rows[0]),
            scale=NAME_SCALE,
            color=color.white,
        )
        card.summary_text = Text(
            model.summary,
            parent=card,
            origin=(-0.5, 0),
            position=(left, CARD_HEIGHT * rows[1]),
            scale=SUMMARY_SCALE,
            color=color.rgba32(*SUMMARY_RGB),
        )
        card.error_text = None
        if model.error is not None:
            card.error_text = Text(
                model.error,
                parent=card,
                origin=(-0.5, 0),
                position=(left, CARD_HEIGHT * rows[2]),
                scale=ERROR_SCALE,
                color=color.rgba32(*ERROR_RGB),
                use_tags=False,
            )
        # Buttons from the right edge: the main action outermost.
        right = CARD_WIDTH / 2 - CARD_PADDING - BUTTON_WIDTH / 2
        card.continue_button = None
        if model.can_continue:
            card.continue_button = self._button(
                card, CONTINUE_LABEL, right, START_RGB, partial(self._continue, lifeform_id)
            )
            right -= BUTTON_WIDTH + BUTTON_GAP
        card.start_button = self._button(
            card,
            NEW_LABEL if has_save else START_LABEL,
            right,
            SECONDARY_RGB if model.can_continue else START_RGB,
            partial(self._new, model),
        )
        return card

    # card actions and the confirm dialog

    def _continue(self, lifeform_id: str) -> None:
        if self.dialog is None:  # no card action while the dialog is open
            self.app.resume(lifeform_id)

    def _new(self, model: CardModel) -> None:
        if self.dialog is not None:
            return
        if not new_needs_confirm(model):
            self.app.start(model.lifeform_id, new_seed())
            return
        self.ask_new(model.lifeform_id)

    def ask_new(self, lifeform_id: str, seed: int | None = None) -> None:
        """Open the ``CONFIRM_NEW`` dialog over ``lifeform_id``'s card (R20).

        Yes starts it fresh with ``seed``, or a new seed when None (the card's New, #31). Cancel or
        Esc closes the dialog and stays on the menu. ``App`` calls this for ``--lifeform ID`` over
        a protected save, with ``--seed`` (R25).
        """
        if self.dialog is not None:
            return
        self.dialog = ConfirmDialog(
            CONFIRM_NEW,
            on_confirm=partial(self._confirmed_new, lifeform_id, seed),
            on_cancel=self._dialog_closed,
            parent=self,
        )
        self._set_cards_clickable(False)

    def _confirmed_new(self, lifeform_id: str, seed: int | None) -> None:
        self._dialog_closed()
        self.app.start(lifeform_id, new_seed() if seed is None else seed)

    def _dialog_closed(self) -> None:
        self.dialog = None
        self._set_cards_clickable(True)

    def _set_cards_clickable(self, clickable: bool) -> None:
        """Ursina's mouse fires ``on_click`` for any hovered collider, whoever ate the key, and
        the dialog's backdrop has no collider; so the card buttons stop being hoverable."""
        for card in self.cards.values():
            for button in (card.start_button, card.continue_button):
                if button is not None:
                    button.collision = clickable

    def input(self, key: str) -> bool:
        """While the dialog is open: Enter / Esc go to it and every key is eaten (True)."""
        if self.dialog is None:
            return False
        self.dialog.handle_key(key)
        return True

    def _button(
        self,
        card: Entity,
        label: str,
        x: float,
        rgb: tuple[int, int, int],
        on_click: Callable[[], object],
    ) -> Button:
        return Button(
            text=label,
            parent=card,
            x=x,
            scale=(BUTTON_WIDTH, BUTTON_HEIGHT),
            color=color.rgba32(*rgb),
            on_click=on_click,
        )

    def close(self) -> None:
        destroy_tree(self)
