"""What each menu card shows: Continue for a good save, or the save's error and New (SPEC §3.1, §4).

Headless (no Ursina): ``alife.screens.menu`` draws these models. A save that reads fine is also
restored once, headless and thrown away (no view is built), so a file whose state the life form
rejects shows its error here instead of failing on Continue (Review Focus 4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from alife.save import MSG_RESTORE_FAILED, SaveError, SaveStore, restore

if TYPE_CHECKING:
    from alife.lifeforms import Lifeform


@dataclass(frozen=True)
class CardModel:
    lifeform_id: str
    name: str
    summary: str
    can_continue: bool  # the save reads and restores
    error: str | None  # card-ready text for a save that exists but cannot be continued


def card_models(registry: Mapping[str, type[Lifeform]], store: SaveStore) -> list[CardModel]:
    """One card per life form, in ``registry`` order. Reads saves; never writes or deletes one."""
    return [_card(lifeform_id, cls, store) for lifeform_id, cls in registry.items()]


def _card(lifeform_id: str, cls: type[Lifeform], store: SaveStore) -> CardModel:
    def card(can_continue: bool, error: str | None) -> CardModel:
        return CardModel(lifeform_id, cls.name, cls.summary, can_continue, error)

    try:
        result = store.read(lifeform_id)
    except OSError:  # e.g. the saves folder cannot be read; read() catches the file's own errors
        return card(False, MSG_RESTORE_FAILED)
    if result.status == "missing":
        return card(False, None)
    if result.status != "ok":
        return card(False, result.message)
    try:
        restore(result.data, {lifeform_id: cls})
    except SaveError:
        return card(False, MSG_RESTORE_FAILED)
    return card(True, None)


def new_needs_confirm(card: CardModel) -> bool:
    """New asks first only over a save that restores: the next save would overwrite it (R20).

    With no save, or one that cannot be continued, there is nothing to lose, so New starts at once.
    """
    return card.can_continue
