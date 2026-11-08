"""What each menu card shows: Continue for a good save, or the save's error and New (SPEC §3.1, §4).

Also whether New (and ``--lifeform ID``) must ask first before a fresh start (R20, R25, M5).

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
    # The save is from a newer app version (``unknown_schema``): this version can't open it, but
    # a newer one could, so New asks first (M5). A damaged save leaves this False.
    newer_version: bool = False


def card_models(registry: Mapping[str, type[Lifeform]], store: SaveStore) -> list[CardModel]:
    """One card per life form, in ``registry`` order. Reads saves; never writes or deletes one."""
    return [card_model(lifeform_id, cls, store) for lifeform_id, cls in registry.items()]


def card_model(lifeform_id: str, cls: type[Lifeform], store: SaveStore) -> CardModel:
    """The card for one life form, as :func:`card_models` builds it."""

    def card(can_continue: bool, error: str | None, newer_version: bool = False) -> CardModel:
        return CardModel(lifeform_id, cls.name, cls.summary, can_continue, error, newer_version)

    try:
        result = store.read(lifeform_id)
    except OSError:  # e.g. the saves folder cannot be read; read() catches the file's own errors
        return card(False, MSG_RESTORE_FAILED)
    if result.status == "missing":
        return card(False, None)
    if result.status != "ok":
        return card(False, result.message, newer_version=result.status == "unknown_schema")
    try:
        restore(result.data, {lifeform_id: cls})
    except SaveError:
        return card(False, MSG_RESTORE_FAILED)
    return card(True, None)


def new_needs_confirm(card: CardModel) -> bool:
    """New asks first over a save the next save would overwrite: one that restores (R20), or one
    from a newer app version, which that version could still open (M5).

    With no save, or a damaged or unreadable one, there is nothing to lose, so New starts at once.
    """
    return card.can_continue or card.newer_version


def cli_start_needs_confirm(card: CardModel, test_run: bool) -> bool:
    """``--lifeform ID`` asks first exactly when that card's New would (R25).

    A test run (``--smoke``, ``--screenshot``) starts at once, as before: without ``ALIFE_DATA_DIR``
    it saves to a throwaway folder (R21), so there is no save to protect.
    """
    return not test_run and new_needs_confirm(card)
