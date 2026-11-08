"""F6: menu card models: Continue, the save's error and New (SPEC §3.1, §4; Review Focus 4)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alife.lifeforms.demo import DemoLifeform
from alife.menu_model import CardModel, card_models, new_needs_confirm
from alife.orbit import DEFAULT_POSE
from alife.save import (
    MSG_CORRUPT,
    MSG_RESTORE_FAILED,
    MSG_UNKNOWN_SCHEMA,
    MSG_WRONG_LIFEFORM,
    LoadResult,
    SaveStore,
    build_save,
)
from alife.session import Session

NOW = datetime(2026, 11, 7, 16, 0, 0, tzinfo=timezone(timedelta(hours=-8)))
REGISTRY = {"demo": DemoLifeform}


@pytest.fixture
def store(tmp_path: Path) -> SaveStore:
    return SaveStore(tmp_path / "saves")


def good_save() -> dict:
    session = Session(DemoLifeform, 11)
    for _ in range(20):
        session.advance(1 / 60)
    return json.loads(json.dumps(build_save(session, DEFAULT_POSE, NOW)))


def write_raw(store: SaveStore, text: str) -> Path:
    path = store.path("demo")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def only_card(store: SaveStore) -> CardModel:
    cards = card_models(REGISTRY, store)
    assert len(cards) == 1
    return cards[0]


def test_card_without_save(store: SaveStore) -> None:
    card = only_card(store)
    assert card == CardModel(
        lifeform_id="demo",
        name=DemoLifeform.name,
        summary=DemoLifeform.summary,
        can_continue=False,
        error=None,
    )


def test_card_with_good_save(store: SaveStore) -> None:
    store.write("demo", good_save())
    card = only_card(store)
    assert card.can_continue is True
    assert card.error is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"schema_version": 1, "trunc', MSG_CORRUPT),  # truncated
        ("[]", MSG_CORRUPT),
        (json.dumps({**good_save(), "schema_version": 2}), MSG_UNKNOWN_SCHEMA),
        (json.dumps({**good_save(), "lifeform_id": "cell"}), MSG_WRONG_LIFEFORM),
    ],
    ids=["truncated", "not_an_object", "unknown_schema", "wrong_lifeform"],
)
def test_card_for_corrupt_save(store: SaveStore, text: str, message: str) -> None:
    write_raw(store, text)
    result: LoadResult = store.read("demo")
    card = only_card(store)
    assert card.can_continue is False
    assert card.error == result.message == message


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": {"walkers": "gone"}},  # set_state raises
        {"state": {}},
        {"tick": -5},
        {"camera": {"position": [0, 1], "rotation": [45, 0, 0]}},
        {"rng_state": [3, [1, 2, 3], None]},
    ],
    ids=["state_rejected", "state_empty", "tick_negative", "camera_bad", "rng_state_bad"],
)
def test_card_for_stale_state(store: SaveStore, overrides: dict) -> None:  # Review Focus 4
    store.write("demo", {**good_save(), **overrides})
    assert store.read("demo").status == "ok"  # the file itself reads fine
    card = only_card(store)
    assert card.can_continue is False
    assert card.error == MSG_RESTORE_FAILED
    assert card.error


def test_cards_never_touch_the_file(store: SaveStore) -> None:
    path = write_raw(store, '{"schema_version": 1, "trunc')
    card_models(REGISTRY, store)
    assert path.read_text() == '{"schema_version": 1, "trunc'  # overwritten only by a save


class NoViewDemo(DemoLifeform):
    def build_view(self, root: object) -> None:
        raise AssertionError("card_models built a view")


def test_dry_run_restore_builds_no_view(store: SaveStore) -> None:
    store.write("demo", good_save())
    [card] = card_models({"demo": NoViewDemo}, store)
    assert card.can_continue is True


class OtherLifeform(DemoLifeform):
    id = "other"
    name = "Another one"
    summary = "Second card."


def test_cards_follow_registry_order(store: SaveStore) -> None:
    store.write("demo", good_save())
    cards = card_models({"other": OtherLifeform, "demo": DemoLifeform}, store)
    assert [c.lifeform_id for c in cards] == ["other", "demo"]
    assert [c.name for c in cards] == ["Another one", DemoLifeform.name]
    assert [c.can_continue for c in cards] == [False, True]


class BrokenStore(SaveStore):
    def read(self, lifeform_id: str) -> LoadResult:
        raise PermissionError(13, "Permission denied")


def test_unreadable_save_dir_shows_error(tmp_path: Path) -> None:
    [card] = card_models(REGISTRY, BrokenStore(tmp_path))
    assert card.can_continue is False
    assert card.error == MSG_RESTORE_FAILED


# --- R20: New asks first only when a save that restores would be lost ---------------------------


@pytest.mark.parametrize(
    ("can_continue", "error", "expected"),
    [
        (True, None, True),  # a good save: the next save would overwrite it
        (False, None, False),  # no save: nothing to lose
        (False, MSG_CORRUPT, False),  # a bad save: nothing to lose
        (False, MSG_RESTORE_FAILED, False),
    ],
    ids=["good_save", "no_save", "corrupt_save", "stale_save"],
)
def test_new_needs_confirm(can_continue: bool, error: str | None, expected: bool) -> None:
    card = CardModel("demo", "Demo", "Summary.", can_continue, error)
    assert new_needs_confirm(card) is expected


def test_new_needs_confirm_from_real_cards(store: SaveStore) -> None:
    assert new_needs_confirm(only_card(store)) is False
    store.write("demo", good_save())
    assert new_needs_confirm(only_card(store)) is True
    store.write("demo", {**good_save(), "state": {}})
    assert new_needs_confirm(only_card(store)) is False
