"""Determinism across save/resume and reset, for every registered life form (SPEC §3.5, M3).

Parametrized over ``registry()``: the cell colony joins automatically when it registers (Plan 2).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from alife.lifeforms import registry
from alife.orbit import DEFAULT_POSE
from alife.save import build_save, restore
from alife.session import TICK_DT, Session

SEED = 20261107
NOW = datetime(2026, 11, 7, 16, 0, 0, tzinfo=timezone(timedelta(hours=-8)))
LIFEFORM_IDS = sorted(registry())


def run(session: Session, ticks: int) -> None:
    for _ in range(ticks):
        assert session.advance(TICK_DT) == 1


def as_json(session: Session) -> str:
    """The session as it would be saved (state, rng, tick, seed, params, speed)."""
    return json.dumps(session.snapshot(), sort_keys=True, allow_nan=False)


def test_registry_is_not_empty() -> None:
    assert "demo" in LIFEFORM_IDS


@pytest.mark.parametrize("lid", LIFEFORM_IDS)
def test_straight_equals_saved_and_resumed(lid: str) -> None:
    cls = registry()[lid]
    straight = Session(cls, SEED)
    start = as_json(straight)
    run(straight, 600)
    assert as_json(straight) != start  # the run changed something

    first = Session(cls, SEED)
    run(first, 300)
    data = json.loads(json.dumps(build_save(first, DEFAULT_POSE, NOW), allow_nan=False))
    resumed, _pose = restore(data, registry())
    run(resumed, 300)

    assert resumed.tick == straight.tick == 600
    assert as_json(resumed) == as_json(straight)


@pytest.mark.parametrize("lid", LIFEFORM_IDS)
def test_reset_replays(lid: str) -> None:
    session = Session(registry()[lid], SEED)
    run(session, 600)
    first = as_json(session)
    session.reset()
    assert session.tick == 0
    run(session, 600)
    assert as_json(session) == first
