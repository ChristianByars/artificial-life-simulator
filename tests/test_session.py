"""Session: fixed-step clock, speed, pause, reset and snapshot round trip (SPEC 3.4, 3.5)."""

import json
import random

import pytest

from alife.lifeforms import default_params
from alife.lifeforms.demo import DemoLifeform
from alife.session import (
    MAX_TICKS_PER_FRAME,
    SPEEDS,
    TICK_DT,
    TICK_RATE,
    Session,
    new_seed,
    rng_state_from_json,
    rng_state_to_json,
)

FRAME = 1 / 60


def make(seed: int = 7, params: dict | None = None) -> Session:
    return Session(DemoLifeform, seed, params)


def run(s: Session, n: int) -> None:
    for _ in range(n):
        assert s.advance(FRAME) == 1


def test_constants() -> None:
    assert TICK_RATE == 60
    assert TICK_DT == 1 / 60
    assert SPEEDS == (0.25, 0.5, 1, 2, 4, 8)
    assert MAX_TICKS_PER_FRAME == 16


def test_new_seed_range() -> None:
    for _ in range(50):
        assert 0 <= new_seed() < 2**31


def test_params_defaults_overlay_and_clamp() -> None:
    assert make().params == default_params(DemoLifeform)
    assert make(params={"count": 5}).params["count"] == 5
    assert make(params={"count": 10_000}).params["count"] == 300
    assert make(params={"count": -4}).params["count"] == 1
    with pytest.raises(ValueError):
        make(params={"nope": 1})


def test_initial_attributes() -> None:
    s = make()
    assert s.tick == 0
    assert s.speed == 1.0
    assert s.paused is False
    assert s.seed == 7
    assert s.sim_time == 0.0
    assert isinstance(s.rng, random.Random)


def test_one_frame_one_tick() -> None:
    s = make()
    assert s.advance(FRAME) == 1
    assert s.tick == 1
    assert s.sim_time == pytest.approx(TICK_DT)


def test_half_frames_accumulate() -> None:
    s = make()
    assert s.advance(0.5 / 60) == 0
    assert s.advance(0.5 / 60) == 1


def test_speed_multiplies() -> None:
    s = make()
    s.speed = 4.0
    assert s.advance(FRAME) == 4
    s2 = make()
    s2.speed = 0.25
    assert [s2.advance(FRAME) for _ in range(4)] == [0, 0, 0, 1]


def test_long_stall_runs_cap_then_recovers() -> None:
    s = make()
    assert s.advance(5.0) == 16
    assert s.advance(FRAME) == 1


def test_bad_frame_dt_runs_nothing() -> None:
    s = make()
    for dt in (-1, float("nan"), float("inf"), float("-inf")):
        assert s.advance(dt) == 0
    assert s.tick == 0
    assert s.advance(FRAME) == 1  # a bad frame leaves no residue


def test_pause_and_step_once() -> None:
    s = make()
    assert s.step_once() is False
    assert s.tick == 0
    s.toggle_pause()
    assert s.paused
    assert s.advance(1.0) == 0
    assert s.step_once() is True
    assert s.tick == 1
    s.toggle_pause()
    assert not s.paused
    assert s.step_once() is False
    assert s.tick == 1


def test_pause_does_not_bank_time() -> None:
    s = make()
    s.toggle_pause()
    s.advance(1.0)
    s.toggle_pause()
    assert s.advance(FRAME) == 1


def test_speed_bounds() -> None:
    s = make()
    for _ in range(10):
        s.speed_up()
    assert s.speed == 8.0
    for _ in range(10):
        s.speed_down()
    assert s.speed == 0.25
    s.speed_up()
    assert s.speed == 0.5


def test_reset_replays() -> None:
    s = make()
    for _ in range(300):
        s.advance(FRAME)
    a = s.lifeform.get_state()
    s.reset()
    assert s.tick == 0
    for _ in range(300):
        s.advance(FRAME)
    assert s.lifeform.get_state() == a
    assert s.tick == 300


def test_reset_makes_fresh_lifeform() -> None:
    s = make()
    old = s.lifeform
    s.reset()
    assert s.lifeform is not old
    assert s.lifeform.sim.rng is s.rng


def test_new_seed_differs() -> None:
    s = make()
    run(s, 300)
    a = s.lifeform.get_state()
    s.new(s.seed + 1)
    assert s.seed == 8
    assert s.tick == 0
    run(s, 300)
    assert s.lifeform.get_state() != a


def test_snapshot_keys_and_json() -> None:
    s = make()
    run(s, 10)
    s.speed_up()
    snap = s.snapshot()
    assert set(snap) == {"lifeform_id", "seed", "rng_state", "params", "tick", "speed", "state"}
    assert snap["lifeform_id"] == "demo"
    assert snap["tick"] == 10
    assert snap["speed"] == 2.0
    assert isinstance(snap["speed"], float)
    assert isinstance(snap["rng_state"], list)
    json.dumps(snap)


def test_snapshot_roundtrip_json() -> None:
    s = make()
    run(s, 120)
    snap = json.loads(json.dumps(s.snapshot()))
    r = Session.from_snapshot(DemoLifeform, snap)
    assert r.tick == 120
    assert r.seed == s.seed
    assert r.params == s.params
    for _ in range(300):
        s.advance(FRAME)
        r.advance(FRAME)
    assert r.lifeform.get_state() == s.lifeform.get_state()
    assert r.tick == s.tick


def test_from_snapshot_shares_rng_with_lifeform() -> None:
    s = make()
    run(s, 5)
    r = Session.from_snapshot(DemoLifeform, json.loads(json.dumps(s.snapshot())))
    assert r.lifeform.sim.rng is r.rng
    assert r.rng.getstate() == s.rng.getstate()


def test_from_snapshot_unknown_speed_snaps_to_nearest() -> None:
    snap = make().snapshot()
    snap["speed"] = 3.1
    assert Session.from_snapshot(DemoLifeform, snap).speed == 4.0
    snap["speed"] = 100
    assert Session.from_snapshot(DemoLifeform, snap).speed == 8.0
    snap["speed"] = 0.0
    assert Session.from_snapshot(DemoLifeform, snap).speed == 0.25
    snap["speed"] = float("nan")
    assert Session.from_snapshot(DemoLifeform, snap).speed == 1.0


def test_rng_state_json_roundtrip_with_gauss_cache() -> None:
    rng = random.Random(3)
    rng.gauss(0, 1)  # leaves gauss_next set
    state = rng.getstate()
    assert state[2] is not None
    obj = json.loads(json.dumps(rng_state_to_json(state)))
    back = rng_state_from_json(obj)
    assert back == state
    other = random.Random()
    other.setstate(back)
    assert other.gauss(0, 1) == rng.gauss(0, 1)
