"""Cell, organelles and the colony sim (task S5, SPEC §6.1)."""

from __future__ import annotations

import json
import math
import random

import pytest

from alife.geometry import DISH_RADIUS, Vec2
from alife.lifeforms import OUR_ASSUMPTION
from alife.lifeforms.cell.cycle import DEFAULT_TIME_SCALE, MPF_THRESHOLD
from alife.lifeforms.cell.params import (
    CLICK_AMOUNT,
    DYING_SECONDS,
    ENERGY_REF,
    MAX_RADIUS,
    PARAMS,
    REFERENCE_PERIOD,
    CellParams,
)
from alife.lifeforms.cell.sim import (
    Cell,
    CellColony,
    CellPhase,
    Died,
    Divided,
    Environment,
    Event,
)
from alife.session import TICK_DT

FAST = 10 * DEFAULT_TIME_SCALE  # cycles 10x faster: one period is 6 s, 360 ticks
PERIOD_TICKS_FAST = round(REFERENCE_PERIOD / FAST / TICK_DT)
STAT_KEYS = {
    "population",
    "births",
    "deaths",
    "mean_energy",
    "oldest_generation",
    "interphase",
    "mitosis",
}


def make_params(**overrides: float) -> dict[str, float]:
    return {**{key: float(p.default) for key, p in PARAMS.items()}, **overrides}


def make_colony(seed: int = 1, **overrides: float) -> CellColony:
    return CellColony(random.Random(seed), make_params(**overrides))


def run(colony: CellColony, ticks: int) -> list[Event]:
    events: list[Event] = []
    for _ in range(ticks):
        events += colony.step(TICK_DT)
    return events


def state_json(colony: CellColony) -> dict:
    """get_state after a JSON round trip (what a save file holds)."""
    return json.loads(json.dumps(colony.get_state(), allow_nan=False))


def living(colony: CellColony) -> list[Cell]:
    return [colony.cells[i] for i in sorted(colony.cells) if not colony.cells[i].is_dying]


def inside_dish(cell: Cell) -> bool:
    return cell.position.length() <= DISH_RADIUS - cell.radius + 1e-9


def finite(p: Vec2) -> bool:
    return math.isfinite(p.x) and math.isfinite(p.y)


# params


def test_params_cover_spec_and_are_our_assumptions():
    assert set(PARAMS) == {
        "initial_cells",
        "nutrient_regen",
        "diffusion",
        "upkeep_cost",
        "time_scale",
    }
    assert all(p.source == OUR_ASSUMPTION for p in PARAMS.values())
    assert PARAMS["time_scale"].default == DEFAULT_TIME_SCALE
    # the validation tests run at 10x the default; that must be a legal setting
    assert PARAMS["time_scale"].min <= FAST <= PARAMS["time_scale"].max
    assert PARAMS["nutrient_regen"].min == 0.0  # the starvation test turns regrowth off


def test_cell_params_from_dict():
    p = CellParams.from_dict(make_params(initial_cells=7.0, time_scale=FAST))
    assert p.initial_cells == 7 and isinstance(p.initial_cells, int)
    assert p.time_scale == FAST
    assert p.upkeep_cost == PARAMS["upkeep_cost"].default
    assert CellParams.from_dict({}) == CellParams.from_dict(make_params())  # defaults fill in
    for bad in (
        {"no_such_param": 1.0},
        {"upkeep_cost": math.nan},
        {"diffusion": -1.0},
        {"time_scale": 0.0},
        {"initial_cells": "many"},
    ):
        with pytest.raises(ValueError):
            CellParams.from_dict(bad)


# colony


def test_seeded_cells_are_unsynchronised_adults():
    colony = make_colony(seed=5, initial_cells=12)
    cells = [colony.cells[i] for i in sorted(colony.cells)]
    assert [c.id for c in cells] == list(range(1, 13))  # one counter, starting at 1
    for c in cells:
        assert c.radius == MAX_RADIUS
        assert c.energy == ENERGY_REF
        assert c.generation == 0
        assert inside_dish(c)
    # each starts at its own point of the cycle, spread over most of a period
    times = sorted(c.cycle.t for c in cells)
    assert len(set(times)) == len(times)
    assert times[-1] - times[0] > 0.5 * REFERENCE_PERIOD


def test_population_invariant():
    colony = make_colony(seed=3, initial_cells=20, time_scale=FAST, nutrient_regen=0.0)
    colony.env.nutrient.fill(0.2)  # a little food: some cells divide, then all starve
    initial, births, deaths, removed = 20, 0, 0, 0
    ids = set(colony.cells)
    for _ in range(4 * PERIOD_TICKS_FAST):
        events = colony.step(TICK_DT)
        parents = {e.parent.id for e in events if isinstance(e, Divided)}
        births += len(parents)
        deaths += sum(isinstance(e, Died) for e in events)
        removed += len(ids - set(colony.cells) - parents)
        ids = set(colony.cells)
        dying = sum(c.is_dying for c in colony.cells.values())
        stats = colony.stats()
        assert stats["population"] + dying == initial + births - removed
        assert (stats["births"], stats["deaths"]) == (births, deaths)
    assert births > 0 and deaths > 0 and removed > 0  # the run exercised every path


def test_colony_deterministic():
    a = make_colony(seed=7, time_scale=FAST)
    b = make_colony(seed=7, time_scale=FAST)
    run(a, 600)
    run(b, 600)
    assert a.stats()["births"] > 0  # divisions (rng angle and spread) are covered
    assert state_json(a) == state_json(b)


def test_colony_restore_exact():
    rng = random.Random(11)
    original = CellColony(rng, make_params(time_scale=FAST))
    run(original, 300)
    saved = state_json(original)
    saved_rng = json.loads(json.dumps(rng.getstate()))
    births_at_save = original.stats()["births"]

    # Rebuild as Session.from_snapshot does: Random(seed) -> init -> set_state -> setstate.
    rng2 = random.Random(11)
    restored = CellColony(rng2, make_params(time_scale=FAST))
    restored.set_state(saved)
    rng2.setstate((saved_rng[0], tuple(saved_rng[1]), saved_rng[2]))
    assert state_json(restored) == saved
    for _ in range(6):
        run(original, 50)
        run(restored, 50)
        assert state_json(restored) == state_json(original)
    assert original.stats()["births"] > births_at_save  # divisions happened after the restore


def test_set_state_rejects_malformed_state():
    colony = make_colony(seed=2, initial_cells=3)
    good = state_json(colony)
    stale_cycle = json.loads(json.dumps(good))
    stale_cycle["cells"][0]["cycle"]["species"] = {"X": 1.0}
    missing_field = json.loads(json.dumps(good))
    del missing_field["cells"][1]["since_division"]
    duplicate = json.loads(json.dumps(good))
    duplicate["cells"][1]["id"] = duplicate["cells"][0]["id"]
    for bad in (
        {},
        {**good, "cells": None},
        {**good, "env": {"shape": [2, 2]}},
        {**good, "time": "noon"},
        {**good, "next_id": 1},  # below the saved ids
        stale_cycle,
        missing_field,
        duplicate,
        [],
    ):
        with pytest.raises(ValueError):
            colony.set_state(bad)
    assert state_json(colony) == good  # a rejected state changes nothing


def test_stats_keys():
    colony = make_colony(seed=4, time_scale=FAST)
    run(colony, 200)
    stats = colony.stats()
    assert set(stats) == STAT_KEYS
    alive = living(colony)
    assert stats["population"] == len(alive)
    assert stats["interphase"] + stats["mitosis"] == len(alive)
    assert stats["mean_energy"] == pytest.approx(sum(c.energy for c in alive) / len(alive))
    assert stats["oldest_generation"] == max(c.generation for c in alive)
    assert all(isinstance(v, int | float) for v in stats.values())


def test_add_nutrient_raises_local_food():
    colony = make_colony(seed=1, initial_cells=1)
    env = colony.env
    here, far = Vec2(-5.0, 3.0), Vec2(5.0, -5.0)
    before_here = env.nutrient[env.index(here)]
    before_far = env.nutrient[env.index(far)]
    total = env.total()
    colony.add_nutrient(here)
    assert colony.env.nutrient[env.index(here)] > before_here
    assert colony.env.nutrient[env.index(far)] == before_far
    assert colony.env.total() == pytest.approx(total + CLICK_AMOUNT)


# cells


def test_cell_dict_roundtrip():
    colony = make_colony(seed=6, time_scale=FAST)
    run(colony, PERIOD_TICKS_FAST)  # some cells have divided, grown and spread
    cell = max(colony.cells.values(), key=lambda c: (c.generation, c.id))
    assert cell.generation >= 1
    saved = json.loads(json.dumps(cell.to_dict(), allow_nan=False))
    twin = Cell.from_dict(saved)
    assert twin.to_dict() == cell.to_dict()
    # stepping the twin against a copy of the same field gives a bitwise-equal cell
    env_a = colony.env
    env_b = Environment.from_dict(env_a.to_dict())
    for _ in range(100):
        cell.step(env_a, TICK_DT, colony.params)
        twin.step(env_b, TICK_DT, colony.params)
    assert twin.to_dict() == cell.to_dict()
    for key in ("energy", "cycle", "mitochondria"):
        broken = json.loads(json.dumps(saved))
        del broken[key]
        with pytest.raises(ValueError):
            Cell.from_dict(broken)


def test_phase_follows_threshold():
    colony = make_colony(seed=8, time_scale=FAST)
    seen = set()
    for _ in range(PERIOD_TICKS_FAST):
        colony.step(TICK_DT)
        for cell in living(colony):
            assert (cell.phase is CellPhase.MITOSIS) == (cell.cycle.mpf > MPF_THRESHOLD)
            assert cell.mpf_high == (cell.phase is CellPhase.MITOSIS)
            seen.add(cell.phase)
    assert seen == {CellPhase.INTERPHASE, CellPhase.MITOSIS}


def test_daughters_separate():
    # Review Focus 2. A cell pressed against the rim splits in every direction: the
    # daughters are finite, distinct and inside the dish.
    rim = Vec2(DISH_RADIUS - MAX_RADIUS, 0.0)
    for seed in range(24):
        cell = make_colony(seed=seed, initial_cells=1).cells[1]
        cell.position = rim
        a, b = cell.divide((2, 3), random.Random(seed))
        assert (a.id, b.id) == (2, 3)
        for d in (a, b):
            assert finite(d.position) and inside_dish(d)
            assert d.radius == pytest.approx(cell.radius / math.sqrt(2))
            assert d.energy == cell.energy / 2
            assert d.generation == cell.generation + 1
            assert d.cycle is not cell.cycle
        assert a.position != b.position

    # In the colony, natural divisions' daughters push apart without NaN. Brownian noise is
    # as large as the push over a few ticks, so compare the mean over every division.
    colony = make_colony(seed=2, time_scale=FAST)
    pairs: list[tuple[Cell, Cell, float]] = []
    last_id = max(colony.cells)
    for _ in range(PERIOD_TICKS_FAST):
        for event in colony.step(TICK_DT):
            if not isinstance(event, Divided):
                continue
            a, b = event.daughters
            assert event.parent.id not in colony.cells
            assert colony.cells[a.id] is a and colony.cells[b.id] is b
            assert (a.id, b.id) == (last_id + 1, last_id + 2)  # the next two ids
            last_id = b.id
            pairs.append((a, b, (a.position - b.position).length()))
    run(colony, 30)
    assert len(pairs) >= 5
    for cell in colony.cells.values():
        assert finite(cell.position) and inside_dish(cell)
    widened = [
        (a.position - b.position).length() - gap
        for a, b, gap in pairs
        if colony.cells.get(a.id) is a and colony.cells.get(b.id) is b
    ]
    assert len(widened) >= 5
    assert sum(widened) / len(widened) > 0.0


def test_coincident_cells_separate():
    # Review Focus 2: two cells on the same spot (e.g. a click-spawn on a cell).
    def run_pair() -> tuple[Vec2, Vec2]:
        colony = make_colony(seed=9, initial_cells=2)
        a, b = colony.cells[1], colony.cells[2]
        a.position = b.position = Vec2(1.0, 1.0)
        colony.step(TICK_DT)
        return a.position, b.position

    pa, pb = run_pair()
    assert finite(pa) and finite(pb) and pa != pb
    assert run_pair() == (pa, pb)


def test_cell_at_rim_feeds():
    # Review Focus 1: a cell at the clamp limit feeds from the grid edge without error.
    colony = make_colony(seed=1, initial_cells=1)
    cell = colony.cells[1]
    cell.position = Vec2(9.65, 0.0)  # DISH_RADIUS - MAX_RADIUS
    cell.energy = ENERGY_REF
    run(colony, 60)
    assert cell.energy > ENERGY_REF
    assert inside_dish(cell)
    assert colony.env.nutrient[colony.env.index(cell.position)] < 1.0  # it ate from the edge


def test_cell_past_rim_is_clamped_and_feeds():
    colony = make_colony(seed=1, initial_cells=1)
    cell = colony.cells[1]
    cell.position = Vec2(10.0, 10.0)  # past the rim, at the grid corner
    cell.energy = ENERGY_REF
    colony.step(TICK_DT)
    assert inside_dish(cell)
    run(colony, 59)
    assert cell.energy > ENERGY_REF


def test_dead_cells_removed_after_fade():
    colony = make_colony(seed=1, initial_cells=3, nutrient_regen=0.0)
    colony.env.nutrient.fill(0.0)
    victim = colony.cells[1]
    victim.energy = 1e-9  # one tick of upkeep finishes it
    events = colony.step(TICK_DT)
    assert [e.cell for e in events if isinstance(e, Died)] == [victim]
    assert victim.is_dying
    assert colony.stats()["deaths"] == 1
    assert victim not in living(colony)
    assert colony.stats()["population"] == len(living(colony))
    position, t, energy = victim.position, victim.cycle.t, victim.energy
    fade_ticks = round(DYING_SECONDS / TICK_DT)
    for _ in range(fade_ticks - 1):
        colony.step(TICK_DT)
        assert colony.cells.get(1) is victim
        # dying cells don't move, feed or cycle
        assert (victim.position, victim.cycle.t, victim.energy) == (position, t, energy)
    colony.step(TICK_DT)
    assert 1 not in colony.cells
    assert colony.stats()["deaths"] == 1  # counted once, at death


def test_step_rejects_bad_dt():
    colony = make_colony(seed=1, initial_cells=1)
    for dt in (0.0, -TICK_DT, math.nan, math.inf):
        with pytest.raises(ValueError):
            colony.step(dt)
