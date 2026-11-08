from random import Random

from alife.geometry import DISH_RADIUS, Vec2
from alife.lifeforms.demo.sim import WalkerSim


def run(sim: WalkerSim, steps: int) -> None:
    for _ in range(steps):
        sim.step(1 / 60)


def test_same_seed_same_state():
    a, b = WalkerSim(Random(1), 30), WalkerSim(Random(1), 30)
    run(a, 600)
    run(b, 600)
    assert a.get_state() == b.get_state()


def test_different_seed_differs():
    a, b = WalkerSim(Random(1), 30), WalkerSim(Random(2), 30)
    run(a, 10)
    run(b, 10)
    assert a.get_state() != b.get_state()


def test_state_roundtrip():
    rng = Random(1)
    a = WalkerSim(rng, 30)
    run(a, 200)
    state, rng_state = a.get_state(), rng.getstate()
    rng2 = Random(999)
    b = WalkerSim(rng2, 1)
    rng2.setstate(rng_state)  # the session owns the rng and restores it (M4)
    b.set_state(state)
    assert b.get_state() == state
    run(a, 100)
    run(b, 100)
    assert a.get_state() == b.get_state()


def test_state_is_json_serializable():
    import json

    sim = WalkerSim(Random(1), 5)
    run(sim, 3)
    sim.on_pick(2)
    assert json.loads(json.dumps(sim.get_state())) == sim.get_state()


def test_walkers_move():
    sim = WalkerSim(Random(1), 5)
    before = [w.position for w in sim.walkers]
    run(sim, 10)
    assert [w.position for w in sim.walkers] != before


def test_walkers_stay_in_dish():
    sim = WalkerSim(Random(3), 50)
    for w in sim.walkers:
        w.position = Vec2(DISH_RADIUS, 0.0)  # start on the rim
    run(sim, 6000)
    assert all(w.position.length() <= DISH_RADIUS + 1e-9 for w in sim.walkers)


def test_shift_click_adds_walker():
    sim = WalkerSim(Random(1), 5)
    sim.on_dish_click(Vec2(1, 1), {"shift"})
    assert len(sim.walkers) == 6
    assert sim.walkers[-1].position == Vec2(1, 1)
    sim.on_dish_click(Vec2(1, 1), set())
    assert len(sim.walkers) == 6


def test_shift_click_outside_dish_is_clamped():
    sim = WalkerSim(Random(1), 1)
    sim.on_dish_click(Vec2(50, 0), {"shift"})
    assert sim.walkers[-1].position.length() <= DISH_RADIUS


def test_ids_unique_and_ascending():
    sim = WalkerSim(Random(1), 3)
    sim.on_dish_click(Vec2(0, 0), {"shift"})
    ids = [w.id for w in sim.walkers]
    assert ids == sorted(set(ids)) and len(ids) == 4


def test_stats_and_pick():
    sim = WalkerSim(Random(1), 7)
    assert sim.stats() == {"walkers": 7}
    sim.on_pick(3)
    assert sim.selected == 3
    sim.on_pick(None)
    assert sim.selected is None
