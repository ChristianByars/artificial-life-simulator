"""Nutrient grid: indexing, conservation, stability, regeneration, uptake, round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alife.geometry import DISH_RADIUS, Vec2
from alife.lifeforms.cell.environment import GRID_N, Environment

DT = 1 / 60


def make(**kw: object) -> Environment:
    args: dict[str, object] = {"diffusion": 0.05, "regen_rate": 0.0}
    args.update(kw)
    return Environment(**args)  # type: ignore[arg-type]


def test_shape_and_geometry() -> None:
    env = make()
    assert env.nutrient.shape == (GRID_N, GRID_N) == (128, 128)
    assert env.nutrient.dtype == np.float64
    assert env.dx == pytest.approx(2 * DISH_RADIUS / GRID_N)
    assert env.total() == pytest.approx((2 * DISH_RADIUS) ** 2)  # initial 1.0 everywhere


def test_index_centres() -> None:
    env = make()
    assert env.index(Vec2(0.0, 0.0)) == (64, 64)
    assert env.index(Vec2(-DISH_RADIUS + 0.5 * env.dx, DISH_RADIUS - 0.5 * env.dx)) == (0, 127)


def test_index_clamps_at_and_beyond_edge() -> None:
    env = make()
    assert env.index(Vec2(10.0, 10.0)) == (127, 127)
    assert env.index(Vec2(-50.0, 3.0))[0] == 0
    assert env.index(Vec2(-10.0, -10.0)) == (0, 0)
    assert env.index(Vec2(1e9, -1e9)) == (127, 0)


def test_no_flux_conserves_mass() -> None:
    env = make(diffusion=0.5)
    rng = np.random.default_rng(1)
    env.nutrient[:] = rng.random((GRID_N, GRID_N))
    before = env.total()
    for _ in range(1000):
        env.step(DT)
    assert env.total() == pytest.approx(before, rel=1e-12)


def test_stable_at_max_diffusion() -> None:
    env = make(diffusion=1.0)
    rng = np.random.default_rng(2)
    env.nutrient[:] = rng.random((GRID_N, GRID_N))
    top = float(env.nutrient.max())
    for _ in range(600):
        env.step(DT)
    assert np.all(np.isfinite(env.nutrient))
    assert env.nutrient.min() >= 0.0
    assert env.nutrient.max() <= top


def test_regen_relaxes_to_capacity() -> None:
    env = make(regen_rate=0.5, capacity=2.0, initial=0.0)
    for _ in range(30 * 60):
        env.step(DT)
    assert env.nutrient.mean() > 0.99 * 2.0
    assert env.nutrient.max() <= 2.0 + 1e-12


def test_regen_pulls_down_from_above_capacity() -> None:
    env = make(regen_rate=0.5, capacity=1.0, initial=3.0)
    env.step(1.0)
    assert 1.0 < env.nutrient.mean() < 3.0


def test_take_never_negative() -> None:
    env = make(initial=1.0)
    p = Vec2(1.0, -2.0)
    radius = 0.5
    ix, iy = env.index(p)
    got = env.take(p, radius, 1e6)
    assert 0.0 < got < 1e6
    assert env.nutrient.min() >= 0.0
    assert env.nutrient[ix, iy] == pytest.approx(0.0, abs=1e-12)  # centre cell emptied
    assert env.take(p, radius, 1e6) == pytest.approx(0.0, abs=1e-12)


def test_take_removes_exactly_what_it_returns() -> None:
    env = make(initial=1.0)
    before = env.total()
    got = env.take(Vec2(0.0, 0.0), 0.4, 0.01)
    assert got == pytest.approx(0.01)
    assert env.total() == pytest.approx(before - got, rel=1e-12)


def test_take_zero_radius_uses_containing_cell() -> None:
    env = make(initial=1.0)
    ix, iy = env.index(Vec2(3.0, 3.0))
    got = env.take(Vec2(3.0, 3.0), 0.0, 1e6)
    assert got == pytest.approx(env.dx**2)
    assert env.nutrient[ix, iy] == pytest.approx(0.0, abs=1e-12)
    assert env.nutrient.sum() == pytest.approx(GRID_N**2 - 1.0)


def test_take_nonpositive_demand_is_zero() -> None:
    env = make()
    before = env.nutrient.copy()
    assert env.take(Vec2(0.0, 0.0), 0.3, 0.0) == 0.0
    assert env.take(Vec2(0.0, 0.0), 0.3, -1.0) == 0.0
    assert np.array_equal(env.nutrient, before)


def test_take_at_rim() -> None:
    env = make()
    assert env.take(Vec2(9.99, 0.0), 0.35, 0.01) > 0.0
    assert env.take(Vec2(10.0, 10.0), 0.35, 0.01) > 0.0
    assert env.take(Vec2(-25.0, 40.0), 0.35, 0.01) > 0.0  # far outside clamps to a corner
    assert env.nutrient.min() >= 0.0


def test_unlimited_returns_demand() -> None:
    env = make(unlimited=True, initial=0.0)
    assert env.take(Vec2(0.0, 0.0), 0.3, 0.5) == 0.5
    assert env.total() == 0.0


def test_add_deposits_mass() -> None:
    env = make(initial=0.0)
    env.add(Vec2(2.0, 2.0), 0.5, 0.5)
    assert env.total() == pytest.approx(0.5, rel=1e-12)
    env.add(Vec2(-10.0, 10.0), 0.25, 0.0)  # corner, zero radius: containing cell
    assert env.total() == pytest.approx(0.75, rel=1e-12)
    assert env.nutrient.min() >= 0.0


def test_add_nonpositive_is_noop() -> None:
    env = make(initial=0.0)
    env.add(Vec2(0.0, 0.0), -1.0, 0.5)
    assert env.total() == 0.0


def test_step_rejects_bad_dt() -> None:
    env = make()
    with pytest.raises(ValueError):
        env.step(float("nan"))
    with pytest.raises(ValueError):
        env.step(-0.1)


def test_constructor_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        make(diffusion=-1.0)
    with pytest.raises(ValueError):
        make(regen_rate=float("nan"))
    with pytest.raises(ValueError):
        make(n=1)


def test_roundtrip_exact() -> None:
    env = make(diffusion=0.3, regen_rate=0.2, capacity=1.5, initial=0.5)
    rng = np.random.default_rng(3)
    env.nutrient[:] = rng.random((GRID_N, GRID_N))
    for _ in range(5):
        env.step(DT)
    d = json.loads(json.dumps(env.to_dict()))  # must survive JSON
    other = Environment.from_dict(d)
    assert np.array_equal(other.nutrient, env.nutrient)
    assert other.dx == env.dx
    env.step(DT)
    other.step(DT)
    assert np.array_equal(other.nutrient, env.nutrient)


def test_roundtrip_keeps_unlimited_and_size() -> None:
    env = make(n=16, half_extent=4.0, unlimited=True)
    other = Environment.from_dict(env.to_dict())
    assert other.nutrient.shape == (16, 16)
    assert other.dx == env.dx
    assert other.take(Vec2(0.0, 0.0), 0.1, 0.7) == 0.7


def test_from_dict_rejects_corrupt_data() -> None:
    good = make(n=8).to_dict()
    with pytest.raises(ValueError):
        Environment.from_dict({**good, "data": "AAAA"})  # wrong length
    with pytest.raises(ValueError):
        Environment.from_dict({**good, "data": "!!not base64!!"})
    with pytest.raises(ValueError):
        Environment.from_dict({k: v for k, v in good.items() if k != "data"})
