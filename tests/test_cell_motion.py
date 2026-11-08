"""Cell motion: Brownian step, soft-sphere repulsion, dish clamping (SPEC 6.1, S4)."""

from __future__ import annotations

import math
import random

from alife.geometry import DISH_RADIUS, ZERO, Vec2
from alife.lifeforms.cell.motion import brownian_step, clamp_to_dish, repulsion
from alife.spatial import SpatialHash

STIFFNESS = 5.0
DT = 1 / 60


def run_repulsion(bodies: list[tuple[int, Vec2, float]]) -> dict[int, Vec2]:
    grid = SpatialHash(1.0)
    grid.rebuild((i, p) for i, p, _ in bodies)
    return repulsion(bodies, grid, STIFFNESS, DT)


def test_brownian_step_draws_x_then_y() -> None:
    sigma = math.sqrt(2 * 0.01 * DT)
    ref = random.Random(7)
    dx, dy = ref.gauss(0, sigma), ref.gauss(0, sigma)
    out = brownian_step(Vec2(1.0, 2.0), 0.01, DT, random.Random(7))
    assert out == Vec2(1.0 + dx, 2.0 + dy)


def test_brownian_zero_diffusion_does_not_move() -> None:
    assert brownian_step(Vec2(1.0, 2.0), 0.0, DT, random.Random(0)) == Vec2(1.0, 2.0)


def test_repulsion_separates_overlap() -> None:
    bodies = [(1, Vec2(0.0, 0.0), 0.5), (2, Vec2(0.6, 0.0), 0.5)]  # overlap 0.4
    disp = run_repulsion(bodies)
    assert disp[1].x < 0.0 < disp[2].x
    assert disp[1].y == disp[2].y == 0.0
    expected = min(STIFFNESS * 0.4 * DT, 0.4 / 2)
    assert math.isclose(disp[2].x, expected)
    assert math.isclose(disp[1].x, -expected)


def test_repulsion_never_overshoots_half_overlap() -> None:
    bodies = [(1, Vec2(0.0, 0.0), 0.5), (2, Vec2(0.6, 0.0), 0.5)]
    grid = SpatialHash(1.0)
    grid.rebuild((i, p) for i, p, _ in bodies)
    disp = repulsion(bodies, grid, 1e6, DT)  # huge stiffness: capped at overlap / 2
    assert math.isclose(disp[2].x, 0.2)
    assert math.isclose(disp[1].x, -0.2)


def test_no_push_beyond_contact() -> None:
    bodies = [(1, Vec2(0.0, 0.0), 0.5), (2, Vec2(1.0, 0.0), 0.5), (3, Vec2(3.0, 0.0), 0.5)]
    disp = run_repulsion(bodies)  # exactly touching, and far apart: no overlap
    assert all(d == ZERO for d in disp.values())


def test_repulsion_returns_entry_for_every_body() -> None:
    bodies = [(4, Vec2(0.0, 0.0), 0.5), (9, Vec2(5.0, 5.0), 0.5)]
    assert set(run_repulsion(bodies)) == {4, 9}
    assert run_repulsion([]) == {}


def test_coincident_cells_separate_without_nan() -> None:
    bodies = [(3, Vec2(0.0, 0.0), 0.5), (8, Vec2(0.0, 0.0), 0.5)]
    first = run_repulsion(bodies)
    again = run_repulsion(bodies)
    assert first == again  # same way every run, no rng
    for d in first.values():
        assert math.isfinite(d.x) and math.isfinite(d.y)
    a, b = first[3], first[8]
    assert a != b
    assert a.length() > 0.0
    # lower id moves along +u, higher along -u, u at angle (min_id * 2.399963) mod 2 pi
    ang = (3 * 2.399963) % (2 * math.pi)
    u = Vec2(math.cos(ang), math.sin(ang))
    assert a.dot(u) > 0.0 > b.dot(u)
    assert math.isclose(a.length(), min(STIFFNESS * 1.0 * DT, 0.5))


def test_repulsion_conserves_centre() -> None:
    rng = random.Random(1)
    bodies = [(i, Vec2(rng.uniform(-1, 1), rng.uniform(-1, 1)), 0.4) for i in range(30)]
    disp = run_repulsion(bodies)
    sx = sum(d.x for d in disp.values())
    sy = sum(d.y for d in disp.values())
    assert abs(sx) < 1e-12 and abs(sy) < 1e-12


def test_clamp_to_dish() -> None:
    r = 0.5
    p = clamp_to_dish(Vec2(15.0, 0.0), r)
    assert math.isclose(p.x, DISH_RADIUS - r) and math.isclose(p.y, 0.0, abs_tol=1e-12)
    q = clamp_to_dish(Vec2(3.0, 4.0) * 3.0, r)  # |p| = 15 on the 3-4-5 ray
    assert math.isclose(q.length(), DISH_RADIUS - r)
    assert math.isclose(q.x / q.y, 3.0 / 4.0)
    inside = Vec2(1.0, -2.0)
    assert clamp_to_dish(inside, r) == inside
    assert clamp_to_dish(ZERO, r) == ZERO
    assert clamp_to_dish(Vec2(15.0, 0.0), 1.0, dish_radius=5.0) == Vec2(4.0, 0.0)


def test_clamp_with_radius_at_least_dish_radius_is_finite() -> None:
    p = clamp_to_dish(Vec2(1.0, 1.0), 12.0)
    assert math.isfinite(p.x) and math.isfinite(p.y)
    assert clamp_to_dish(ZERO, 12.0) == ZERO
