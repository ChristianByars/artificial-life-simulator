"""Uniform-grid spatial hash: results must equal brute force."""

from __future__ import annotations

import random

import pytest

from alife.geometry import Vec2
from alife.spatial import SpatialHash


def _points(n: int = 500, seed: int = 7) -> list[tuple[int, Vec2]]:
    rng = random.Random(seed)
    return [(i, Vec2(rng.uniform(-10, 10), rng.uniform(-10, 10))) for i in range(n)]


def _brute_query(items: list[tuple[int, Vec2]], p: Vec2, radius: float) -> list[int]:
    r2 = radius * radius
    return sorted(i for i, q in items if (q - p).length_sq() <= r2)


def _brute_pairs(items: list[tuple[int, Vec2]], radius: float) -> list[tuple[int, int]]:
    r2 = radius * radius
    out = []
    for a, (i, p) in enumerate(items):
        for j, q in items[a + 1 :]:
            if (q - p).length_sq() <= r2:
                out.append((min(i, j), max(i, j)))
    return sorted(out)


@pytest.mark.parametrize("cell_size", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("radius", [0.1, 0.7, 3.0])
def test_query_matches_brute_force(cell_size: float, radius: float) -> None:
    items = _points()
    h = SpatialHash(cell_size)
    h.rebuild(items)
    rng = random.Random(11)
    for _ in range(40):
        p = Vec2(rng.uniform(-11, 11), rng.uniform(-11, 11))
        assert h.query(p, radius) == _brute_query(items, p, radius)
    # query centred on stored points (distance 0 hits itself)
    for i, p in items[:40]:
        got = h.query(p, radius)
        assert i in got
        assert got == _brute_query(items, p, radius)


@pytest.mark.parametrize("radius", [0.1, 0.7, 3.0])
def test_pairs_match_brute_force(radius: float) -> None:
    items = _points(300)
    h = SpatialHash(1.0)
    h.rebuild(items)
    assert h.pairs(radius) == _brute_pairs(items, radius)


def test_negative_and_boundary_coords() -> None:
    # points exactly on bucket edges, on both sides of zero
    items = [
        (0, Vec2(0.0, 0.0)),
        (1, Vec2(1.0, 0.0)),
        (2, Vec2(-1.0, 0.0)),
        (3, Vec2(0.0, -1.0)),
        (4, Vec2(-1.0, -1.0)),
        (5, Vec2(2.0, 2.0)),
        (6, Vec2(-3.0, 4.0)),
        (7, Vec2(-1e-12, -1e-12)),
    ]
    h = SpatialHash(1.0)
    h.rebuild(items)
    for radius in (0.0, 0.5, 1.0, 1.5, 2.0, 5.0):
        for _, p in items:
            assert h.query(p, radius) == _brute_query(items, p, radius)
        assert h.pairs(radius) == _brute_pairs(items, radius)
    # distance exactly == radius counts as within
    assert h.query(Vec2(0.0, 0.0), 1.0) == [0, 1, 2, 3, 7]


def test_radius_larger_than_cell_size() -> None:
    items = _points(200)
    h = SpatialHash(0.25)
    h.rebuild(items)
    assert h.query(Vec2(0.0, 0.0), 4.0) == _brute_query(items, Vec2(0.0, 0.0), 4.0)
    assert h.pairs(2.0) == _brute_pairs(items, 2.0)


def test_empty() -> None:
    h = SpatialHash(1.0)
    assert h.query(Vec2(0.0, 0.0), 5.0) == []
    assert h.pairs(5.0) == []
    h.rebuild([])
    assert h.query(Vec2(0.0, 0.0), 5.0) == []
    assert h.pairs(5.0) == []


def test_rebuild_replaces_previous_contents() -> None:
    h = SpatialHash(1.0)
    h.rebuild([(1, Vec2(0.0, 0.0)), (2, Vec2(0.5, 0.0))])
    h.rebuild([(3, Vec2(0.0, 0.0))])
    assert h.query(Vec2(0.0, 0.0), 10.0) == [3]
    assert h.pairs(10.0) == []


def test_rebuild_accepts_generator() -> None:
    h = SpatialHash(1.0)
    h.rebuild((i, Vec2(float(i), 0.0)) for i in range(4))
    assert h.query(Vec2(0.0, 0.0), 1.0) == [0, 1]


def test_results_sorted() -> None:
    rng = random.Random(3)
    ids = rng.sample(range(-50, 1000), 200)  # arbitrary, unordered, includes negatives
    items = [(i, Vec2(rng.uniform(-3, 3), rng.uniform(-3, 3))) for i in ids]
    h = SpatialHash(1.0)
    h.rebuild(items)
    q = h.query(Vec2(0.0, 0.0), 4.0)
    assert q == sorted(q)
    pr = h.pairs(1.0)
    assert pr == sorted(pr)
    assert all(a < b for a, b in pr)
    assert pr == _brute_pairs(items, 1.0)


def test_coincident_points_pair_once() -> None:
    h = SpatialHash(1.0)
    h.rebuild([(5, Vec2(1.0, 1.0)), (2, Vec2(1.0, 1.0))])
    assert h.pairs(0.0) == [(2, 5)]
    assert h.query(Vec2(1.0, 1.0), 0.0) == [2, 5]


def test_invalid_cell_size() -> None:
    with pytest.raises(ValueError):
        SpatialHash(0.0)
    with pytest.raises(ValueError):
        SpatialHash(-1.0)
