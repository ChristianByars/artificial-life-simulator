"""SPEC 6.1 test_brownian_msd: an isolated cell's mean squared displacement is 4 D t.

Source: Einstein 1905. For free 2-D diffusion with coefficient D, <|r(t) - r(0)|^2> = 4 D t.
Each tick adds N(0, sqrt(2 D dt)) per axis (the exact discretisation), so the walk matches
at every tick. The dish clamp is not applied.
"""

from __future__ import annotations

import random

from alife.geometry import ZERO, Vec2
from alife.lifeforms.cell.motion import brownian_step

D = 0.01  # dish units^2 / s (our assumption; any value works for the test)
DT = 1 / 60
WALKERS = 4000
CHECKPOINTS = (2.5, 5.0, 10.0)  # s
TOLERANCE = 0.05  # relative (SPEC 6.1)


def test_brownian_msd() -> None:
    rng = random.Random(0)
    pos = [ZERO] * WALKERS
    ticks = {round(t / DT): t for t in CHECKPOINTS}
    for tick in range(1, max(ticks) + 1):
        pos = [brownian_step(p, D, DT, rng) for p in pos]
        if tick in ticks:
            t = ticks[tick]
            msd = sum(p.length_sq() for p in pos) / WALKERS
            expected = 4 * D * t
            assert abs(msd - expected) / expected < TOLERANCE, (t, msd, expected)


def test_brownian_msd_starts_from_given_point() -> None:
    p0 = Vec2(3.0, -2.0)
    rng = random.Random(1)
    pos = [p0] * 500
    for _ in range(60):
        pos = [brownian_step(p, D, DT, rng) for p in pos]
    msd = sum((p - p0).length_sq() for p in pos) / len(pos)
    assert abs(msd - 4 * D * 1.0) / (4 * D) < 0.2
