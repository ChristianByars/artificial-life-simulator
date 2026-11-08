"""Cell motion: Brownian steps, soft-sphere repulsion and dish clamping (sim code, no Ursina).

Brownian step: Einstein 1905, each axis gains N(0, sqrt(2 D dt)), so the 2-D mean squared
displacement is 4 D t (SPEC 6.1 test_brownian_msd). Repulsion is an overdamped soft-sphere
contact (textbook); stiffness and the per-pair cap are our assumptions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from alife.geometry import DISH_RADIUS, ZERO, Vec2
from alife.spatial import SpatialHash

# our assumption: below this centre distance two cells count as coincident and the push
# direction comes from the id, not from the (undefined) vector between them.
_COINCIDENT = 1e-9
# Golden angle in radians (2 pi (1 - 1/phi)): spreads successive ids evenly around the circle.
_GOLDEN_ANGLE = 2.399963


def brownian_step(p: Vec2, diffusion: float, dt: float, rng: random.Random) -> Vec2:
    """Return ``p`` after one Brownian step: x then y drawn from ``N(0, sqrt(2 D dt))``."""
    sigma = math.sqrt(2.0 * diffusion * dt)
    dx = rng.gauss(0.0, sigma)
    dy = rng.gauss(0.0, sigma)
    return Vec2(p.x + dx, p.y + dy)


def repulsion(
    bodies: Sequence[tuple[int, Vec2, float]],
    grid: SpatialHash,
    stiffness: float,
    dt: float,
) -> dict[int, Vec2]:
    """Displacement per body id from overdamped soft-sphere contacts.

    ``bodies`` holds ``(id, position, radius)``; the caller has rebuilt ``grid`` from the same
    positions. Each overlapping pair (overlap = r_i + r_j - d > 0) moves apart by
    ``min(stiffness * overlap * dt, overlap / 2)`` each along the pair's unit vector, so a
    pair never crosses over and the displacements sum to zero. Coincident pairs use the angle
    ``(min_id * 2.399963) mod 2 pi``: the lower id moves along +u, the higher along -u.
    Deterministic: pairs are visited in sorted order and no rng is used.
    """
    pos: dict[int, Vec2] = {}
    radius: dict[int, float] = {}
    for body_id, p, r in bodies:
        pos[body_id] = p
        radius[body_id] = r
    out: dict[int, Vec2] = {body_id: ZERO for body_id in pos}
    if len(pos) < 2:
        return out
    max_contact = 2.0 * max(radius.values())
    for i, j in grid.pairs(max_contact):
        delta = pos[j] - pos[i]
        d = delta.length()
        overlap = radius[i] + radius[j] - d
        if overlap <= 0.0:
            continue
        move = min(stiffness * overlap * dt, overlap / 2.0)
        if d < _COINCIDENT:
            angle = math.fmod(i * _GOLDEN_ANGLE, 2.0 * math.pi)  # i < j, so i is the min id
            u = Vec2(math.cos(angle), math.sin(angle))
            lower_dir = u
        else:
            u = delta / d  # points from i to j
            lower_dir = -u
        push = lower_dir * move
        out[i] = out[i] + push
        out[j] = out[j] - push
    return out


def clamp_to_dish(p: Vec2, radius: float, dish_radius: float = DISH_RADIUS) -> Vec2:
    """Keep a cell of ``radius`` inside the dish: scale ``p`` onto ``|p| = dish_radius - radius``.

    A point already inside, or at the exact origin, is returned unchanged. A cell too big for the
    dish (``radius >= dish_radius``) is pinned to the origin.
    """
    limit = max(dish_radius - radius, 0.0)
    dist = p.length()
    if dist <= limit or dist == 0.0:
        return p
    return p * (limit / dist)
