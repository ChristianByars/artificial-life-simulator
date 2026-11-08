"""Plane geometry shared by every life form (sim code, no Ursina)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

DISH_RADIUS = 10.0  # our assumption: view scale, world units


@dataclass(frozen=True, slots=True)
class Vec2:
    """A point or vector on the dish plane."""

    x: float
    y: float

    def __add__(self, o: Vec2) -> Vec2:
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: Vec2) -> Vec2:
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> Vec2:
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> Vec2:
        return Vec2(self.x / k, self.y / k)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def dot(self, o: Vec2) -> float:
        return self.x * o.x + self.y * o.y

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def to_list(self) -> list[float]:
        return [self.x, self.y]

    @classmethod
    def from_list(cls, seq: Sequence[float]) -> Vec2:
        return cls(float(seq[0]), float(seq[1]))


ZERO = Vec2(0.0, 0.0)


def to_world(p: Vec2, y: float = 0.0) -> tuple[float, float, float]:
    """Sim (x, y) maps to Ursina (x, height, y) (decision M3)."""
    return (p.x, y, p.y)


def from_world(x: float, y: float, z: float) -> Vec2:
    return Vec2(x, z)


def ray_ground_point(
    origin: tuple[float, float, float], direction: tuple[float, float, float]
) -> Vec2 | None:
    """Where a world-space ray meets the dish plane y = 0, as a sim point (decision M3).

    None when it never does: the ray is parallel to the plane, points away from it, or is not
    finite. ``direction`` need not be a unit vector.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction
    if not all(math.isfinite(v) for v in (ox, oy, oz, dx, dy, dz)) or dy == 0:
        return None
    if oy != 0 and (oy > 0) == (dy > 0):  # starts above and points up, or below and points down
        return None
    x, z = ox - oy * (dx / dy), oz - oy * (dz / dy)
    if not (math.isfinite(x) and math.isfinite(z)):
        return None
    return Vec2(x, z)


def dish_point(
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    radius: float = DISH_RADIUS,
) -> Vec2 | None:
    """:func:`ray_ground_point`, but only inside the dish rim (on the rim counts)."""
    p = ray_ground_point(origin, direction)
    return p if p is not None and p.length() <= radius else None
