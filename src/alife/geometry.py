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
