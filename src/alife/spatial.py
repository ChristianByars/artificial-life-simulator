"""Uniform-grid spatial hash for neighbour queries and repulsion pairs (sim code, no Ursina).

Buckets are keyed by ``(floor(x / cell_size), floor(y / cell_size))``. A query visits every
bucket its radius can reach, so a radius larger than ``cell_size`` still works. Distances are
compared as squared values; "within" means ``<= radius``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from alife.geometry import Vec2

# our assumption: widen the searched bucket range by this much so a point whose distance is
# exactly `radius` can't be missed through float rounding of `x - radius` at a bucket edge.
_REACH_PAD = 1e-9


class SpatialHash:
    """Uniform grid over 2D points, rebuilt each tick."""

    __slots__ = ("_buckets", "_cell_size", "_ids", "_inv", "_xs", "_ys")

    def __init__(self, cell_size: float) -> None:
        if not cell_size > 0.0 or not math.isfinite(cell_size):
            raise ValueError(f"cell_size must be a positive finite number, got {cell_size!r}")
        self._cell_size = float(cell_size)
        self._inv = 1.0 / self._cell_size
        self._buckets: dict[tuple[int, int], list[int]] = {}
        self._ids: list[int] = []
        self._xs: list[float] = []
        self._ys: list[float] = []

    @property
    def cell_size(self) -> float:
        return self._cell_size

    def rebuild(self, items: Iterable[tuple[int, Vec2]]) -> None:
        """Replace the contents with ``items`` (id, position)."""
        buckets: dict[tuple[int, int], list[int]] = {}
        ids: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        floor = math.floor
        cs = self._cell_size
        for idx, (item_id, p) in enumerate(items):
            x, y = p.x, p.y
            ids.append(item_id)
            xs.append(x)
            ys.append(y)
            key = (floor(x / cs), floor(y / cs))
            bucket = buckets.get(key)
            if bucket is None:
                buckets[key] = [idx]
            else:
                bucket.append(idx)
        self._buckets = buckets
        self._ids = ids
        self._xs = xs
        self._ys = ys

    def query(self, p: Vec2, radius: float) -> list[int]:
        """Ids at distance <= ``radius`` from ``p``, ascending."""
        if not self._ids or radius < 0.0:
            return []
        px, py = p.x, p.y
        cs = self._cell_size
        reach = radius + _REACH_PAD
        floor = math.floor
        x0, x1 = floor((px - reach) / cs), floor((px + reach) / cs)
        y0, y1 = floor((py - reach) / cs), floor((py + reach) / cs)
        r2 = radius * radius
        buckets, ids, xs, ys = self._buckets, self._ids, self._xs, self._ys
        out: list[int] = []
        if (x1 - x0 + 1) * (y1 - y0 + 1) > len(buckets):
            # radius spans more buckets than exist: walk the occupied ones instead
            for (cx, cy), bucket in buckets.items():
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    for i in bucket:
                        dx, dy = xs[i] - px, ys[i] - py
                        if dx * dx + dy * dy <= r2:
                            out.append(ids[i])
        else:
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    bucket = buckets.get((cx, cy))
                    if bucket is None:
                        continue
                    for i in bucket:
                        dx, dy = xs[i] - px, ys[i] - py
                        if dx * dx + dy * dy <= r2:
                            out.append(ids[i])
        out.sort()
        return out

    def pairs(self, radius: float) -> list[tuple[int, int]]:
        """Each pair of ids at distance <= ``radius`` once, as ``(i, j)`` with ``i < j``, sorted."""
        if len(self._ids) < 2 or radius < 0.0:
            return []
        r2 = radius * radius
        k = math.ceil((radius + _REACH_PAD) * self._inv)
        buckets, ids, xs, ys = self._buckets, self._ids, self._xs, self._ys
        out: list[tuple[int, int]] = []
        for (cx, cy), bucket in buckets.items():
            n = len(bucket)
            # pairs inside this bucket
            for a in range(n - 1):
                i = bucket[a]
                xi, yi = xs[i], ys[i]
                for b in range(a + 1, n):
                    j = bucket[b]
                    dx, dy = xs[j] - xi, ys[j] - yi
                    if dx * dx + dy * dy <= r2:
                        ia, ja = ids[i], ids[j]
                        out.append((ia, ja) if ia < ja else (ja, ia))
            # pairs with the "forward" half of the neighbourhood, so each bucket pair is seen once
            for ox in range(0, k + 1):
                for oy in range(-k if ox else 1, k + 1):
                    other = buckets.get((cx + ox, cy + oy))
                    if other is None:
                        continue
                    for i in bucket:
                        xi, yi = xs[i], ys[i]
                        for j in other:
                            dx, dy = xs[j] - xi, ys[j] - yi
                            if dx * dx + dy * dy <= r2:
                                ia, ja = ids[i], ids[j]
                                out.append((ia, ja) if ia < ja else (ja, ia))
        out.sort()
        return out
