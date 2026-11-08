"""Nutrient field on the dish floor (SPEC 6.1, decisions M7 and M10). Sim code, no Ursina.

Source model: Fick's second law, dN/dt = D * laplacian(N), solved with explicit finite
differences (FTCS, 5-point Laplacian), a textbook scheme. The edges of the square grid are
no-flux, so diffusion alone conserves mass.

Units: positions and `half_extent` in dish units; `diffusion` in dish units^2 per sim second;
`regen_rate` in 1/s; concentration N per dish unit^2, so mass in a cell is N * dx^2.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np

from alife.geometry import DISH_RADIUS, Vec2

GRID_N = 128  # SPEC 6.1 "e.g. 128x128"
MAX_DIFFUSION_NUMBER = 0.2  # D*dt/dx^2 per substep; FTCS is stable and non-negative below 0.25


class Environment:
    """A square nutrient grid, indexed `nutrient[ix, iy]`."""

    def __init__(
        self,
        *,
        diffusion: float,
        regen_rate: float,
        capacity: float = 1.0,
        initial: float = 1.0,
        n: int = GRID_N,
        half_extent: float = DISH_RADIUS,
        unlimited: bool = False,
    ) -> None:
        for name, value in (
            ("diffusion", diffusion),
            ("regen_rate", regen_rate),
            ("capacity", capacity),
            ("initial", initial),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if not math.isfinite(half_extent) or half_extent <= 0.0:
            raise ValueError(f"half_extent must be finite and > 0, got {half_extent!r}")
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n!r}")
        self.diffusion = float(diffusion)  # dish units^2 / s
        self.regen_rate = float(regen_rate)  # 1/s
        self.capacity = float(capacity)
        self.n = int(n)
        self.half_extent = float(half_extent)
        self.unlimited = bool(unlimited)
        self.nutrient = np.full((self.n, self.n), float(initial), dtype=np.float64)

    @property
    def dx(self) -> float:
        """Grid spacing in dish units."""
        return 2.0 * self.half_extent / self.n

    def index(self, p: Vec2) -> tuple[int, int]:
        """Grid cell containing `p`, clamped to the grid."""
        dx = self.dx
        ix = math.floor((p.x + self.half_extent) / dx)
        iy = math.floor((p.y + self.half_extent) / dx)
        return min(max(ix, 0), self.n - 1), min(max(iy, 0), self.n - 1)

    def total(self) -> float:
        """Total nutrient mass, sum of N * dx^2."""
        return float(self.nutrient.sum()) * self.dx**2

    def step(self, dt: float) -> None:
        """Diffuse (no-flux edges), then regenerate, over `dt` sim seconds."""
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError(f"dt must be finite and >= 0, got {dt!r}")
        if dt == 0.0:
            return
        dx2 = self.dx**2
        substeps = max(1, math.ceil(self.diffusion * dt / (MAX_DIFFUSION_NUMBER * dx2)))
        k = self.diffusion * (dt / substeps) / dx2
        if k > 0.0:
            field = self.nutrient
            for _ in range(substeps):
                padded = np.pad(field, 1, mode="edge")  # replicated edges: zero flux
                lap = (
                    padded[2:, 1:-1]
                    + padded[:-2, 1:-1]
                    + padded[1:-1, 2:]
                    + padded[1:-1, :-2]
                    - 4.0 * field
                )
                field = field + k * lap
            self.nutrient = field
        if self.regen_rate > 0.0:
            # our assumption: linear relaxation toward capacity (clamped so it cannot overshoot)
            rate = min(self.regen_rate * dt, 1.0)
            self.nutrient += rate * (self.capacity - self.nutrient)

    def _footprint(self, p: Vec2, radius: float) -> tuple[slice, slice, np.ndarray]:
        """Grid cells whose centres lie within `radius` of `p`; the containing cell if none.

        Returns (x slice, y slice, boolean mask over that block).
        """
        dx = self.dx
        r = max(float(radius), 0.0)
        lo_x = max(math.ceil((p.x - r + self.half_extent) / dx - 0.5), 0)
        hi_x = min(math.floor((p.x + r + self.half_extent) / dx - 0.5), self.n - 1)
        lo_y = max(math.ceil((p.y - r + self.half_extent) / dx - 0.5), 0)
        hi_y = min(math.floor((p.y + r + self.half_extent) / dx - 0.5), self.n - 1)
        if lo_x <= hi_x and lo_y <= hi_y:
            cx = -self.half_extent + (np.arange(lo_x, hi_x + 1) + 0.5) * dx
            cy = -self.half_extent + (np.arange(lo_y, hi_y + 1) + 0.5) * dx
            mask = (cx[:, None] - p.x) ** 2 + (cy[None, :] - p.y) ** 2 <= r * r
            if mask.any():
                return slice(lo_x, hi_x + 1), slice(lo_y, hi_y + 1), mask
        ix, iy = self.index(p)
        return slice(ix, ix + 1), slice(iy, iy + 1), np.ones((1, 1), dtype=bool)

    def take(self, p: Vec2, radius: float, demand: float) -> float:
        """Remove up to `demand` mass from the cells under a disc; return what was removed.

        Proportional to each cell's content, so the grid never goes below 0. With `unlimited`
        the grid is untouched and `demand` is returned.
        """
        if not demand > 0.0:
            return 0.0
        if self.unlimited:
            return float(demand)
        sx, sy, mask = self._footprint(p, radius)
        block = self.nutrient[sx, sy]
        available = float(block[mask].sum()) * self.dx**2
        if available <= 0.0:
            return 0.0
        fraction = min(1.0, demand / available)
        block[mask] *= 1.0 - fraction
        return available * fraction

    def add(self, p: Vec2, amount: float, radius: float) -> None:
        """Deposit `amount` mass, spread evenly over the cells under a disc."""
        if not amount > 0.0 or not math.isfinite(amount):
            return
        sx, sy, mask = self._footprint(p, radius)
        block = self.nutrient[sx, sy]
        block[mask] += amount / (int(mask.sum()) * self.dx**2)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe state (decision M4): grid as base64 of little-endian float64."""
        return {
            "shape": [self.n, self.n],
            "data": base64.b64encode(self.nutrient.astype("<f8").tobytes()).decode("ascii"),
            "diffusion": self.diffusion,
            "regen_rate": self.regen_rate,
            "capacity": self.capacity,
            "half_extent": self.half_extent,
            "unlimited": self.unlimited,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Environment:
        """Rebuild from `to_dict`; raises ValueError on a malformed dict."""
        try:
            shape = d["shape"]
            if len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(f"grid must be square, got shape {shape!r}")
            n = int(shape[0])
            raw = base64.b64decode(d["data"], validate=True)
            if len(raw) != n * n * 8:
                raise ValueError(f"grid data is {len(raw)} bytes, expected {n * n * 8}")
            grid = np.frombuffer(raw, dtype="<f8").reshape(n, n).astype(np.float64)
            if not np.all(np.isfinite(grid)) or grid.min() < 0.0:
                raise ValueError("grid contains non-finite or negative values")
            env = cls(
                diffusion=float(d["diffusion"]),
                regen_rate=float(d["regen_rate"]),
                capacity=float(d["capacity"]),
                n=n,
                half_extent=float(d["half_extent"]),
                unlimited=bool(d["unlimited"]),
            )
        except ValueError:
            raise
        except (KeyError, TypeError) as exc:
            raise ValueError(f"bad environment state: {exc!r}") from exc
        env.nutrient = grid
        return env
