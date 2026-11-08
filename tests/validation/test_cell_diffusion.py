"""SPEC 6.1 test_diffusion_analytic: nutrient spreads as the analytic 2-D Gaussian.

Source: Fick's second law, dN/dt = D * laplacian(N). For unit mass released at the origin of
an unbounded plane the solution is G(r, t) = exp(-r^2 / (4 D t)) / (4 pi D t) (textbook result,
e.g. Crank, The Mathematics of Diffusion). The dish edge is far from the cloud, so the
no-flux boundary does not matter here.
"""

from __future__ import annotations

import numpy as np

from alife.lifeforms.cell.environment import Environment

D = 0.05  # dish units^2 / s (our assumption; any value works for the test)
T0 = 2.0  # s, age of the Gaussian at the start
TICKS = 600
DT = 1 / 60
TOLERANCE = 0.02  # of the analytic peak (SPEC 6.1)


def analytic(env: Environment, t: float) -> np.ndarray:
    centres = -env.half_extent + (np.arange(env.n) + 0.5) * env.dx
    x, y = np.meshgrid(centres, centres, indexing="ij")
    return np.exp(-(x * x + y * y) / (4 * D * t)) / (4 * np.pi * D * t)


def test_diffusion_analytic() -> None:
    env = Environment(diffusion=D, regen_rate=0.0, initial=0.0)
    g0 = analytic(env, T0)
    env.nutrient[:] = g0 / (g0.sum() * env.dx**2)  # mass exactly 1 on the grid
    assert abs(env.total() - 1.0) <= 1e-12

    for _ in range(TICKS):
        env.step(DT)

    t_end = T0 + TICKS * DT
    expected = analytic(env, t_end)
    err = np.max(np.abs(env.nutrient - expected))
    assert err <= TOLERANCE * expected.max(), (err, expected.max())
    assert abs(env.total() - 1.0) <= 1e-9
    assert env.nutrient.min() >= 0.0
