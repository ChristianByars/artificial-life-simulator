"""The cell colony stays stable for 10 minutes at default params (SPEC §8 M4). Task S6.

Slow (one long headless run per seed): run with `uv run pytest -m slow`.
"""

from __future__ import annotations

import random
from statistics import mean

import pytest

from alife.lifeforms.cell.params import PARAMS
from alife.lifeforms.cell.sim import CellColony
from alife.session import TICK_DT

TICKS = 36_000  # 10 sim minutes at 60 ticks/s, default time_scale (1x speed)
LATE = TICKS // 2  # the late half: the last 5 minutes
TICKS_PER_MINUTE = 3_600


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_population_stable_10_min(seed: int) -> None:
    params = {key: float(p.default) for key, p in PARAMS.items()}
    colony = CellColony(random.Random(seed), params)

    pop: list[int] = []
    for _ in range(TICKS):
        colony.step(TICK_DT)
        pop.append(int(colony.stats()["population"]))
    per_minute = pop[TICKS_PER_MINUTE - 1 :: TICKS_PER_MINUTE]

    assert min(pop) > 0, f"extinct; population at each minute {per_minute}"
    late = pop[LATE:]
    m = mean(late)
    # our definition of stable (flag in Plan 2 review): the late half stays within 50% of its mean
    assert 0.5 * m <= min(late) and max(late) <= 1.5 * m, (
        f"late half min {min(late)}, mean {m:.1f}, max {max(late)}; "
        f"population at each minute {per_minute}"
    )
