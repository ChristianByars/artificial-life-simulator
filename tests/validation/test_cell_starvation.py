"""SPEC §6.1 test_starvation_arrest: with zero nutrients, cells stop dividing.

Source: Tyson 1991 (BIOMD0000000005) oscillates only while cyclin synthesis is high enough;
below about 0.72 of the published rate MPF no longer spikes (measured in S3). The nutrient
coupling (our assumption, SPEC §6.1) sets synthesis from the cell's energy, so a cell with no
food burns its energy, slows, and arrests. Cells that divide early in the run may finish, but
none divides after the first period, while arrested cells are still alive to show it is the
cycle that stopped, not just the cells that died. time_scale is 10x the default to keep the
test short (same physics: the cells' biology runs on model time).
"""

from __future__ import annotations

import random

from alife.lifeforms.cell.cycle import DEFAULT_TIME_SCALE
from alife.lifeforms.cell.params import PARAMS
from alife.lifeforms.cell.sim import CellColony, Divided
from alife.session import TICK_DT

P = 35.60  # model minutes: reference period of BIOMD0000000005 (SPIKE.md C2)
TIME_SCALE = 10 * DEFAULT_TIME_SCALE
P_SCREEN = P / TIME_SCALE  # sim seconds per period (6 s)
INITIAL_CELLS = 20
RUN_PERIODS = 3


def test_starvation_arrest():
    params = {key: float(p.default) for key, p in PARAMS.items()}
    params.update(initial_cells=INITIAL_CELLS, nutrient_regen=0.0, time_scale=TIME_SCALE)
    colony = CellColony(random.Random(1991), params)
    colony.env.nutrient.fill(0.0)  # no food, and none regrows

    early: list[Divided] = []
    late: list[tuple[float, Divided]] = []
    alive_after_first_period = None
    for _ in range(round(RUN_PERIODS * P_SCREEN / TICK_DT)):
        events = colony.step(TICK_DT)
        divided = [e for e in events if isinstance(e, Divided)]
        if colony.time <= P_SCREEN:
            early += divided
        else:
            late += [(colony.time, e) for e in divided]
            if alive_after_first_period is None:
                alive_after_first_period = colony.stats()["population"]

    assert early, "no cell divided at all: the cycle never ran, so the test shows nothing"
    assert alive_after_first_period, "every cell had died by the first period's end"
    assert not late, f"divisions after the first period at t = {[round(t, 2) for t, _ in late]} s"
