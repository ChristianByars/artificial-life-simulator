"""SPEC §6.1 test_doubling_time: with unlimited nutrients, the colony's population doubling
time matches the cell-cycle period within 10%.

Source: Tyson 1991 (BIOMD0000000005). Each cell divides once per MPF oscillation, so an
unsynchronised colony with unlimited food grows as 2^(t / P), where P = 35.60 model minutes
(SPIKE.md C2) is the oscillator's period. time_scale is 10x the default so the test is short;
the cells' biology (cycle, feeding, upkeep, growth) all runs on model time, so it is the same
physics, sped up on screen.
"""

from __future__ import annotations

import random

import numpy as np

from alife.lifeforms.cell.cycle import DEFAULT_TIME_SCALE
from alife.lifeforms.cell.params import PARAMS
from alife.lifeforms.cell.sim import CellColony
from alife.session import TICK_DT

P = 35.60  # model minutes: reference period of BIOMD0000000005 (SPIKE.md C2)
TIME_SCALE = 10 * DEFAULT_TIME_SCALE
P_SCREEN = P / TIME_SCALE  # sim seconds per period (6 s)
INITIAL_CELLS = 8
RUN_PERIODS = 4.5
FIT_FROM, FIT_TO = 1.0, 4.0  # fit window, in periods
TOLERANCE = 0.10  # relative (SPEC §6.1)


def test_doubling_time():
    params = {key: float(p.default) for key, p in PARAMS.items()}
    params.update(initial_cells=INITIAL_CELLS, time_scale=TIME_SCALE)
    colony = CellColony(random.Random(2026), params)
    colony.env.unlimited = True

    t, population = [], []
    for _ in range(round(RUN_PERIODS * P_SCREEN / TICK_DT)):
        colony.step(TICK_DT)
        t.append(colony.time)
        population.append(colony.stats()["population"])
    t_arr = np.array(t)
    pop = np.array(population, dtype=float)

    window = (t_arr >= FIT_FROM * P_SCREEN) & (t_arr <= FIT_TO * P_SCREEN)
    slope = np.polyfit(t_arr[window], np.log2(pop[window]), 1)[0]  # doublings per sim second
    doubling = 1.0 / slope

    assert colony.stats()["deaths"] == 0  # unlimited food: nobody starves
    error = abs(doubling - P_SCREEN) / P_SCREEN
    assert error < TOLERANCE, (
        f"doubling {doubling:.4f} s vs period {P_SCREEN:.4f} s ({error:.2%}); "
        f"population {population[0]} -> {population[-1]}"
    )
