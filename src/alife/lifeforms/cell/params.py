"""Cell colony tunables and model constants (SPEC §6.1). Sim code, no Ursina.

Units: sim time in seconds (s); model time in minutes (min) of the Tyson 1991 model, where one
sim second is `time_scale` model minutes; lengths in dish units; nutrient concentration per
dish unit^2 (the grid's capacity is 1); energy in units of ENERGY_REF.

A cell's own biology (cell cycle, feeding, upkeep, growth) runs on model time, so `time_scale`
speeds up the whole cell, not just its cycle (our assumption). The dish's physics (nutrient
diffusion and regrowth, Brownian motion, repulsion, the death fade) runs on sim time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from alife.lifeforms import OUR_ASSUMPTION, Param
from alife.lifeforms.cell.cycle import DEFAULT_TIME_SCALE

# Period of the Tyson 1991 oscillator (BIOMD0000000005) at the published parameters, measured
# in M0 (SPIKE.md C2). Model minutes.
REFERENCE_PERIOD = 35.60

PARAMS: dict[str, Param] = {
    "initial_cells": Param("Initial cells", 20, 1, 200, OUR_ASSUMPTION, "cells"),
    "nutrient_regen": Param("Nutrient regen rate", 0.02, 0.0, 0.5, OUR_ASSUMPTION, "1/s"),
    "diffusion": Param("Diffusion rate", 0.05, 0.0, 0.5, OUR_ASSUMPTION, "units^2/s"),
    # With the energy constants below, test_starvation_arrest holds for upkeep ~0.025-0.07
    # (measured in S5): lower, a starving cell still divides after the first period; higher,
    # every cell is dead before the first period ends.
    "upkeep_cost": Param("Upkeep cost", 0.05, 0.0, 0.5, OUR_ASSUMPTION, "energy/min/unit^2"),
    # Max is just above 10x the default, the value the validation tests run at.
    "time_scale": Param("Time scale", DEFAULT_TIME_SCALE, 0.05, 6.0, OUR_ASSUMPTION, "min/s"),
}

# Energy. A cell cycles at full speed with energy >= ENERGY_REF: the cyclin synthesis scale is
# clamp(energy / ENERGY_REF, 0, 1) (SPEC §6.1 nutrient coupling). The cycle slows below that
# and arrests below ~0.72 x ENERGY_REF, where Tyson 1991 stops oscillating (synthesis scale
# 0.70-0.75, measured in S3).
ENERGY_REF = 1.0  # our assumption: the energy unit
ENERGY_MAX = 2.0 * ENERGY_REF  # our assumption: a full cell's daughters each start at ENERGY_REF
SEED_ENERGY = ENERGY_REF  # our assumption: seeded cells can cycle at full speed, no reserve

# Size. Upkeep and uptake scale with the cell's footprint area.
MAX_RADIUS = 0.35  # our assumption: adult radius, dish units (view scale)
MAX_AREA = math.pi * MAX_RADIUS**2
# our assumption: energy per dish unit^2 of new area; growing from birth size (half the adult
# area) to adult costs 0.5 x ENERGY_REF
GROWTH_COST = ENERGY_REF / MAX_AREA

# Feeding: Michaelis-Menten uptake (textbook), rate = VMAX * area * N / (KM + N), where N is
# the nutrient concentration under the cell's centre. Rate constants are our assumption,
# scaled to the dish (SPEC §6.1).
VMAX = 1.0  # our assumption: nutrient per model minute per dish unit^2 of cell, saturated
KM = 0.5  # our assumption: half-saturation concentration (the grid's capacity is 1)
YIELD = 1.0  # our assumption: energy per unit of nutrient

# Motion (sim time).
MOTILITY = 0.01  # our assumption: Brownian diffusion coefficient, dish units^2/s, visible motion
STIFFNESS = 5.0  # our assumption: soft-sphere repulsion rate, 1/s

# Division and death.
SPREAD = 0.02  # our assumption: each daughter's rate constants = published x (1 + U(-2%, 2%))
# our assumption: a division needs at least half a reference period since the last one, so a
# wobble of MPF around the threshold can't split a cell twice. Spread periods stay within a few
# % of REFERENCE_PERIOD, so this never blocks a real division.
MIN_DIVISION_FRACTION = 0.5
MIN_DIVISION_TIME = MIN_DIVISION_FRACTION * REFERENCE_PERIOD  # model minutes
DYING_SECONDS = 1.0  # our assumption: a dead cell fades for this long (sim s), then is removed

# Seeding: cells plated from an unsynchronised culture, each at a random point of its cycle.
SEED_WARMUP = 2 * REFERENCE_PERIOD  # our assumption: model minutes from the published initial
# state onto the limit cycle (the cycle tests skip the first two peaks the same way)

# Shift+click food.
CLICK_AMOUNT = 5.0  # our assumption: nutrient added per click, about a cycle's food for 2 cells
CLICK_RADIUS = 1.0  # our assumption: spread over a disc of this radius, dish units


@dataclass(frozen=True, slots=True)
class CellParams:
    """The colony's tunables (PARAMS), checked and typed."""

    initial_cells: int
    nutrient_regen: float  # 1/s
    diffusion: float  # dish units^2/s
    upkeep_cost: float  # energy per model minute per dish unit^2 of cell
    time_scale: float  # model minutes per sim second

    @classmethod
    def from_dict(cls, params: Mapping[str, float]) -> CellParams:
        """Build from a params dict; missing keys take PARAMS defaults.

        Raises ValueError for an unknown key, a non-number, a non-finite or negative value,
        or time_scale <= 0. Values are not clamped to the Param ranges (Session does that).
        """
        unknown = sorted(set(params) - set(PARAMS))
        if unknown:
            raise ValueError(f"unknown cell parameters {unknown}")
        values: dict[str, float] = {}
        for key, param in PARAMS.items():
            raw = params.get(key, param.default)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be a number, got {raw!r}") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be finite and >= 0, got {raw!r}")
            values[key] = value
        if values["time_scale"] <= 0.0:
            raise ValueError(f"time_scale must be > 0, got {values['time_scale']!r}")
        return cls(
            initial_cells=round(values["initial_cells"]),
            nutrient_regen=values["nutrient_regen"],
            diffusion=values["diffusion"],
            upkeep_cost=values["upkeep_cost"],
            time_scale=values["time_scale"],
        )
