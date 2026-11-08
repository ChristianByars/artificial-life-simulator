"""Cell colony simulation (SPEC §3.3, §6.1). Sim code, no Ursina.

Each cell is a plain object that owns one Tyson 1991 cell-cycle model (CellCycle). The colony
steps them on a shared nutrient grid. One tick, in SPEC §6.1 order:

1. Environment: the nutrient grid diffuses and regrows (Environment.step).
2. Motion: Brownian step plus soft-sphere repulsion (spatial hash rebuilt each tick), then
   clamped to the dish.
3. Feeding: Michaelis-Menten uptake from the grid under the cell (Environment.take).
4. Upkeep proportional to area; surplus energy above ENERGY_MAX grows the radius.
5. Cell cycle: synthesis scale = clamp(energy / ENERGY_REF, 0, 1), then advance by
   dt x time_scale model minutes.
6. Division: MPF falls back below MPF_THRESHOLD, at least MIN_DIVISION_TIME after the last one.
7. Death: energy <= 0; the cell fades for DYING_SECONDS, then is removed.

Steps 3-7 are Cell.step, run per cell in ascending id order (SPEC §3.5). All randomness comes
from the rng the colony is given (the session's).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from alife.geometry import DISH_RADIUS, ZERO, Vec2
from alife.lifeforms.cell.cycle import MPF_THRESHOLD, CellCycle
from alife.lifeforms.cell.environment import Environment
from alife.lifeforms.cell.motion import brownian_step, clamp_to_dish, repulsion
from alife.lifeforms.cell.params import (
    CLICK_AMOUNT,
    CLICK_RADIUS,
    DYING_SECONDS,
    ENERGY_MAX,
    ENERGY_REF,
    GROWTH_COST,
    KM,
    MAX_AREA,
    MAX_RADIUS,
    MIN_DIVISION_TIME,
    MOTILITY,
    REFERENCE_PERIOD,
    SEED_ENERGY,
    SEED_WARMUP,
    SPREAD,
    STIFFNESS,
    VMAX,
    YIELD,
    CellParams,
)
from alife.spatial import SpatialHash

__all__ = [
    "Cell",
    "CellColony",
    "CellPhase",
    "Died",
    "Divided",
    "Environment",
    "Event",
    "MitosisEnded",
    "Mitochondrion",
    "Nucleus",
]

# Organelle layout, for the view only (our assumption). Offsets are in units of the cell radius.
NUCLEUS_FRACTION = 0.4  # our assumption: nucleus radius / cell radius
MITOCHONDRIA_PER_CELL = 4  # our assumption: a few to draw; real cells have far more
MITOCHONDRIA_RING = (0.55, 0.8)  # our assumption: between the nucleus and the membrane

# Float slack when comparing the summed fade time with DYING_SECONDS.
_FADE_EPSILON = 1e-9


class CellPhase(Enum):
    """Shown phase. Tyson 1991 has no S phase, so this is only mitosis or not (SPEC §6.1)."""

    INTERPHASE = "interphase"
    MITOSIS = "mitosis"


@dataclass(frozen=True, slots=True)
class Nucleus:
    radius: float  # dish units


@dataclass(frozen=True, slots=True)
class Mitochondrion:
    offset: Vec2  # from the cell centre, in units of the cell radius
    angle: float  # long-axis orientation, radians


@dataclass(frozen=True, eq=False)
class Divided:
    """The colony split `parent` into two `daughters` this tick."""

    parent: Cell
    daughters: tuple[Cell, Cell]


@dataclass(frozen=True, eq=False)
class Died:
    """`cell` ran out of energy this tick and started to fade."""

    cell: Cell


@dataclass(frozen=True, eq=False)
class MitosisEnded:
    """Cell.step's request to divide: MPF fell below the threshold at the end of mitosis.

    The colony turns it into a Divided, since the daughters need its ids and rng.
    """

    cell: Cell


Event = Divided | Died | MitosisEnded


@dataclass(eq=False)
class Cell:
    """One eukaryotic cell on the dish (SPEC §3.3)."""

    id: int
    position: Vec2  # dish units
    velocity: Vec2  # dish units per sim second, over the last tick
    radius: float  # dish units
    energy: float  # units of ENERGY_REF; <= 0 means dying
    age: float  # model minutes since birth (or seeding), while alive
    generation: int  # divisions since a seeded ancestor
    cycle: CellCycle
    nucleus: Nucleus
    mitochondria: list[Mitochondrion]
    dying: float = 0.0  # sim seconds spent fading; used only while is_dying
    mpf_high: bool = False  # MPF above MPF_THRESHOLD at the end of the last tick
    since_division: float = 0.0  # model minutes since birth or seeding

    @property
    def phase(self) -> CellPhase:
        """Derived from the cycle state, not stored (SPEC §3.3)."""
        return CellPhase.MITOSIS if self.cycle.mpf > MPF_THRESHOLD else CellPhase.INTERPHASE

    @property
    def is_dying(self) -> bool:
        return self.energy <= 0.0

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius

    def step(self, env: Environment, dt: float, params: CellParams) -> list[Event]:
        """Steps 3-7 of one tick of `dt` sim seconds (plan M11: takes the colony's params).

        Returns MitosisEnded when the cell should divide and Died when it starts dying.
        A dying cell only counts its fade time: it doesn't move, feed or cycle.
        """
        if self.is_dying:
            self.dying += dt
            return []
        dt_model = dt * params.time_scale
        self._feed(env, dt_model, params.upkeep_cost)
        self.cycle.set_synthesis_scale(min(max(self.energy / ENERGY_REF, 0.0), 1.0))
        self.cycle.advance(dt_model)
        self.age += dt_model
        self.since_division += dt_model
        high = self.cycle.mpf > MPF_THRESHOLD
        mitosis_ended = self.mpf_high and not high
        self.mpf_high = high
        events: list[Event] = []
        if mitosis_ended and self.since_division >= MIN_DIVISION_TIME and self.energy > 0.0:
            events.append(MitosisEnded(self))
        if self.energy <= 0.0:
            self.dying = 0.0
            self.velocity = ZERO
            events.append(Died(self))
        return events

    def _feed(self, env: Environment, dt_model: float, upkeep_cost: float) -> None:
        """Feeding, upkeep and growth (steps 3-4)."""
        area = self.area
        n = float(env.nutrient[env.index(self.position)])
        uptake = VMAX * area * n / (KM + n) * dt_model  # Michaelis-Menten, nutrient mass
        upkeep = upkeep_cost * area * dt_model
        # our assumption: a cell takes only what it can use (refill, grow, pay this tick's
        # upkeep), so full adults leave food for others
        room = ENERGY_MAX - self.energy + GROWTH_COST * max(MAX_AREA - area, 0.0) + upkeep
        demand = min(uptake, max(room, 0.0) / YIELD)
        self.energy += YIELD * env.take(self.position, self.radius, demand)
        self.energy -= upkeep
        surplus = self.energy - ENERGY_MAX
        if surplus > 0.0:
            self.energy = ENERGY_MAX
            grown = min(area + surplus / GROWTH_COST, MAX_AREA)
            self._resize(min(math.sqrt(grown / math.pi), MAX_RADIUS))

    def _resize(self, radius: float) -> None:
        self.radius = radius
        self.nucleus = Nucleus(NUCLEUS_FRACTION * radius)

    def divide(self, ids: tuple[int, int], rng: random.Random) -> tuple[Cell, Cell]:
        """Split into two daughters (SPEC §6.1 step 6), with the given ids.

        Each gets half the energy and half the area (radius / sqrt 2), a copy of the cycle
        state with parameters spread around the published values (SPREAD), and generation + 1.
        They sit at p +- u r / 2 along a random direction u, clamped inside the dish.
        """
        angle = rng.random() * math.tau
        u = Vec2(math.cos(angle), math.sin(angle))
        radius = self.radius / math.sqrt(2.0)
        daughters: list[Cell] = []
        for cid, side in zip(ids, (1.0, -1.0), strict=True):
            daughters.append(
                Cell(
                    id=cid,
                    position=clamp_to_dish(self.position + u * (side * self.radius / 2), radius),
                    velocity=ZERO,
                    radius=radius,
                    energy=self.energy / 2,
                    age=0.0,
                    generation=self.generation + 1,
                    cycle=self.cycle.copy(rng, SPREAD),
                    nucleus=Nucleus(NUCLEUS_FRACTION * radius),
                    mitochondria=_random_mitochondria(rng),
                    mpf_high=self.mpf_high,
                    since_division=0.0,
                )
            )
        return daughters[0], daughters[1]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot, including the cycle (from_dict rebuilds it bitwise)."""
        return {
            "id": self.id,
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
            "radius": self.radius,
            "energy": self.energy,
            "age": self.age,
            "generation": self.generation,
            "dying": self.dying,
            "mpf_high": self.mpf_high,
            "since_division": self.since_division,
            "cycle": self.cycle.to_dict(),
            "nucleus": {"radius": self.nucleus.radius},
            "mitochondria": [
                {"offset": m.offset.to_list(), "angle": m.angle} for m in self.mitochondria
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Cell:
        """Rebuild a to_dict() snapshot; raises ValueError if anything is missing or malformed."""
        try:
            mpf_high = d["mpf_high"]
            if not isinstance(mpf_high, bool):
                raise ValueError(f"mpf_high must be a bool, got {mpf_high!r}")
            return cls(
                id=_int(d["id"], "id", minimum=1),
                position=_vec(d["position"], "position"),
                velocity=_vec(d["velocity"], "velocity"),
                radius=_positive(d["radius"], "radius"),
                energy=_finite(d["energy"], "energy"),
                age=_finite(d["age"], "age"),
                generation=_int(d["generation"], "generation", minimum=0),
                cycle=CellCycle.from_dict(d["cycle"]),
                nucleus=Nucleus(_positive(d["nucleus"]["radius"], "nucleus radius")),
                mitochondria=[
                    Mitochondrion(_vec(m["offset"], "offset"), _finite(m["angle"], "angle"))
                    for m in d["mitochondria"]
                ],
                dying=_finite(d["dying"], "dying"),
                mpf_high=mpf_high,
                since_division=_finite(d["since_division"], "since_division"),
            )
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(f"bad cell state: {exc!r}") from exc


class CellColony:
    """The headless cell colony (SPEC §6.1) that the Plan 2 `cell` life form wraps."""

    def __init__(self, rng: random.Random, params: Mapping[str, float] | CellParams) -> None:
        self.rng = rng
        self.params = params if isinstance(params, CellParams) else CellParams.from_dict(params)
        self.env = Environment(
            diffusion=self.params.diffusion, regen_rate=self.params.nutrient_regen
        )
        self.cells: dict[int, Cell] = {}
        self.time = 0.0  # sim seconds
        self.births = 0  # divisions
        self.deaths = 0
        self._next_id = 1
        # bucket size = the largest contact distance, so repulsion pairs look one bucket out
        self._grid = SpatialHash(2.0 * MAX_RADIUS)
        for _ in range(self.params.initial_cells):
            self._seed_cell()

    # simulation

    def step(self, dt: float) -> list[Event]:
        """Advance one tick of `dt` sim seconds; return the tick's Divided and Died events."""
        if not 0.0 < dt < math.inf:
            raise ValueError(f"dt must be positive and finite, got {dt!r}")
        self.env.step(dt)
        self._move(dt)
        events: list[Event] = []
        for cid in sorted(self.cells):
            cell = self.cells[cid]
            for event in cell.step(self.env, dt, self.params):
                if isinstance(event, MitosisEnded):
                    events.append(self._divide(cell))
                elif isinstance(event, Died):
                    self.deaths += 1
                    events.append(event)
            if cell.is_dying and cell.dying >= DYING_SECONDS - _FADE_EPSILON:
                del self.cells[cid]
        self.time += dt
        return events

    def _move(self, dt: float) -> None:
        """Brownian motion plus soft-sphere repulsion, clamped to the dish (step 2)."""
        movers = [self.cells[cid] for cid in sorted(self.cells) if not self.cells[cid].is_dying]
        if not movers:
            return
        self._grid.rebuild((c.id, c.position) for c in movers)
        push = repulsion([(c.id, c.position, c.radius) for c in movers], self._grid, STIFFNESS, dt)
        for c in movers:
            start = c.position
            moved = brownian_step(start, MOTILITY, dt, self.rng) + push[c.id]
            c.position = clamp_to_dish(moved, c.radius)
            c.velocity = (c.position - start) / dt

    def _divide(self, cell: Cell) -> Divided:
        ids = (self._next_id, self._next_id + 1)
        self._next_id += 2
        daughters = cell.divide(ids, self.rng)
        del self.cells[cell.id]
        for d in daughters:
            self.cells[d.id] = d
        self.births += 1
        return Divided(cell, daughters)

    def _seed_cell(self) -> None:
        """Plate an adult cell at a random spot and point of its cycle (our assumption)."""
        rng = self.rng
        r = (DISH_RADIUS - MAX_RADIUS) * math.sqrt(rng.random())  # uniform over the disc
        a = rng.random() * math.tau
        cycle = CellCycle.fresh().copy(rng, SPREAD)
        cycle.advance(SEED_WARMUP + rng.random() * REFERENCE_PERIOD)
        cid = self._next_id
        self._next_id += 1
        self.cells[cid] = Cell(
            id=cid,
            position=Vec2(r * math.cos(a), r * math.sin(a)),
            velocity=ZERO,
            radius=MAX_RADIUS,
            energy=SEED_ENERGY,
            age=0.0,
            generation=0,
            cycle=cycle,
            nucleus=Nucleus(NUCLEUS_FRACTION * MAX_RADIUS),
            mitochondria=_random_mitochondria(rng),
            mpf_high=cycle.mpf > MPF_THRESHOLD,
            since_division=MIN_DIVISION_TIME,  # no last division to wait for
        )

    # interaction

    def add_nutrient(self, p: Vec2) -> None:
        """Drop CLICK_AMOUNT of nutrient around `p` (Shift+click)."""
        self.env.add(p, CLICK_AMOUNT, CLICK_RADIUS)

    # HUD

    def stats(self) -> dict[str, str | float]:
        """SPEC §6.1 stats over living cells; oldest_generation is the deepest generation alive."""
        alive = [self.cells[i] for i in sorted(self.cells) if not self.cells[i].is_dying]
        mitosis = sum(c.phase is CellPhase.MITOSIS for c in alive)
        return {
            "population": len(alive),
            "births": self.births,
            "deaths": self.deaths,
            "mean_energy": math.fsum(c.energy for c in alive) / len(alive) if alive else 0.0,
            "oldest_generation": max((c.generation for c in alive), default=0),
            "interphase": len(alive) - mitosis,
            "mitosis": mitosis,
        }

    # save / load

    def get_state(self) -> dict[str, Any]:
        """JSON-safe state: cells (with their cycles), the nutrient grid, counters and time."""
        return {
            "time": self.time,
            "next_id": self._next_id,
            "births": self.births,
            "deaths": self.deaths,
            "cells": [self.cells[i].to_dict() for i in sorted(self.cells)],
            "env": self.env.to_dict(),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore get_state() output; raises ValueError, changing nothing, if it is malformed."""
        try:
            time = _finite(state["time"], "time")
            next_id = _int(state["next_id"], "next_id", minimum=1)
            births = _int(state["births"], "births", minimum=0)
            deaths = _int(state["deaths"], "deaths", minimum=0)
            env = Environment.from_dict(state["env"])
            cells: dict[int, Cell] = {}
            for d in state["cells"]:
                cell = Cell.from_dict(d)
                if cell.id in cells:
                    raise ValueError(f"duplicate cell id {cell.id}")
                if cell.id >= next_id:
                    raise ValueError(f"cell id {cell.id} is not below next_id {next_id}")
                cells[cell.id] = cell
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(f"bad colony state: {exc!r}") from exc
        self.time = time
        self._next_id = next_id
        self.births = births
        self.deaths = deaths
        self.env = env
        self.cells = dict(sorted(cells.items()))


def _random_mitochondria(rng: random.Random) -> list[Mitochondrion]:
    lo, hi = MITOCHONDRIA_RING
    out: list[Mitochondrion] = []
    for _ in range(MITOCHONDRIA_PER_CELL):
        rho = rng.uniform(lo, hi)
        theta = rng.random() * math.tau
        offset = Vec2(rho * math.cos(theta), rho * math.sin(theta))
        out.append(Mitochondrion(offset, rng.random() * math.pi))
    return out


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a number, got {value!r}") from exc
    if not math.isfinite(x):
        raise ValueError(f"{what} is not finite: {value!r}")
    return x


def _positive(value: Any, what: str) -> float:
    x = _finite(value, what)
    if x <= 0.0:
        raise ValueError(f"{what} must be > 0, got {value!r}")
    return x


def _int(value: Any, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{what} must be an int >= {minimum}, got {value!r}")
    return value


def _vec(value: Sequence[Any], what: str) -> Vec2:
    if len(value) != 2:
        raise ValueError(f"{what} must have 2 components, got {value!r}")
    return Vec2(_finite(value[0], what), _finite(value[1], what))
