"""Developer placeholder sim: Gaussian random walkers. Not a biological model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from alife.geometry import DISH_RADIUS, Vec2

STEP_SIGMA = 0.05  # our assumption: per-tick step size, world units
SPAWN_FRACTION = 0.9  # our assumption: initial walkers start inside this fraction of the dish


@dataclass
class Walker:
    id: int
    position: Vec2


def clamp_to_dish(p: Vec2) -> Vec2:
    n = p.length()
    return p if n <= DISH_RADIUS else p * (DISH_RADIUS / n)


class WalkerSim:
    def __init__(self, rng: random.Random, count: int) -> None:
        self.rng = rng
        self.walkers: list[Walker] = []
        self.next_id = 0
        self.selected: int | None = None
        for _ in range(count):
            r = DISH_RADIUS * SPAWN_FRACTION * math.sqrt(rng.random())  # uniform over the disc
            a = rng.random() * math.tau
            self.add(Vec2(r * math.cos(a), r * math.sin(a)))

    def add(self, position: Vec2) -> Walker:
        walker = Walker(self.next_id, clamp_to_dish(position))
        self.next_id += 1
        self.walkers.append(walker)  # ids only grow, so the list stays sorted by id
        return walker

    def step(self, dt: float) -> None:
        # One step per fixed tick; dt is unused because the walk is defined per tick.
        for w in self.walkers:
            w.position = clamp_to_dish(
                w.position + Vec2(self.rng.gauss(0.0, STEP_SIGMA), self.rng.gauss(0.0, STEP_SIGMA))
            )

    def stats(self) -> dict[str, str | float]:
        return {"walkers": len(self.walkers)}

    def get_state(self) -> dict:
        return {
            "next_id": self.next_id,
            "selected": self.selected,
            "walkers": [{"id": w.id, "position": w.position.to_list()} for w in self.walkers],
        }

    def set_state(self, state: dict) -> None:
        self.next_id = int(state["next_id"])
        self.selected = state["selected"]
        self.walkers = [
            Walker(int(d["id"]), Vec2.from_list(d["position"])) for d in state["walkers"]
        ]

    def on_dish_click(self, world_pos: Vec2, mods: set) -> None:
        if "shift" in mods:
            self.add(world_pos)

    def on_pick(self, entity_id: int | None) -> None:
        self.selected = entity_id
