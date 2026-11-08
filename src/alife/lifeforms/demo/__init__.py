"""Developer placeholder life form: dots doing random walks.

Exempt from the biology rule (not biology); removed in Plan 2. Keep this module headless:
the view is imported inside ``build_view``.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from alife.geometry import Vec2
from alife.lifeforms import OUR_ASSUMPTION, InfoNote, Param, Source

from .sim import WalkerSim

if TYPE_CHECKING:
    from ursina import Entity

    from .view import DemoView


class DemoLifeform:
    id = "demo"
    name = "Random walkers (dev)"
    summary = "Developer placeholder: dots doing random walks."
    biology = [
        InfoNote(
            "Developer placeholder",
            "Not a biological model. Exists to test the app loop.",
            assumed=True,
        )
    ]
    sources: list[Source] = []
    params = {"count": Param("Walkers", 30, 1, 300, OUR_ASSUMPTION)}

    def __init__(self) -> None:
        self.sim: WalkerSim | None = None
        self.view: DemoView | None = None

    def init(self, rng: random.Random, params: dict) -> None:
        self.sim = WalkerSim(rng, int(params["count"]))

    def step(self, dt: float) -> None:
        self.sim.step(dt)

    def stats(self) -> dict[str, str | float]:
        return self.sim.stats()

    def get_state(self) -> dict:
        return self.sim.get_state()

    def set_state(self, state: dict) -> None:
        self.sim.set_state(state)

    def build_view(self, root: Entity) -> None:
        from .view import DemoView  # lazy: pulls in ursina

        self.view = DemoView(root)

    def sync_view(self) -> None:
        if self.view is not None:
            self.view.sync(self.sim)

    def on_pick(self, entity_id: int | None) -> None:
        self.sim.on_pick(entity_id)

    def on_dish_click(self, world_pos: Vec2, mods: set) -> None:
        self.sim.on_dish_click(world_pos, mods)


LIFEFORM = DemoLifeform
