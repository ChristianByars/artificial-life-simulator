"""Ursina view for the demo life form (GUI-only; not run headless)."""

from __future__ import annotations

from ursina import Entity, color
from ursina.models.procedural.circle import Circle

from alife.geometry import DISH_RADIUS, to_world

from .sim import WalkerSim

WALKER_RADIUS = 0.2  # our assumption: view scale
WALKER_COLOR = color.azure
SELECTED_COLOR = color.orange


class DemoView:
    """One sphere per walker (``pick_id``) on a flat dish (``is_dish``), under ``root``."""

    def __init__(self, root: Entity) -> None:
        self.root = root
        self.dish = Entity(
            parent=root,
            # Circle defaults to radius .5; pass DISH_RADIUS so the visible dish and its
            # collider match the sim's rim. No scale on the entity.
            model=Circle(resolution=64, radius=DISH_RADIUS),
            rotation_x=90,  # lie flat: the circle's normal points up
            color=color.dark_gray,
            collider="box",
        )
        self.dish.is_dish = True
        self.entities: dict[int, Entity] = {}

    def sync(self, sim: WalkerSim) -> None:
        live = set()
        for w in sim.walkers:
            live.add(w.id)
            e = self.entities.get(w.id)
            if e is None:
                e = Entity(
                    parent=self.root,
                    model="sphere",
                    scale=WALKER_RADIUS * 2,  # the 'sphere' model has diameter 1
                    collider="sphere",
                )
                e.pick_id = w.id
                self.entities[w.id] = e
            e.position = to_world(w.position, y=WALKER_RADIUS)
            e.color = SELECTED_COLOR if w.id == sim.selected else WALKER_COLOR
        for wid in [i for i in self.entities if i not in live]:
            self.entities.pop(wid).disable()
