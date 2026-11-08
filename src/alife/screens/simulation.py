"""Simulation screen: runs one life form and owns its 3D scene (SPEC 3.1). GUI-only.

F5 adds the HUD and controls; F6 adds autosave.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ursina import Entity, destroy, time

from alife.camera import OrbitCamera
from alife.session import Session
from alife.ui import destroy_tree

if TYPE_CHECKING:
    from alife.app import App

ESCAPE = "escape"  # ursina/input_handler.py: Keys.escape


class SimulationScreen(Entity):
    """Steps ``session`` once per frame and draws it under ``root``.

    The life form builds its view under ``root`` (``build_view``); the orbit camera (F4) lives
    there too, so closing the screen removes everything it made.
    """

    def __init__(self, app: App, session: Session) -> None:
        super().__init__(name="simulation_screen")
        self.app = app
        self.session = session
        self.root = Entity(name="simulation_root")
        self.orbit_camera = OrbitCamera()
        self.orbit_camera.parent = self.root
        session.lifeform.build_view(self.root)
        session.lifeform.sync_view()  # draw the starting state on the first frame

    def update(self) -> None:
        self.session.advance(time.dt)
        self.session.lifeform.sync_view()

    def input(self, key: str) -> bool:
        if key == ESCAPE:
            self.app.show_menu()
            return True  # consume it, so the menu just built does not also get this Esc
        return False

    def close(self) -> None:
        """Remove the screen and its scene. Runs on Esc and on quit."""
        destroy_tree(self.root)
        destroy(self)
