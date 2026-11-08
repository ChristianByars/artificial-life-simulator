"""Simulation screen: runs one life form and owns its 3D scene, HUD and controls (SPEC 3.1, 5).

GUI-only. Keys map to actions in ``alife.keys``; the HUD text comes from ``alife.ui_text``.
Cmd/Ctrl+S saves with a toast; Esc autosaves, then shows the menu (the app autosaves on quit).
M5 adds the info panel (I).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ursina import Entity, camera, destroy, held_keys, mouse, time

from alife import keys
from alife.camera import OrbitCamera
from alife.geometry import from_world
from alife.save import Pose, SaveError, build_save
from alife.session import Session, new_seed
from alife.ui import destroy_tree
from alife.ui.confirm import ConfirmDialog
from alife.ui.hud import Hud
from alife.ui.toast import Toast
from alife.ui_text import CONFIRM_NEW, CONFIRM_RESET, SAVE_FAILED, SAVED, FpsMeter, hud_lines

if TYPE_CHECKING:
    from alife.app import App
    from alife.keys import Action


class SimulationScreen(Entity):
    """Steps ``session`` once per frame and draws it.

    3D: ``root`` holds the orbit camera (F4) and ``view_root``, where the life form builds its
    view; Reset and New rebuild only ``view_root``, so the camera keeps its pose. 2D: ``ui_root``
    (under ``camera.ui``) holds the HUD, the toast and any confirm dialog. Closing the screen
    destroys both trees. ``camera_pose`` (a restored save's) places the camera; None is the
    default pose.
    """

    def __init__(self, app: App, session: Session, camera_pose: Pose | None = None) -> None:
        super().__init__(name="simulation_screen")
        self.app = app
        self.session = session
        self.root = Entity(name="simulation_root")
        self.orbit_camera = OrbitCamera(camera_pose)  # WASD through keys.pan_vector (camera.py)
        self.orbit_camera.parent = self.root
        self.view_root = self._build_view()
        self.ui_root = Entity(parent=camera.ui, name="simulation_ui")
        self.hud = Hud(parent=self.ui_root)
        self.toast = Toast(parent=self.ui_root)
        self.dialog: ConfirmDialog | None = None
        self.picked: Entity | None = None  # the entity last picked by a click (C1: pick_id)
        self.fps = FpsMeter()
        # The first frame's time includes building the screen (and, at launch, opening the
        # window); counting it would show "FPS 1" for the first half second.
        self._fps_started = False
        self._handlers: dict[Action, Callable[[], object]] = {
            "pause": lambda: self.session.toggle_pause(),
            "step": lambda: self.session.step_once(),  # runs a tick only while paused
            "speed_up": lambda: self.session.speed_up(),
            "speed_down": lambda: self.session.speed_down(),
            "save": self._save_with_toast,
            "reset": lambda: self._ask(CONFIRM_RESET, self._reset),
            "new": lambda: self._ask(CONFIRM_NEW, self._new),
            "menu": self._to_menu,
            "focus": self._focus,
            # "info" (M5) is not wired yet.
        }
        self._refresh_hud()

    # per frame

    def update(self) -> None:
        self.session.advance(time.dt)
        self.session.lifeform.sync_view()
        if self._fps_started:
            self.fps.add(time.dt_unscaled)
        self._fps_started = True
        self._refresh_hud()

    def _refresh_hud(self) -> None:
        s, lifeform = self.session, self.session.lifeform
        self.hud.show(
            hud_lines(
                name=lifeform.name,
                tick=s.tick,
                sim_time=s.sim_time,
                speed=s.speed,
                fps=self.fps.fps,
                stats=lifeform.stats(),
                paused=s.paused,
                selected=self._picked_id(),
            )
        )

    # input

    def input(self, key: str) -> bool:
        """Returns True to eat ``key``: Ursina then stops offering it to later entities."""
        if self.dialog is not None:
            # Modal: Enter / Esc go to the dialog, everything else is dropped. Eating every key
            # keeps it from the orbit camera and any entity created after this screen. Clicks
            # still reach the dialog's buttons: Ursina's mouse.input runs after every entity.
            self.dialog.handle_key(key)
            return True
        mods = keys.held_mods(held_keys)
        if key == keys.LEFT_CLICK:
            self._click(mods)
            return False
        action = keys.action_for(key, mods)
        handler = self._handlers.get(action) if action is not None else None
        if handler is None:
            return False
        handler()
        return True  # e.g. the Esc that opened the menu must not reach the menu just built

    def _click(self, mods: frozenset[str]) -> None:
        hit = mouse.hovered_entity
        if hit is not None and hit.is_empty():
            hit = None  # destroyed (after Reset, or a closed dialog): Ursina keeps the reference
        if hit is not None and hit.has_ancestor(camera.ui):
            return  # a UI control: its own on_click handles the click
        lifeform = self.session.lifeform
        if keys.SHIFT in mods and getattr(hit, "is_dish", False):
            point = mouse.world_point
            if point is not None:
                lifeform.on_dish_click(from_world(*point), set(mods))
            return
        pick_id = getattr(hit, "pick_id", None)
        if pick_id is None:  # the dish, an unpickable entity, or empty space
            self.picked = None
            lifeform.on_pick(None)
        else:
            self.picked = hit
            lifeform.on_pick(pick_id)

    def _picked_entity(self) -> Entity | None:
        """The remembered pick, if it still exists: not destroyed and not disabled."""
        e = self.picked
        if e is None or e.is_empty() or not e.enabled:
            return None
        return e

    def _picked_id(self) -> int | None:
        e = self._picked_entity()
        return None if e is None else e.pick_id

    def _focus(self) -> None:
        e = self._picked_entity()
        if e is None:
            return
        x, _, z = (float(v) for v in e.world_position)
        if math.isfinite(x) and math.isfinite(z):
            # The ground point under the entity: orbit targets stay on y = 0 (decision M5).
            self.orbit_camera.focus((x, 0.0, z))

    # save and autosave

    def save(self) -> Path:
        """Write the session and camera pose to the life form's slot. Raises ``SaveError``."""
        data = build_save(self.session, self.orbit_camera.get_pose(), datetime.now().astimezone())
        return self.app.store.write(self.session.lifeform_cls.id, data)

    def autosave(self) -> bool:
        """Save without a toast; on failure print one line to stderr. Never raises: a failed
        autosave must not stop the menu or quitting."""
        try:
            self.save()
        except Exception as exc:  # SaveError, or a life form's get_state failing
            print(f"alife: autosave failed: {_one_line(exc)}", file=sys.stderr, flush=True)
            return False
        return True

    def _save_with_toast(self) -> None:
        try:
            self.save()
        except Exception as exc:  # SaveError, or a life form's get_state failing
            print(f"alife: save failed: {_one_line(exc)}", file=sys.stderr, flush=True)
            reason = exc.reason if isinstance(exc, SaveError) else type(exc).__name__
            self.toast.show(SAVE_FAILED.format(reason=reason))
            return
        self.toast.show(SAVED)

    def _to_menu(self) -> None:
        # Save while the session and camera still exist; show_menu() closes this screen.
        self.autosave()
        self.app.show_menu()

    # confirm dialog, reset and new

    def _ask(self, message: str, on_yes: Callable[[], None]) -> None:
        def confirmed() -> None:
            self._dialog_closed()
            on_yes()

        self.dialog = ConfirmDialog(
            message, on_confirm=confirmed, on_cancel=self._dialog_closed, parent=self.ui_root
        )
        self.orbit_camera.ignore = True  # no right-drag orbit, middle-drag or WASD pan meanwhile

    def _dialog_closed(self) -> None:
        self.dialog = None
        self.orbit_camera.ignore = False

    def _reset(self) -> None:
        self.session.reset()  # same seed and params; pause and speed are kept
        self._rebuild_view()

    def _new(self) -> None:
        self.session.new(new_seed())  # on this session, not app.start: --seed must not stick
        self._rebuild_view()

    def _build_view(self) -> Entity:
        view_root = Entity(parent=self.root, name="lifeform_view")
        self.session.lifeform.build_view(view_root)
        self.session.lifeform.sync_view()  # draw the starting state on the first frame
        return view_root

    def _rebuild_view(self) -> None:
        """The session holds a fresh life form: replace the old one's view with a new one."""
        self.picked = None
        destroy_tree(self.view_root)
        self.view_root = self._build_view()
        self._refresh_hud()

    def close(self) -> None:
        """Remove the screen, its scene and its UI. Runs on Esc and on quit."""
        destroy_tree(self.root)
        destroy_tree(self.ui_root)
        destroy(self)


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
