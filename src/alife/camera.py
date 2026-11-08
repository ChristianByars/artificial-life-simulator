"""Orbit camera controller (GUI module: imports Ursina).

A thin ``Entity`` that turns mouse and key input into ``alife.orbit`` calls and applies the
resulting pose to ``ursina.camera`` each frame. All maths lives in ``alife.orbit``.

Input (SPEC section 5): right-drag orbits, middle-drag and WASD pan, scroll zooms.
"""

from __future__ import annotations

from collections.abc import Callable

from ursina import Entity, camera, held_keys, mouse, time

from alife import orbit as om

# Tunables: all our assumptions (feel, not biology).
ORBIT_DEG_PER_SCREEN = 180.0  # degrees of orbit per screen-height of right-drag  # our assumption
MOUSE_PAN_PER_SCREEN = 1.0  # target moves this many distances per screen-height  # our assumption
KEY_PAN_PER_SECOND = 0.8  # WASD pan, in camera distances per second  # our assumption
ZOOM_STEP = 1.1  # distance factor per scroll notch (brief)

# Ursina input names (ursina/input_handler.py Keys, ursina/mouse.py).
SCROLL_UP = "scroll up"
SCROLL_DOWN = "scroll down"


def held_pan_vector() -> tuple[float, float]:
    """(right, forward) from W A S D, each -1..1.

    Reads ``held_keys`` directly until F5's ``keys.pan_vector`` lands; F5 swaps this call by
    passing its own function as ``OrbitCamera(pan_source=...)``.
    """
    right = held_keys["d"] - held_keys["a"]
    forward = held_keys["w"] - held_keys["s"]
    return float(right), float(forward)


class OrbitCamera(Entity):
    def __init__(
        self,
        pose: dict[str, list[float]] | None = None,
        pan_source: Callable[[], tuple[float, float]] = held_pan_vector,
    ) -> None:
        super().__init__(name="orbit_camera")
        self.pan_source = pan_source
        self.state = om.from_pose(pose if pose is not None else om.DEFAULT_POSE)
        self._apply()

    def get_pose(self) -> dict[str, list[float]]:
        return om.to_pose(self.state)

    def set_pose(self, pose: dict[str, list[float]]) -> None:
        self.state = om.from_pose(pose)
        self._apply()

    def focus(self, point: om.Vec3) -> None:
        self.state = om.focus(self.state, point)
        self._apply()

    def input(self, key: str) -> None:
        if key == SCROLL_UP:
            self.state = om.zoom(self.state, 1 / ZOOM_STEP)
        elif key == SCROLL_DOWN:
            self.state = om.zoom(self.state, ZOOM_STEP)

    def update(self) -> None:
        s = self.state
        # mouse.velocity is in screen-height fractions per frame (zero outside the window).
        vx, vy = mouse.velocity.x, mouse.velocity.y
        if mouse.right:
            # Grab-the-world: dragging right swings the view right; dragging down looks more down.
            s = om.orbit(s, vx * ORBIT_DEG_PER_SCREEN, -vy * ORBIT_DEG_PER_SCREEN)
        if mouse.middle:
            k = MOUSE_PAN_PER_SCREEN * s.distance
            s = om.pan(s, -vx * k, -vy * k)
        right, forward = self.pan_source()
        if right or forward:
            step = KEY_PAN_PER_SECOND * s.distance * time.dt
            s = om.pan(s, right * step, forward * step)
        if s != self.state:
            self.state = s
            self._apply()

    def _apply(self) -> None:
        pose = om.to_pose(self.state)
        camera.position = tuple(pose["position"])
        camera.rotation = tuple(pose["rotation"])
