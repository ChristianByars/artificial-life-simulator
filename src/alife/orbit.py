"""Orbit camera maths (headless: no Ursina import).

Angles are degrees. Ursina convention: ``rotation_x`` is pitch (positive looks down) and
``rotation_y`` is yaw. The camera looks along

    forward = (sin yaw * cos pitch, -sin pitch, cos yaw * cos pitch)

and sits at ``target - forward * distance``. Every operation returns a new, clamped
:class:`OrbitState`; nothing mutates in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]

# Default camera pose (Ursina position / rotation; SPEC section 4 format). Our assumption (ruling
# R18): 45 degrees down, aimed at the dish centre, 32 * sqrt(2) = 45.3 units away. Ursina's
# horizontal FOV is 40 degrees, so a wide window is limited by its height. The radius-10 rim
# reaches 91% of the way from the frame's centre to its edge at 16:9, and 78% in Ursina's default
# window on this Mac (1209 x 785). SPEC section 4's example pose ([0, 8, -12] / [35, 0, 0]) showed
# only part of the dish (F2).
DEFAULT_POSE: dict[str, list[float]] = {"position": [0, 32, -32], "rotation": [45, 0, 0]}

# M5: pitch and distance limits. Our assumption (view comfort, not biology).
PITCH_MIN = 10.0  # our assumption
PITCH_MAX = 89.0  # our assumption
DIST_MIN = 2.0  # our assumption
DIST_MAX = 90.0  # our assumption: about 2x the default 45.3 (60 left only 1.3x of zoom-out)

# A level or upward-looking camera never meets the ground plane; pretend the target is this far
# ahead instead (F4 brief). Our assumption.
_LEVEL_TARGET_DISTANCE = 14.0
_LEVEL_EPS = 1e-6


@dataclass(frozen=True)
class OrbitState:
    target: Vec3
    yaw: float
    pitch: float
    distance: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamped(target: Vec3, yaw: float, pitch: float, distance: float) -> OrbitState:
    return OrbitState(
        target=(float(target[0]), float(target[1]), float(target[2])),
        yaw=float(yaw),
        pitch=_clamp(float(pitch), PITCH_MIN, PITCH_MAX),
        distance=_clamp(float(distance), DIST_MIN, DIST_MAX),
    )


def _forward(yaw_deg: float, pitch_deg: float) -> Vec3:
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    return (math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch))


def to_pose(s: OrbitState) -> dict[str, list[float]]:
    """Camera pose in the save format (SPEC section 4): Ursina position and rotation."""
    fx, fy, fz = _forward(s.yaw, s.pitch)
    tx, ty, tz = s.target
    return {
        "position": [tx - fx * s.distance, ty - fy * s.distance, tz - fz * s.distance],
        "rotation": [s.pitch, s.yaw, 0.0],
    }


def from_pose(pose: dict[str, list[float]]) -> OrbitState:
    """Recover an orbit state; the target is where the view ray meets y = 0 (M5).

    Raises ValueError for a malformed or non-finite pose.
    """
    try:
        px, py, pz = (float(v) for v in pose["position"])
        pitch, yaw = (float(v) for v in pose["rotation"][:2])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"bad camera pose: {pose!r}") from exc
    if not all(math.isfinite(v) for v in (px, py, pz, pitch, yaw)):
        raise ValueError(f"camera pose is not finite: {pose!r}")

    fx, fy, fz = _forward(yaw, pitch)
    t = -py / fy if fy < -_LEVEL_EPS else -1.0  # distance along the ray to y = 0
    if t > 0.0:
        target = (px + fx * t, 0.0, pz + fz * t)
    else:
        # Level, upward, or below the ground: no ground hit in front of the camera.
        t = _LEVEL_TARGET_DISTANCE
        target = (px + fx * t, py + fy * t, pz + fz * t)
    return _clamped(target, yaw, pitch, t)


def orbit(s: OrbitState, dyaw: float, dpitch: float) -> OrbitState:
    """Rotate around the target. Positive dpitch looks further down."""
    return _clamped(s.target, s.yaw + dyaw, s.pitch + dpitch, s.distance)


def pan(s: OrbitState, right: float, forward: float) -> OrbitState:
    """Move the target on the ground plane, relative to yaw (target y is kept)."""
    yaw = math.radians(s.yaw)
    sin_y, cos_y = math.sin(yaw), math.cos(yaw)
    tx, ty, tz = s.target
    # forward = (sin yaw, 0, cos yaw); right = (cos yaw, 0, -sin yaw)
    nx = tx + forward * sin_y + right * cos_y
    nz = tz + forward * cos_y - right * sin_y
    return _clamped((nx, ty, nz), s.yaw, s.pitch, s.distance)


def zoom(s: OrbitState, factor: float) -> OrbitState:
    """Scale the distance (factor < 1 zooms in)."""
    return _clamped(s.target, s.yaw, s.pitch, s.distance * factor)


def focus(s: OrbitState, point: Vec3) -> OrbitState:
    """Look at ``point``, keeping yaw, pitch and distance."""
    return _clamped(point, s.yaw, s.pitch, s.distance)
