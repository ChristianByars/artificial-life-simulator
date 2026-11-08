import math

import pytest

from alife.orbit import (
    DEFAULT_POSE,
    DIST_MAX,
    DIST_MIN,
    PITCH_MAX,
    PITCH_MIN,
    OrbitState,
    focus,
    from_pose,
    orbit,
    pan,
    to_pose,
    zoom,
)


def default_state() -> OrbitState:
    return from_pose(DEFAULT_POSE)


def test_default_pose_roundtrip():
    pose = to_pose(from_pose(DEFAULT_POSE))
    assert pose["position"] == pytest.approx(DEFAULT_POSE["position"], abs=1e-9)
    assert pose["rotation"] == pytest.approx(DEFAULT_POSE["rotation"], abs=1e-9)


def test_default_target_on_ground():
    assert from_pose(DEFAULT_POSE).target[1] == pytest.approx(0, abs=1e-9)


def test_pose_roundtrip_off_default():
    pose = {"position": [4.0, 9.0, 7.0], "rotation": [50.0, 200.0, 0.0]}
    back = to_pose(from_pose(pose))
    assert back["position"] == pytest.approx(pose["position"], abs=1e-9)
    assert back["rotation"] == pytest.approx(pose["rotation"], abs=1e-9)


def test_pitch_clamped():
    s = default_state()
    assert orbit(s, 0, 500).pitch == PITCH_MAX == 89
    assert orbit(s, 0, -500).pitch == PITCH_MIN == 10


def test_yaw_not_clamped():
    assert orbit(default_state(), 90, 0).yaw == pytest.approx(90)


def test_zoom_clamped():
    s = default_state()
    assert zoom(s, 1e6).distance == DIST_MAX == 90
    assert zoom(s, 1e-6).distance == DIST_MIN == 2


def test_far_pose_survives_a_save():  # #37: Continue keeps a camera zoomed out past 60
    far = OrbitState(target=(0.0, 0.0, 0.0), yaw=0.0, pitch=45.0, distance=85.0)
    assert from_pose(to_pose(far)).distance == pytest.approx(85.0)


def test_zoom_scales_distance():
    s = default_state()
    assert zoom(s, 1.1).distance == pytest.approx(s.distance * 1.1)


def test_pan_follows_yaw():
    s = OrbitState(target=(0.0, 0.0, 0.0), yaw=90.0, pitch=35.0, distance=10.0)
    moved = pan(s, 0, 1)
    assert moved.target[0] > 0.9
    assert moved.target[1] == 0
    assert moved.target[2] == pytest.approx(0, abs=1e-9)


def test_pan_right_at_yaw_zero():
    s = OrbitState(target=(0.0, 0.0, 0.0), yaw=0.0, pitch=35.0, distance=10.0)
    assert pan(s, 1, 0).target == pytest.approx((1.0, 0.0, 0.0))


def test_level_camera_no_division_by_zero():
    s = from_pose({"position": [0, 5, 0], "rotation": [0, 0, 0]})
    assert all(math.isfinite(v) for v in s.target)
    assert math.isfinite(s.distance) and math.isfinite(s.yaw)
    assert s.pitch == 10


def test_upward_camera_is_clamped():
    s = from_pose({"position": [0, 5, 0], "rotation": [-30, 0, 0]})
    assert s.pitch == 10
    assert all(math.isfinite(v) for v in s.target)


def test_nonfinite_pose_rejected():
    with pytest.raises(ValueError):
        from_pose({"position": [0, float("nan"), 0], "rotation": [35, 0, 0]})


def test_focus_moves_target():
    s = default_state()
    f = focus(s, (3, 0, 4))
    assert f.target == (3, 0, 4)
    assert (f.yaw, f.pitch, f.distance) == (s.yaw, s.pitch, s.distance)


def test_to_pose_looks_at_target():
    s = OrbitState(target=(1.0, 0.0, 2.0), yaw=30.0, pitch=40.0, distance=12.0)
    pose = to_pose(s)
    px, py, pz = pose["position"]
    assert math.dist((px, py, pz), s.target) == pytest.approx(12.0)
    assert pose["rotation"] == pytest.approx([40.0, 30.0, 0.0])
