import math

from alife.geometry import (
    DISH_RADIUS,
    ZERO,
    Vec2,
    dish_point,
    from_world,
    ray_ground_point,
    to_world,
)


def test_dish_radius():
    assert DISH_RADIUS == 10.0


def test_arithmetic():
    a, b = Vec2(1.0, 2.0), Vec2(3.0, -4.0)
    assert a + b == Vec2(4.0, -2.0)
    assert a - b == Vec2(-2.0, 6.0)
    assert a * 2 == Vec2(2.0, 4.0)
    assert 2 * a == Vec2(2.0, 4.0)
    assert b / 2 == Vec2(1.5, -2.0)
    assert -a == Vec2(-1.0, -2.0)
    assert ZERO == Vec2(0.0, 0.0)


def test_dot_and_length():
    a, b = Vec2(1.0, 2.0), Vec2(3.0, -4.0)
    assert a.dot(b) == -5.0
    assert b.length() == 5.0
    assert b.length_sq() == 25.0
    assert math.isclose(Vec2(1.0, 1.0).length(), math.sqrt(2))


def test_world_roundtrip():
    p = Vec2(1.5, -2.5)
    assert to_world(p) == (1.5, 0.0, -2.5)
    assert to_world(p, y=0.3) == (1.5, 0.3, -2.5)
    assert from_world(*to_world(p, y=7.0)) == p


def test_list_roundtrip():
    p = Vec2(0.1, -0.2)
    assert p.to_list() == [0.1, -0.2]
    assert Vec2.from_list(p.to_list()) == p
    assert Vec2.from_list((1, 2)) == Vec2(1.0, 2.0)


# --- the mouse ray meets the dish plane y = 0 (I1: Shift+click always acts on the dish) ----------


def test_ray_down_hits_ground():
    # From (1, 10, 2) straight down: the ground point under it, as sim (x, y) = world (x, z).
    assert ray_ground_point((1.0, 10.0, 2.0), (0.0, -1.0, 0.0)) == Vec2(1.0, 2.0)


def test_slanted_ray_hits_ground():
    # 45 degrees down along +z from height 4: meets y = 0 four units ahead.
    hit = ray_ground_point((0.0, 4.0, -3.0), (0.0, -2.0, 2.0))
    assert hit == Vec2(0.0, 1.0)


def test_parallel_ray_misses():
    assert ray_ground_point((0.0, 5.0, 0.0), (1.0, 0.0, 1.0)) is None
    assert ray_ground_point((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) is None  # lying in the plane


def test_ray_pointing_away_misses():
    assert ray_ground_point((0.0, 5.0, 0.0), (0.0, 1.0, 0.0)) is None  # above, looking up
    assert ray_ground_point((0.0, -5.0, 0.0), (0.0, -1.0, 0.0)) is None  # below, looking down


def test_nonfinite_ray_misses():
    nan, inf = math.nan, math.inf
    assert ray_ground_point((nan, 5.0, 0.0), (0.0, -1.0, 0.0)) is None
    assert ray_ground_point((0.0, 5.0, 0.0), (0.0, nan, 0.0)) is None
    assert ray_ground_point((0.0, inf, 0.0), (0.0, -1.0, 0.0)) is None
    assert ray_ground_point((0.0, 5.0, 0.0), (inf, -1.0, 0.0)) is None
    assert ray_ground_point((0.0, 5.0, 0.0), (1e-320, -1e-320, 0.0)) == Vec2(5.0, 0.0)


def test_dish_point_inside_rim():
    assert dish_point((3.0, 8.0, -4.0), (0.0, -1.0, 0.0)) == Vec2(3.0, -4.0)
    assert dish_point((DISH_RADIUS, 8.0, 0.0), (0.0, -1.0, 0.0)) == Vec2(DISH_RADIUS, 0.0)  # rim


def test_dish_point_outside_rim():
    assert dish_point((DISH_RADIUS + 0.01, 8.0, 0.0), (0.0, -1.0, 0.0)) is None
    assert dish_point((8.0, 8.0, 8.0), (0.0, -1.0, 0.0)) is None  # inside the square, not the disc
    assert dish_point((0.0, 5.0, 0.0), (1.0, 0.0, 0.0)) is None  # no hit at all
    assert dish_point((0.0, 5.0, 0.0), (0.0, 1.0, 0.0)) is None


def test_dish_point_radius_argument():
    assert dish_point((3.0, 1.0, 0.0), (0.0, -1.0, 0.0), radius=2.0) is None
    assert dish_point((3.0, 1.0, 0.0), (0.0, -1.0, 0.0), radius=3.0) == Vec2(3.0, 0.0)
