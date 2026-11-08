import math

from alife.geometry import DISH_RADIUS, ZERO, Vec2, from_world, to_world


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
