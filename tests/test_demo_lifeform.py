from random import Random

from alife.geometry import Vec2
from alife.lifeforms import default_params, registry


def test_demo_lifeform_end_to_end():
    cls = registry()["demo"]
    assert cls.id == "demo"
    assert cls.summary == "Developer placeholder: dots doing random walks."
    assert cls.biology[0].assumed and cls.sources == []
    lf = cls()
    lf.init(Random(1), {"count": 12})
    assert lf.stats() == {"walkers": 12}
    for _ in range(5):
        lf.step(1 / 60)
    state = lf.get_state()
    lf2 = cls()
    lf2.init(Random(1), default_params(cls))
    lf2.set_state(state)
    assert lf2.get_state() == state
    lf.on_dish_click(Vec2(0, 0), {"shift"})
    assert lf.stats() == {"walkers": 13}
    lf.on_pick(4)
    assert lf.sim.selected == 4
    assert "selected" not in lf.get_state()  # I3: selection is UI state, never saved
