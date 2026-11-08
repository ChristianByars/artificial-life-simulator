"""CellCycle: the per-cell Tyson 1991 model wrapper (task S3, SPEC §6.1)."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys

import pytest
import roadrunner

from alife.lifeforms.cell.cycle import DEFAULT_TIME_SCALE, MODEL_PATH, MPF_THRESHOLD, CellCycle
from alife.session import TICK_DT
from tests.validation.helpers import peak_times

DT_MODEL = TICK_DT * DEFAULT_TIME_SCALE  # model minutes per tick
PERIOD = 35.60  # model minutes: reference period measured in M0 (SPIKE.md C2)
SYNTHESIS = "Reaction6_k1aa"  # cyclin synthesis, k1aa in Tyson 1991

# The ten rate constants of BIOMD0000000005, promoted from reaction-local to global ids.
KINETIC_PARAMS = {
    "Reaction1_k6",
    "Reaction2_k8notP",
    "Reaction3_k9",
    "Reaction4_k3",
    "Reaction5_k5notP",
    "Reaction6_k1aa",
    "Reaction7_k2",
    "Reaction8_k7",
    "Reaction9_k4prime",
    "Reaction9_k4",
}


def ticks(minutes: float) -> int:
    return round(minutes / DT_MODEL)


def run_mpf(cycle: CellCycle, minutes: float) -> tuple[list[float], list[float]]:
    """Advance tick by tick; return model time and MPF, sampled at the start and every tick."""
    t, mpf = [cycle.t], [cycle.mpf]
    for _ in range(ticks(minutes)):
        cycle.advance(DT_MODEL)
        t.append(cycle.t)
        mpf.append(cycle.mpf)
    return t, mpf


def advance_to_spike_upstroke(cycle: CellCycle) -> None:
    """Stop on the rising edge of the next MPF spike, the stiffest stretch of the cycle."""
    for _ in range(ticks(2 * PERIOD)):
        if cycle.mpf > 0.05:
            return
        cycle.advance(DT_MODEL)
    raise AssertionError("no MPF upstroke within two periods")


def assert_lockstep(a: CellCycle, b: CellCycle, n: int) -> float:
    """Advance both n ticks, asserting bitwise-equal state after each; return the MPF peak seen."""
    peak = a.mpf
    for _ in range(n):
        a.advance(DT_MODEL)
        b.advance(DT_MODEL)
        assert b.state_vector() == a.state_vector()
        peak = max(peak, a.mpf)
    return peak


def test_fresh_starts_at_published_initial_state():
    cycle = CellCycle.fresh()
    # (t, C2, CP, M, pM, Y, YP): initial values of BIOMD0000000005 (Tyson 1991)
    assert cycle.state_vector() == (0.0, 0.0, 0.75, 0.0, 0.25, 0.0, 0.0)
    assert cycle.t == 0.0
    assert cycle.mpf == 0.0
    assert cycle.cyclin == 0.25  # YT = Y + YP + M + pM
    assert cycle.synthesis_scale == 1.0
    assert set(cycle.params) == KINETIC_PARAMS
    assert cycle.params[SYNTHESIS] == 0.015  # k1aa, Tyson 1991 (BIOMD0000000005)


def test_oscillates():
    t, mpf = run_mpf(CellCycle.fresh(), 4 * PERIOD)
    assert len(peak_times(t, mpf)) >= 3


def test_copy_is_bitwise_identical():
    parent = CellCycle.fresh().copy(random.Random(5), 0.02)  # params off the published values
    parent.set_synthesis_scale(0.8)  # the copy must carry this, or the trajectories split
    advance_to_spike_upstroke(parent)
    child = parent.copy()
    assert child.state_vector() == parent.state_vector()
    assert child.params == parent.params
    peak = assert_lockstep(parent, child, 500)
    assert peak > MPF_THRESHOLD  # the 500 ticks crossed a whole MPF spike


def test_copy_does_not_share_state():
    parent = CellCycle.fresh()
    before = parent.state_vector()
    child = parent.copy()
    child.advance(DT_MODEL)
    child.set_synthesis_scale(0.3)
    assert parent.state_vector() == before
    assert parent.synthesis_scale == 1.0


def test_restore_is_bitwise_identical():
    # Every saved field differs from a fresh cell: time, species, spread params, scale.
    cycle = CellCycle.fresh().copy(random.Random(1), 0.05)
    cycle.set_synthesis_scale(0.9)  # still spiking: below ~0.7 MPF stays under threshold
    advance_to_spike_upstroke(cycle)
    saved = json.loads(json.dumps(cycle.to_dict(), allow_nan=False))
    restored = CellCycle.from_dict(saved)
    assert restored.state_vector() == cycle.state_vector()
    assert restored.params == cycle.params
    assert restored.synthesis_scale == 0.9
    peak = assert_lockstep(cycle, restored, 500)
    assert peak > MPF_THRESHOLD


def _drop_species(d: dict) -> None:
    del d["species"]["M"]


def _extra_species(d: dict) -> None:
    d["species"]["Z"] = 0.0


def _stale_param(d: dict) -> None:
    d["params"]["Reaction6_k1"] = d["params"].pop(SYNTHESIS)


def _drop_time(d: dict) -> None:
    del d["t"]


def _nan_time(d: dict) -> None:
    d["t"] = math.nan


def _inf_species(d: dict) -> None:
    d["species"]["M"] = math.inf


def _nan_param(d: dict) -> None:
    d["params"][SYNTHESIS] = math.nan


def _nan_scale(d: dict) -> None:
    d["synthesis_scale"] = math.nan


def _text_time(d: dict) -> None:
    d["t"] = "soon"


def _null_params(d: dict) -> None:
    d["params"] = None


@pytest.mark.parametrize(
    "corrupt",
    [
        _drop_species,
        _extra_species,
        _stale_param,
        _drop_time,
        _nan_time,
        _inf_species,
        _nan_param,
        _nan_scale,
        _text_time,
        _null_params,
    ],
)
def test_from_dict_rejects_stale_or_corrupt_state(corrupt):
    d = json.loads(json.dumps(CellCycle.fresh().to_dict()))
    corrupt(d)
    with pytest.raises(ValueError):
        CellCycle.from_dict(d)


def test_synthesis_zero_arrests():
    cycle = CellCycle.fresh()
    cycle.set_synthesis_scale(0.0)
    t, mpf = run_mpf(cycle, 4 * PERIOD)
    assert [p for p in peak_times(t, mpf) if p > PERIOD] == []


def test_scale_multiplies_published_synthesis_rate():
    # Reference: plain libroadrunner on the same model, k1aa set by hand to published × 0.5.
    rr = roadrunner.RoadRunner(roadrunner.RoadRunner.getParamPromotedSBML(MODEL_PATH.read_text()))
    rr[SYNTHESIS] = 0.015 * 0.5  # k1aa = 0.015, Tyson 1991 (BIOMD0000000005)
    cycle = CellCycle.fresh()
    cycle.set_synthesis_scale(0.5)
    t = 0.0
    for _ in range(ticks(PERIOD)):
        t = rr.oneStep(t, DT_MODEL)
        cycle.advance(DT_MODEL)
    species = tuple(float(v) for v in rr.model.getFloatingSpeciesAmounts())
    assert cycle.state_vector() == (t, *species)


@pytest.mark.parametrize(("given", "stored"), [(1.5, 1.0), (-0.2, 0.0), (0.25, 0.25)])
def test_set_synthesis_scale_clamps_to_unit_interval(given, stored):
    clamped, twin = CellCycle.fresh(), CellCycle.fresh()
    clamped.set_synthesis_scale(given)
    twin.set_synthesis_scale(stored)
    assert clamped.synthesis_scale == stored
    assert_lockstep(clamped, twin, ticks(PERIOD / 2))  # runs exactly as the clamped value


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_set_synthesis_scale_rejects_non_finite(bad):
    cycle = CellCycle.fresh()
    cycle.set_synthesis_scale(0.4)
    with pytest.raises(ValueError):
        cycle.set_synthesis_scale(bad)
    assert cycle.synthesis_scale == 0.4


@pytest.mark.parametrize("bad", [0.0, -DT_MODEL, math.nan, math.inf])
def test_advance_rejects_bad_step(bad):
    cycle = CellCycle.fresh()
    cycle.advance(DT_MODEL)
    before = cycle.state_vector()
    with pytest.raises(ValueError):
        cycle.advance(bad)
    assert cycle.state_vector() == before


def test_spread_deterministic():
    published = CellCycle.fresh().params
    parent = CellCycle.fresh().copy(random.Random(7), 0.02)  # params off the published values
    parent.set_synthesis_scale(0.6)
    for _ in range(100):
        parent.advance(DT_MODEL)
    a = parent.copy(random.Random(3), 0.02)
    b = parent.copy(random.Random(3), 0.02)
    assert a.params == b.params
    assert a.params != parent.params
    # Ruling R13: published value × (1 + U(-spread, spread)), drawn in sorted-id order;
    # the parent's own (already spread) values play no part.
    rng = random.Random(3)
    expected = {k: v * (1 + rng.uniform(-0.02, 0.02)) for k, v in sorted(published.items())}
    assert a.params == expected
    # Spread touches parameters only: state and scale carry over from the parent.
    assert a.state_vector() == parent.state_vector()
    assert a.synthesis_scale == 0.6


def test_spread_does_not_compound_over_generations():
    # Ruling R13 (SPEC §1.1): a lineage never walks away from Tyson 1991's parameters.
    published = CellCycle.fresh().params
    rng = random.Random(11)
    cell = CellCycle.fresh()
    for _ in range(50):
        cell = cell.copy(rng, 0.02)
        for pid, value in cell.params.items():
            if published[pid] == 0.0:
                assert value == 0.0
            else:
                assert abs(value / published[pid] - 1) <= 0.02, pid


def test_spread_params_drive_the_model():
    # Reference: plain libroadrunner with every rate constant set by hand to the child's values.
    child = CellCycle.fresh().copy(random.Random(3), 0.02)
    rr = roadrunner.RoadRunner(roadrunner.RoadRunner.getParamPromotedSBML(MODEL_PATH.read_text()))
    for pid, value in child.params.items():
        rr[pid] = value
    t = 0.0
    for _ in range(ticks(PERIOD)):
        t = rr.oneStep(t, DT_MODEL)
        child.advance(DT_MODEL)
    species = tuple(float(v) for v in rr.model.getFloatingSpeciesAmounts())
    assert child.state_vector() == (t, *species)


@pytest.mark.parametrize(
    ("rng", "spread"),
    [
        (None, 0.02),
        (random.Random(0), -0.01),
        (random.Random(0), 1.0),
        (random.Random(0), math.nan),
        (random.Random(0), math.inf),
    ],
)
def test_copy_rejects_bad_spread(rng, spread):
    with pytest.raises(ValueError):
        CellCycle.fresh().copy(rng, spread)


def test_instances_independent():
    a, b, control = CellCycle.fresh(), CellCycle.fresh(), CellCycle.fresh()
    for cycle in (a, b, control):
        for _ in range(ticks(PERIOD / 2)):
            cycle.advance(DT_MODEL)
    b.set_synthesis_scale(0.4)
    for _ in range(ticks(PERIOD)):
        a.advance(DT_MODEL)
        b.advance(DT_MODEL)
        control.advance(DT_MODEL)
    assert a.state_vector() == control.state_vector()
    assert b.state_vector() != a.state_vector()


def test_runtime_has_no_tellurium():
    # Exercise every runtime path, not only the import: the template is built lazily.
    code = (
        "import json, random, sys\n"
        "from alife.lifeforms.cell.cycle import CellCycle\n"
        "c = CellCycle.fresh()\n"
        "c.set_synthesis_scale(0.5)\n"
        "c.advance(0.01)\n"
        "CellCycle.from_dict(c.copy(random.Random(0), 0.02).to_dict()).advance(0.01)\n"
        "names = sorted(m for m in sys.modules if m.startswith(('tellurium', 'antimony')))\n"
        "print(json.dumps(names))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []
