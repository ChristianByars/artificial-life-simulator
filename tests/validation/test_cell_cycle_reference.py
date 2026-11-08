"""SPEC §6.1 test_cycle_reference: one isolated cell's MPF oscillation period matches a
standalone Tellurium run of BIOMD0000000005 within 1%, with time_scale applied."""

from __future__ import annotations

import math

import pytest

from alife.lifeforms.cell.cycle import DEFAULT_TIME_SCALE, MODEL_PATH, CellCycle
from alife.session import TICK_DT
from tests.validation.helpers import mean_period, peak_times

P = 35.60  # model minutes: M0 reference period (SPIKE.md C2); sets the run length only
N_PERIODS = 12


@pytest.fixture(scope="module")
def ref_period() -> float:
    import tellurium as te  # dev-only, for the reference run (plan M9)

    result = te.loadSBMLModel(str(MODEL_PATH)).simulate(0, N_PERIODS * P, 24000)
    return mean_period(result["time"], result["[M]"])


def test_cycle_reference(ref_period):
    cycle = CellCycle.fresh()
    dt_model = TICK_DT * DEFAULT_TIME_SCALE
    screen_t, mpf = [0.0], [cycle.mpf]
    for tick in range(1, round(N_PERIODS * P / dt_model) + 1):
        cycle.advance(dt_model)
        screen_t.append(tick * TICK_DT)
        mpf.append(cycle.mpf)
    ours_screen_period = mean_period(screen_t, mpf)  # screen seconds
    assert abs(ours_screen_period * DEFAULT_TIME_SCALE - ref_period) / ref_period < 0.01, (
        f"ours {ours_screen_period * DEFAULT_TIME_SCALE} vs ref {ref_period} model minutes"
    )


def test_default_time_scale_gives_a_minute_per_cycle(ref_period):
    # SPIKE.md D3 (our assumption): one cycle takes about a minute on screen (SPEC §6.1).
    assert ref_period / DEFAULT_TIME_SCALE == pytest.approx(60.0, rel=0.01)


# The helpers measure every period above, so they get their own known-answer checks.

SINE_T = 7.0  # period of the test sine, arbitrary units


def sine(n: int, dt: float, amplitude: float = 1.0) -> tuple[list[float], list[float]]:
    """Samples offset from the peaks, so interpolation has work to do."""
    t = [0.013 + i * dt for i in range(n)]
    return t, [amplitude * math.sin(2 * math.pi * x / SINE_T) for x in t]


def test_peak_times_interpolates_between_samples():
    t, y = sine(1000, 0.05)  # t in [0.013, 49.96]: peaks at 1.75 + 7k, k = 0..6
    expected = [SINE_T / 4 + k * SINE_T for k in range(7)]
    assert peak_times(t, y) == pytest.approx(expected, abs=1e-4)
    nearest_sample = min(t, key=lambda x: abs(x - expected[0]))
    assert abs(nearest_sample - expected[0]) > 1e-3  # so the 1e-4 match is interpolation


def test_peak_times_ignores_maxima_below_min_height():
    t, y = sine(1000, 0.05, amplitude=0.05)  # under MPF_THRESHOLD, the default
    assert peak_times(t, y) == []
    assert len(peak_times(t, y, min_height=0.0)) == 7


def test_mean_period_of_sine_after_skip():
    t, y = sine(1000, 0.05)
    assert mean_period(t, y) == pytest.approx(SINE_T, abs=1e-4)
    assert mean_period(t, y, skip=5) == pytest.approx(SINE_T, abs=1e-4)


def test_mean_period_needs_two_peaks_after_skip():
    t, y = sine(1000, 0.05)
    with pytest.raises(ValueError):
        mean_period(t, y, skip=6)  # 7 peaks, 6 skipped: one left, no interval
