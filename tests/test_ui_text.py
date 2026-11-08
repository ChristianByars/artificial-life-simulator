"""HUD text and UI copy (F5, M2). Headless: ``alife.ui_text`` never imports Ursina."""

from __future__ import annotations

import math

import pytest

from alife import ui_text
from alife.session import SPEEDS
from alife.ui_text import (
    FpsMeter,
    format_fps,
    format_sim_time,
    format_speed,
    format_stat,
    hud_lines,
)


def lines(**overrides) -> list[str]:
    args = {
        "name": "Random walkers (dev)",
        "tick": 5400,
        "sim_time": 90.0,
        "speed": 2.0,
        "fps": 59.6,
        "stats": {"walkers": 30},
        "paused": False,
        "selected": None,
    }
    args.update(overrides)
    return hud_lines(**args)


def test_format_speed():
    assert [format_speed(s) for s in SPEEDS] == ["0.25×", "0.5×", "1×", "2×", "4×", "8×"]


def test_format_sim_time():
    assert format_sim_time(90.4) == "01:30"
    assert format_sim_time(3600) == "60:00"
    assert format_sim_time(0) == "00:00"
    assert format_sim_time(59.999) == "00:59"  # floors: a second shows only once it has passed
    assert format_sim_time(7200) == "120:00"  # minutes do not wrap into hours


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0])
def test_format_sim_time_rejects_bad_values(bad):
    assert format_sim_time(bad) == "--:--"


def test_format_fps():
    assert format_fps(59.6) == "60"
    assert format_fps(None) == "--"
    for bad in (0.0, -5.0, math.nan, math.inf):
        assert format_fps(bad) == "--"


def test_format_stat():
    assert format_stat(30) == "30"
    assert format_stat(30.0) == "30"
    assert format_stat(0.73456) == "0.735"
    assert format_stat(12.345) == "12.3"
    assert format_stat(1234.6) == "1235"
    assert format_stat("G1") == "G1"
    assert format_stat(math.nan) == "nan"


def test_hud_first_line_is_name():
    assert lines()[0] == "Random walkers (dev)"


def test_hud_paused_banner():
    assert "PAUSED" in lines(paused=True)
    assert "PAUSED" not in lines(paused=False)
    assert ui_text.PAUSED == "PAUSED"


def test_hud_shows_clock_speed_fps_and_stats():
    text = "\n".join(lines())
    assert "Tick 5400" in text
    assert "01:30" in text
    assert "2×" in text
    assert "FPS 60" in text
    assert "walkers: 30" in lines()


def test_hud_selected():
    assert "Selected #12" in lines(selected=12)
    assert "Selected #0" in lines(selected=0)  # id 0 is a real id, not "nothing"
    assert not any(line.startswith("Selected") for line in lines(selected=None))


def test_hud_unknown_fps():
    assert "FPS --" in "\n".join(lines(fps=None))


def test_copy_constants():
    assert ui_text.SAVED == "Saved"
    assert ui_text.SAVE_FAILED == "Save failed: {reason}"
    assert ui_text.SAVE_FAILED.format(reason="disk full") == "Save failed: disk full"
    assert ui_text.CONFIRM_RESET == "Reset to the start? Same seed, same settings."
    assert ui_text.CONFIRM_NEW == "Start over with a new seed?"


def test_fps_meter_averages_over_its_window():
    meter = FpsMeter(window=0.5)
    assert meter.fps is None  # nothing measured yet
    for _ in range(29):
        meter.add(1 / 60)
    assert meter.fps is None  # 29 frames = 0.483 s: the window has not filled
    meter.add(1 / 60)
    assert meter.fps == pytest.approx(60)
    for _ in range(15):
        meter.add(1 / 30)
    assert meter.fps == pytest.approx(30)  # a new window replaces the old value


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.1])
def test_fps_meter_skips_bad_frame_times(bad):
    meter = FpsMeter(window=0.5)
    meter.add(bad)
    for _ in range(30):
        meter.add(1 / 60)
    assert meter.fps == pytest.approx(60)
