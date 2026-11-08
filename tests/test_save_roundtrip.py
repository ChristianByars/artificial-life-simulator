"""F6: build a save from a session, restore it exactly, and the app's save wiring (SPEC §4, M3).

Headless tests first. Tests marked ``gui`` open a short, self-closing window:
``uv run pytest -m gui``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import platformdirs
import pytest

import alife
from alife import orbit as om
from alife.lifeforms import registry
from alife.lifeforms.demo import DemoLifeform
from alife.save import (
    SAVE_KEYS,
    SCHEMA_VERSION,
    SaveError,
    SaveStore,
    build_save,
    restore,
    store_for_run,
)
from alife.session import TICK_DT, Session

NOW = datetime(2026, 11, 7, 16, 0, 0, tzinfo=timezone(timedelta(hours=-8)))
CUSTOM_POSE = om.to_pose(om.OrbitState(target=(1.5, 0.0, -2.25), yaw=30.0, pitch=60.0, distance=20))


def run(session: Session, ticks: int) -> None:
    for _ in range(ticks):
        assert session.advance(TICK_DT) == 1


def demo_save(ticks: int = 30, **overrides: object) -> dict:
    session = Session(DemoLifeform, 7)
    run(session, ticks)
    data = json.loads(json.dumps(build_save(session, CUSTOM_POSE, NOW), allow_nan=False))
    data.update(overrides)
    return data


# --- build_save -------------------------------------------------------------------------------


def test_build_save_has_every_key() -> None:
    session = Session(DemoLifeform, 7)
    run(session, 5)
    data = build_save(session, CUSTOM_POSE, NOW)
    assert tuple(data) == SAVE_KEYS
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["app_version"] == alife.__version__
    assert data["lifeform_id"] == "demo"
    assert data["seed"] == 7
    assert data["tick"] == 5
    assert data["speed"] == 1.0
    assert data["camera"] == CUSTOM_POSE
    assert data["state"] == session.lifeform.get_state()
    json.dumps(data, allow_nan=False)  # what SaveStore.write does


def test_saved_at_keeps_an_aware_offset() -> None:
    session = Session(DemoLifeform, 7)
    assert build_save(session, CUSTOM_POSE, NOW)["saved_at"] == "2026-11-07T16:00:00-08:00"


def test_saved_at_gets_the_local_offset_when_naive() -> None:
    session = Session(DemoLifeform, 7)
    naive = datetime(2026, 11, 7, 16, 0, 0, 123456)
    saved_at = build_save(session, CUSTOM_POSE, naive)["saved_at"]
    assert re.fullmatch(r"2026-11-07T16:00:00[+-]\d\d:\d\d", saved_at), saved_at
    assert saved_at == naive.astimezone().isoformat(timespec="seconds")


def test_build_save_copies_the_pose() -> None:
    pose = copy.deepcopy(CUSTOM_POSE)
    data = build_save(Session(DemoLifeform, 7), pose, NOW)
    pose["position"][0] = 99.0
    assert data["camera"] == CUSTOM_POSE


# --- round trip -------------------------------------------------------------------------------


def test_roundtrip_restores_everything(data_dir: Path) -> None:
    original = Session(DemoLifeform, 4242)
    run(original, 300)
    original.speed = 2.0
    store = SaveStore()
    store.write("demo", build_save(original, CUSTOM_POSE, NOW))
    result = store.read("demo")
    assert result.status == "ok"

    restored, pose = restore(result.data, registry())

    assert restored.lifeform_cls is DemoLifeform
    assert restored.lifeform.get_state() == original.lifeform.get_state()
    assert restored.tick == original.tick == 300
    assert restored.speed == 2.0
    assert restored.seed == 4242
    assert restored.params == original.params
    assert pose == CUSTOM_POSE
    assert restored.rng.random() == original.rng.random()  # the next draw matches


def test_selection_is_not_saved() -> None:  # I3: selection is UI state, never saved
    session = Session(DemoLifeform, 7)
    session.lifeform.on_pick(3)
    assert "selected" not in build_save(session, CUSTOM_POSE, NOW)["state"]


def test_restore_ignores_a_saved_selection() -> None:  # I3: saves from before the fix
    data = demo_save()
    data["state"]["selected"] = 3
    restored, _ = restore(data, registry())
    assert restored.lifeform.sim.selected is None
    assert "selected" not in restored.lifeform.get_state()


def test_restore_continues_on_the_lifeforms_rng() -> None:
    restored, _ = restore(demo_save(), registry())
    assert restored.lifeform.sim.rng is restored.rng


# --- restore turns every failure into SaveError (SPEC §4, Review Focus 4) ----------------------


def _without(key: str) -> dict:
    data = demo_save()
    del data[key]
    return data


BAD_SAVES = {
    "not_a_dict": lambda: [1, 2, 3],
    "missing_state": lambda: _without("state"),
    "missing_camera": lambda: _without("camera"),
    "unknown_schema": lambda: demo_save(schema_version=2),
    "unknown_lifeform": lambda: demo_save(lifeform_id="nope"),
    "lifeform_id_not_str": lambda: demo_save(lifeform_id=["demo"]),
    "tick_negative": lambda: demo_save(tick=-1),
    "tick_float": lambda: demo_save(tick=30.5),
    "tick_bool": lambda: demo_save(tick=True),
    "tick_str": lambda: demo_save(tick="30"),
    "tick_2_53": lambda: demo_save(tick=2**53),  # M3: the first tick past the bound
    "tick_10_400": lambda: demo_save(tick=10**400),  # M3: sim_time would overflow a float
    "seed_str": lambda: demo_save(seed="7"),
    "seed_float": lambda: demo_save(seed=7.0),
    "speed_str": lambda: demo_save(speed="fast"),
    "speed_bool": lambda: demo_save(speed=True),
    "params_list": lambda: demo_save(params=[1]),
    "params_unknown_key": lambda: demo_save(params={"bogus": 1.0}),
    "params_value_str": lambda: demo_save(params={"count": "many"}),
    "state_rejected": lambda: demo_save(state={}),  # set_state raises KeyError
    "state_wrong_shape": lambda: demo_save(state={"next_id": 1, "selected": None, "walkers": [5]}),
    "state_bad_number": lambda: demo_save(state={"next_id": "x", "selected": None, "walkers": []}),
    "state_not_a_dict": lambda: demo_save(state="state"),
    "rng_state_short": lambda: demo_save(rng_state=[3, [1, 2, 3], None]),
    "rng_state_not_list": lambda: demo_save(rng_state="abc"),
    "rng_state_overflow": lambda: demo_save(rng_state=[3, [2**70] * 624 + [624], None]),
    "camera_not_dict": lambda: demo_save(camera=[0, 1, 2]),
    "camera_short_position": lambda: demo_save(camera={"position": [0, 1], "rotation": [45, 0, 0]}),
    "camera_text": lambda: demo_save(camera={"position": [0, "a", 0], "rotation": [45, 0, 0]}),
    "camera_roll_text": lambda: demo_save(
        camera={"position": [0, 32, -32], "rotation": [45, 0, "a"]}
    ),
    "camera_long_rotation": lambda: demo_save(
        camera={"position": [0, 32, -32], "rotation": [45, 0, 0, 0]}
    ),
}


@pytest.mark.parametrize("case", sorted(BAD_SAVES))
def test_restore_raises_save_error(case: str) -> None:
    with pytest.raises(SaveError):
        restore(BAD_SAVES[case](), registry())


def test_restore_reports_why() -> None:
    with pytest.raises(SaveError, match="tick"):
        restore(demo_save(tick=-1), registry())
    with pytest.raises(SaveError, match="nope"):
        restore(demo_save(lifeform_id="nope"), registry())
    with pytest.raises(SaveError, match="KeyError"):
        restore(demo_save(state={}), registry())


def test_restore_accepts_the_largest_tick() -> None:  # M3: the bound is 2**53, exclusive
    restored, _ = restore(demo_save(tick=2**53 - 1), registry())
    assert restored.tick == 2**53 - 1
    assert restored.sim_time == (2**53 - 1) * TICK_DT  # finite: the HUD can show it


def test_restore_accepts_a_two_value_rotation() -> None:
    data = demo_save(camera={"position": [0, 32, -32], "rotation": [45, 0]})
    _, pose = restore(data, registry())
    assert pose == {"position": [0.0, 32.0, -32.0], "rotation": [45.0, 0.0]}


def test_restore_leaves_data_untouched() -> None:
    data = demo_save()
    before = copy.deepcopy(data)
    restore(data, registry())
    assert data == before


def test_save_error_reason_is_short() -> None:
    try:
        try:
            raise PermissionError(13, "Permission denied", "/x/demo.json")
        except OSError as exc:
            raise SaveError("could not write save 'demo': long text") from exc
    except SaveError as err:
        assert err.reason == "Permission denied"
    try:
        try:
            json.dumps(float("nan"), allow_nan=False)
        except ValueError as exc:
            raise SaveError("could not write save 'demo'") from exc
    except SaveError as err:
        assert err.reason == "ValueError"
    assert SaveError("plain").reason == "plain"


# --- R21: --smoke / --screenshot never write the real save folder -------------------------------


@pytest.fixture
def fake_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """No ALIFE_DATA_DIR; platformdirs and the temp dir both point into tmp_path."""
    platform, temp = tmp_path / "platform", tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.delenv("ALIFE_DATA_DIR", raising=False)
    monkeypatch.setattr(platformdirs, "user_data_dir", lambda *a, **k: str(platform))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return platform, temp


def test_test_runs_get_a_throwaway_store(fake_platform: tuple[Path, Path]) -> None:
    platform, temp = fake_platform
    store = store_for_run(test_run=True)
    assert store.root.name == "saves"
    assert store.root.parent.parent == temp
    store.write("demo", demo_save())
    assert store.read("demo").status == "ok"
    assert not platform.exists()
    assert store_for_run(test_run=True).root != store.root  # a fresh folder per run


def test_normal_runs_use_the_platform_dir(fake_platform: tuple[Path, Path]) -> None:
    platform, _ = fake_platform
    assert store_for_run(test_run=False).root == platform / "saves"


def test_test_runs_keep_alife_data_dir(data_dir: Path) -> None:
    assert store_for_run(test_run=True).root == data_dir / "saves"


# --- the app (GUI: each test opens a short, self-closing window) -------------------------------

GUI_TIMEOUT = 60  # seconds; a hung run fails instead of blocking the suite
GLSL_ERROR = "error occurred while compiling GLSL"
AUTOSAVE_FAILED = "alife: autosave failed"


def run_app(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alife", *args],
        capture_output=True,
        text=True,
        timeout=GUI_TIMEOUT,
        env=env,
    )


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=GUI_TIMEOUT,
    )


def checks_of(proc: subprocess.CompletedProcess[str]) -> dict[str, str]:
    return {
        parts[1]: parts[2]
        for parts in (line.split(maxsplit=3) for line in proc.stdout.splitlines())
        if parts and parts[0] == "CHECK"
    }


def assert_checks(proc: subprocess.CompletedProcess[str], expected: set[str]) -> None:
    both = proc.stdout + proc.stderr
    assert proc.returncode == 0, both[-4000:]
    assert GLSL_ERROR not in both, both[-4000:]
    checks = checks_of(proc)
    assert checks.keys() == expected, both[-4000:]
    failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("CHECK") and "FAIL" in ln]
    assert not failed, failed


def read_ok(data_dir: Path, lifeform_id: str = "demo") -> dict:
    result = SaveStore(data_dir / "saves").read(lifeform_id)
    assert result.status == "ok", result
    restore(result.data, registry())  # and it restores
    return result.data


@pytest.mark.gui
def test_smoke_autosaves(data_dir: Path) -> None:
    proc = run_app("--lifeform", "demo", "--seed", "1", "--smoke", "2")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    data = read_ok(data_dir)
    assert data["seed"] == 1
    assert data["tick"] > 0
    assert AUTOSAVE_FAILED not in proc.stderr


@pytest.mark.gui
def test_smoke_menu_writes_no_save(data_dir: Path) -> None:
    proc = run_app("--smoke", "1")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    assert not (data_dir / "saves" / "demo.json").exists()


@pytest.mark.gui
def test_failed_autosave_does_not_block_quit(data_dir: Path) -> None:
    (data_dir / "saves").write_text("a file where the saves folder should be")
    proc = run_app("--lifeform", "demo", "--smoke", "1")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    lines = [ln for ln in proc.stderr.splitlines() if ln.startswith(AUTOSAVE_FAILED)]
    assert len(lines) == 1, proc.stderr[-3000:]


@pytest.mark.gui
@pytest.mark.parametrize("flag", ["--smoke", "--screenshot"])
def test_test_runs_never_touch_the_platform_dir(flag: str, tmp_path: Path) -> None:  # R21
    # No ALIFE_DATA_DIR. HOME points platformdirs into tmp_path, so the user's real folder is
    # never the target; TMPDIR catches the throwaway folder.
    home, temp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    temp.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "ALIFE_DATA_DIR"}
    env.update(HOME=str(home), TMPDIR=str(temp))
    platform_saves = home / "Library" / "Application Support" / "alife" / "saves"
    where = subprocess.run(
        [sys.executable, "-c", "from alife.save import saves_dir; print(saves_dir())"],
        capture_output=True,
        text=True,
        env=env,
        timeout=GUI_TIMEOUT,
    )
    assert Path(where.stdout.strip()) == platform_saves  # checked before any window opens

    value = "1" if flag == "--smoke" else str(tmp_path / "shot.png")
    proc = run_app("--lifeform", "demo", "--seed", "2", flag, value, env=env)

    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    assert not platform_saves.parent.exists()
    assert list(home.rglob("demo.json")) == []
    autosaves = list(temp.glob("alife-*/saves/demo.json"))
    assert len(autosaves) == 1, list(temp.rglob("*"))
    assert read_ok(autosaves[0].parent.parent)["seed"] == 2  # autosave still ran
    assert "saves go to" in proc.stderr


# Each quit path the user has: the red close button (NSWindow performClose:, exactly what the
# button sends), Cmd+Q (the app menu's Quit item sends NSApp terminate:, which Panda's
# applicationShouldTerminate turns into closing the windows), and the watchdog (the window
# closes but Panda sends no window-event). Each must autosave.
QUIT_PATH = textwrap.dedent(
    """
    import ctypes
    import sys

    from alife.__main__ import parse_args
    from alife.app import App

    mode = sys.argv[1]
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]


    def send(obj, selector, *args):
        msg = objc.objc_msgSend
        msg.restype = ctypes.c_void_p
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_void_p] * len(args)
        return msg(obj, objc.sel_registerName(selector.encode()), *args)


    def ns_app():
        return send(objc.objc_getClass(b"NSApplication"), "sharedApplication")


    app = App(parse_args(["--lifeform", "demo", "--seed", "3"]))
    frames = [0]


    def step(task):
        frames[0] += 1
        if frames[0] < 30:
            return task.cont
        print("TICK BEFORE QUIT", app.screen.session.tick, flush=True)
        if mode == "close_button":
            windows = send(ns_app(), "windows")
            count = send(windows, "count")
            print("WINDOWS", count, flush=True)
            for i in range(count):
                win = send(windows, "objectAtIndex:", ctypes.c_void_p(i))
                send(win, "performClose:", None)
        elif mode == "cmd_q":
            send(ns_app(), "terminate:", None)
        elif mode == "watchdog":
            app.ursina.ignore("window-event")  # Panda drops the event; the watchdog notices
            app.quit()
        return task.done


    app.ursina.taskMgr.add(step, "quit-path")
    sys.exit(app.run())
    """
)


@pytest.mark.gui
@pytest.mark.parametrize("mode", ["close_button", "cmd_q", "watchdog"])
def test_quit_path_autosaves(mode: str, data_dir: Path) -> None:
    proc = run_script(QUIT_PATH, mode)
    both = proc.stdout + proc.stderr
    assert proc.returncode == 0, both[-3000:]
    before = [ln for ln in proc.stdout.splitlines() if ln.startswith("TICK BEFORE QUIT")]
    assert before, both[-3000:]
    data = read_ok(data_dir)
    assert data["seed"] == 3
    assert data["tick"] >= int(before[0].split()[-1]) > 0


def write_demo_save(data_dir: Path, **overrides: object) -> dict:
    data = demo_save(ticks=300, **overrides)
    SaveStore(data_dir / "saves").write("demo", data)
    return data


CONTINUE_FLAG = textwrap.dedent(
    """
    import json
    import sys

    from alife.__main__ import parse_args
    from alife.app import App
    from alife.screens.simulation import SimulationScreen

    saved = json.loads(sys.argv[1])
    app = App(parse_args(["--continue", "demo"]))


    def check(name, ok, detail=""):
        print("CHECK", name, "ok" if ok else "FAIL", detail, flush=True)


    def first_frame(task):
        s = app.screen
        check("continue_starts_sim", isinstance(s, SimulationScreen))
        check("seed_restored", s.session.seed == saved["seed"])
        check("speed_restored", s.session.speed == saved["speed"])
        check("tick_restored", s.session.tick >= saved["tick"], s.session.tick)
        pose = s.orbit_camera.get_pose()
        close = all(
            abs(a - b) < 1e-9
            for key in ("position", "rotation")
            for a, b in zip(pose[key], saved["camera"][key])
        )
        check("camera_restored", close, pose)
        app.quit()
        return task.done


    app.ursina.taskMgr.doMethodLater(0.3, first_frame, "continue-check")
    sys.exit(app.run())
    """
)

CONTINUE_FLAG_CHECKS = {
    "continue_starts_sim",
    "seed_restored",
    "speed_restored",
    "tick_restored",
    "camera_restored",
}


@pytest.mark.gui
def test_continue_flag_starts_from_save(data_dir: Path) -> None:
    saved = write_demo_save(data_dir, speed=4.0)
    proc = run_script(CONTINUE_FLAG, json.dumps(saved))
    assert_checks(proc, CONTINUE_FLAG_CHECKS)


@pytest.mark.gui
def test_continue_flag_with_bad_save_shows_menu(data_dir: Path) -> None:
    path = data_dir / "saves" / "demo.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": 1, "trunc')
    proc = run_app("--continue", "demo", "--smoke", "1")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    assert "Save file is damaged." in proc.stderr
    assert path.read_text() == '{"schema_version": 1, "trunc'  # the menu never rewrites it


# One window: Cmd+S saves with a toast; Esc autosaves and the card offers Continue; New on that card
# asks first (R20) and Cancel keeps the save; Continue restores tick, speed, camera and state; a
# stale save shows its error and New, which starts at once and leaves the bad file until the next
# save; confirming New on a good save starts fresh.
SCENARIO = textwrap.dedent(
    """
    import sys

    from panda3d.core import Filename, KeyboardButton as K, MouseButton as M
    from ursina import Entity, Text, camera, destroy, mouse, window

    from alife import orbit as om
    from alife.__main__ import parse_args
    from alife.app import App
    from alife.lifeforms import registry
    from alife.lifeforms.demo import DemoLifeform
    from alife.lifeforms.demo.view import SELECTED_COLOR
    from alife.save import MSG_RESTORE_FAILED, SaveStore, restore
    from alife.screens.menu import BUTTON_WIDTH, SUMMARY_SCALE, MenuScreen
    from alife.screens.simulation import SimulationScreen
    from alife.ui_text import CONFIRM_NEW, SAVED

    store = SaveStore()
    app = App(parse_args(["--lifeform", "demo", "--seed", "5"]))
    A = K.ascii_key
    POSE = om.to_pose(om.OrbitState(target=(2.0, 0.0, -1.0), yaw=-40.0, pitch=50.0, distance=25))
    pick_calls = []  # every on_pick call, as (life form, id) (I3)
    _on_pick = DemoLifeform.on_pick


    def spy_on_pick(self, entity_id):
        pick_calls.append((self, entity_id))
        _on_pick(self, entity_id)


    DemoLifeform.on_pick = spy_on_pick


    def check(name, ok, detail=""):
        print("CHECK", name, "ok" if ok else "FAIL", detail, flush=True)


    def dev():
        return app.ursina.win.get_input_device(0)


    def press(button, raw=False):
        if raw:
            dev().raw_button_down(button)
        dev().button_down(button)
        if raw:
            dev().raw_button_up(button)
        dev().button_up(button)


    def cmd_s():
        dev().button_down(K.meta())
        dev().button_down(K.lmeta())
        press(A("s"), raw=True)
        dev().button_up(K.lmeta())
        dev().button_up(K.meta())


    def point_at(entity):
        p = entity.get_pos(camera.ui)
        w, h = window.size
        dev().set_pointer_in_window((p.x / window.aspect_ratio + 0.5) * w, (0.5 - p.y) * h)


    def click():
        dev().button_down(M.one())
        dev().button_up(M.one())


    class KeySpy(Entity):
        def __init__(self):
            super().__init__()
            self.seen = []

        def input(self, key):
            self.seen.append(key)


    def read():
        return store.read("demo")


    def close(a, b):
        return all(abs(x - y) < 1e-9 for x, y in zip(a, b)) and len(a) == len(b)


    def pose_close(a, b):
        return close(a["position"], b["position"]) and close(a["rotation"], b["rotation"])


    def texts(entity):
        return [e.text for e in entity.get_descendants() if isinstance(e, Text) and e.enabled]


    def orange(e):
        return all(abs(a - b) < 1e-3 for a, b in zip(e.color, SELECTED_COLOR))


    def scenario():
        yield 20
        s = app.screen
        check("no_save_at_start", read().status == "missing")
        s.session.speed_up()  # 2x
        s.orbit_camera.set_pose(POSE)
        yield 5
        cmd_s()
        yield 3
        r = read()
        check("cmd_s_saves", r.status == "ok", r.status)
        check("cmd_s_toast", s.toast.enabled and s.toast.message == SAVED, s.toast.message)
        check("save_has_speed", r.status == "ok" and r.data["speed"] == 2.0)
        check("save_has_pose", r.status == "ok" and pose_close(r.data["camera"], POSE))
        first_tick = r.data["tick"] if r.status == "ok" else -1
        yield 10

        # A pick, as a click makes it: drawn orange, but it is UI state and not saved (I3).
        views = s.session.lifeform.view.entities
        wid = sorted(views)[3]
        s.picked = views[wid]
        s.session.lifeform.on_pick(wid)
        yield 2
        check("pick_drawn_orange", orange(views[wid]))

        # Esc autosaves, then shows the menu with Continue.
        live = s.session
        press(K.escape())
        yield 3
        r = read()
        check("esc_autosaves", r.status == "ok" and r.data["tick"] > first_tick)
        saved = r.data
        check("esc_saved_live_state", saved["state"] == live.lifeform.get_state())
        check("esc_save_has_no_pick", "selected" not in saved["state"], sorted(saved["state"]))
        # Saves written before I3 kept the pick in the state; Continue must ignore it.
        store.write("demo", dict(saved, state=dict(saved["state"], selected=wid)))
        m = app.screen
        check("esc_shows_menu", isinstance(m, MenuScreen))
        card = m.cards["demo"]
        check("card_has_continue", card.continue_button is not None)
        check("card_no_error", card.error_text is None)
        check("card_new_button", card.start_button.text == "New", card.start_button.text)
        summary_right = card.summary_text.x + card.summary_text.width * SUMMARY_SCALE
        buttons_left = min(card.start_button.x, card.continue_button.x) - BUTTON_WIDTH / 2
        check("summary_clears_buttons", summary_right < buttons_left, (summary_right, buttons_left))

        # R20: New on a card whose save restores asks first; meanwhile nothing reaches the cards.
        good_bytes = store.path("demo").read_bytes()
        point_at(card.start_button)
        yield 3
        check("hover_new", mouse.hovered_entity is card.start_button, mouse.hovered_entity)
        click()
        yield 2
        check("new_asks_first", app.screen is m and m.dialog is not None)
        check("new_dialog_text", m.dialog is not None and m.dialog.message == CONFIRM_NEW)
        spy = KeySpy()
        point_at(card.continue_button)
        yield 3
        hovered = mouse.hovered_entity
        check("dialog_blocks_hover", hovered is not card.continue_button, hovered)
        click()
        yield 2
        card.continue_button.on_click()  # even a direct call does nothing while the dialog is up
        card.start_button.on_click()
        yield 2
        press(K.space())
        press(A("n"), raw=True)
        yield 2
        check("dialog_blocks_cards", app.screen is m and m.dialog is not None)
        check("dialog_eats_keys", spy.seen == [], spy.seen)
        destroy(spy)
        press(K.escape())
        yield 2
        check("cancel_stays_on_menu", app.screen is m and m.dialog is None)
        check("cancel_keeps_save", store.path("demo").read_bytes() == good_bytes)
        point_at(card.continue_button)
        yield 3
        hovered = mouse.hovered_entity
        check("cards_clickable_again", hovered is card.continue_button, hovered)

        # Continue restores exactly what was saved.
        card.continue_button.on_click()
        s = app.screen
        check("continue_starts_sim", isinstance(s, SimulationScreen))
        check("continue_tick", s.session.tick == saved["tick"], (s.session.tick, saved["tick"]))
        check("continue_speed", s.session.speed == 2.0)
        check("continue_state", s.session.lifeform.get_state() == saved["state"])
        check("continue_pose", pose_close(s.orbit_camera.get_pose(), POSE))
        check("continue_view_built", len(s.session.lifeform.view.entities) == 30)
        yield 5
        lf = s.session.lifeform
        check("continue_clears_pick", (lf, None) in pick_calls)
        hud_lines = [t.text for t in s.hud.texts]
        nothing = lf.sim.selected is None and s.picked is None
        nothing = nothing and not any(orange(e) for e in lf.view.entities.values())
        nothing = nothing and not any(t.startswith("Selected") for t in hud_lines)
        check("continue_nothing_selected", nothing, (lf.sim.selected, hud_lines))

        # A stale save: valid JSON and schema, but set_state rejects its state.
        press(K.escape())
        yield 3
        stale = dict(read().data, state={"walkers": "gone"})
        store.write("demo", stale)
        stale_bytes = store.path("demo").read_bytes()
        app.show_menu()
        yield 3
        card = app.screen.cards["demo"]
        check("stale_no_continue", card.continue_button is None)
        check("stale_error_shown", MSG_RESTORE_FAILED in texts(card), texts(card))
        check("stale_new_button", card.start_button.text == "New", card.start_button.text)
        app.ursina.win.save_screenshot(Filename.from_os_specific(sys.argv[1]))
        card.start_button.on_click()
        yield 3
        s = app.screen
        check("new_starts_fresh", isinstance(s, SimulationScreen) and s.session.tick < 10)
        check("new_keeps_bad_file", store.path("demo").read_bytes() == stale_bytes)
        cmd_s()
        yield 3
        r = read()
        check("next_save_replaces", r.status == "ok" and r.data["state"] != stale["state"])
        try:
            restore(r.data, registry())
            check("next_save_restores", True)
        except Exception as exc:
            check("next_save_restores", False, repr(exc))

        # R20: confirming New on a good save starts fresh; the save stays until the next save.
        press(K.escape())
        yield 3
        m = app.screen
        good_bytes = store.path("demo").read_bytes()
        m.cards["demo"].start_button.on_click()
        yield 2
        check("good_save_new_asks", app.screen is m and m.dialog is not None)
        press(K.enter())
        yield 3
        s = app.screen
        check("enter_starts_fresh", isinstance(s, SimulationScreen) and s.session.tick < 10)
        check("confirm_keeps_save_until_next", store.path("demo").read_bytes() == good_bytes)
        print("TICK BEFORE QUIT", s.session.tick, flush=True)
        app.quit()


    steps = scenario()
    wait = [0]


    def drive(task):
        if wait[0] > 0:
            wait[0] -= 1
            return task.cont
        try:
            wait[0] = next(steps)
        except StopIteration:
            return task.done
        return task.cont


    app.ursina.taskMgr.add(drive, "scenario")
    sys.exit(app.run())
    """
)

SCENARIO_CHECKS = {
    "no_save_at_start",
    "cmd_s_saves",
    "cmd_s_toast",
    "save_has_speed",
    "save_has_pose",
    "pick_drawn_orange",
    "esc_autosaves",
    "esc_saved_live_state",
    "esc_save_has_no_pick",
    "esc_shows_menu",
    "card_has_continue",
    "card_no_error",
    "card_new_button",
    "summary_clears_buttons",
    "hover_new",
    "new_asks_first",
    "new_dialog_text",
    "dialog_blocks_hover",
    "dialog_blocks_cards",
    "dialog_eats_keys",
    "cancel_stays_on_menu",
    "cancel_keeps_save",
    "cards_clickable_again",
    "continue_starts_sim",
    "continue_tick",
    "continue_speed",
    "continue_state",
    "continue_pose",
    "continue_view_built",
    "continue_clears_pick",
    "continue_nothing_selected",
    "stale_no_continue",
    "stale_error_shown",
    "stale_new_button",
    "new_starts_fresh",
    "new_keeps_bad_file",
    "next_save_replaces",
    "next_save_restores",
    "good_save_new_asks",
    "enter_starts_fresh",
    "confirm_keeps_save_until_next",
}


@pytest.mark.gui
def test_save_and_continue_in_app(tmp_path: Path, data_dir: Path) -> None:
    from alife.screens import menu

    shot = tmp_path / "stale_card.png"
    proc = run_script(SCENARIO, str(shot))
    assert_checks(proc, SCENARIO_CHECKS)
    before = [ln for ln in proc.stdout.splitlines() if ln.startswith("TICK BEFORE QUIT")]
    data = read_ok(data_dir)
    assert data["tick"] >= int(before[0].split()[-1])  # app.quit() autosaved too
    # The stale card draws its error line in the error colour.
    from PIL import Image

    with Image.open(shot) as im:
        rgb = im.convert("RGB")
        found = {c for _n, c in rgb.getcolors(maxcolors=rgb.width * rgb.height)}
    target = menu.ERROR_RGB
    assert any(all(abs(a - b) <= 3 for a, b in zip(c, target, strict=True)) for c in found)


# R25: --lifeform ID over a good save opens the menu with that card's CONFIRM_NEW dialog up.
# argv[1] is the answer: "cancel" (Esc) stays on the menu; "yes" (Enter) starts fresh with --seed.
LIFEFORM_OVER_SAVE = textwrap.dedent(
    """
    import sys

    from panda3d.core import KeyboardButton as K

    from alife.__main__ import parse_args
    from alife.app import App
    from alife.screens.menu import MenuScreen
    from alife.screens.simulation import SimulationScreen
    from alife.ui_text import CONFIRM_NEW

    answer = sys.argv[1]
    app = App(parse_args(["--lifeform", "demo", "--seed", "9"]))


    def check(name, ok, detail=""):
        print("CHECK", name, "ok" if ok else "FAIL", detail, flush=True)


    def press(button):
        dev = app.ursina.win.get_input_device(0)
        dev.button_down(button)
        dev.button_up(button)


    def scenario():
        yield 10
        m = app.screen
        on_menu = isinstance(m, MenuScreen)
        check("opens_menu", on_menu, type(m).__name__)
        dialog = m.dialog if on_menu else None
        check("dialog_up", dialog is not None and dialog.message == CONFIRM_NEW)
        check("card_offers_continue", on_menu and m.cards["demo"].continue_button is not None)
        press(K.escape() if answer == "cancel" else K.enter())
        yield 3
        s = app.screen
        if answer == "cancel":
            check("cancel_stays_on_menu", s is m and on_menu and m.dialog is None)
        else:
            # Fresh: the old save is at tick 300; a few frames run at most 16 ticks each.
            in_sim = isinstance(s, SimulationScreen)
            tick = s.session.tick if in_sim else None
            check("yes_starts_fresh", in_sim and tick < 100, tick)
            check("yes_uses_cli_seed", in_sim and s.session.seed == 9)
        app.quit()


    steps = scenario()
    wait = [0]


    def drive(task):
        if wait[0] > 0:
            wait[0] -= 1
            return task.cont
        try:
            wait[0] = next(steps)
        except StopIteration:
            return task.done
        return task.cont


    app.ursina.taskMgr.add(drive, "lifeform-over-save")
    sys.exit(app.run())
    """
)

LIFEFORM_OVER_SAVE_CHECKS = {
    "cancel": {"opens_menu", "dialog_up", "card_offers_continue", "cancel_stays_on_menu"},
    "yes": {
        "opens_menu",
        "dialog_up",
        "card_offers_continue",
        "yes_starts_fresh",
        "yes_uses_cli_seed",
    },
}


@pytest.mark.gui
@pytest.mark.parametrize("answer", ["cancel", "yes"])
def test_lifeform_flag_over_a_save_in_app(answer: str, data_dir: Path) -> None:  # R25 (I2)
    saved = write_demo_save(data_dir)
    path = data_dir / "saves" / "demo.json"
    before = path.read_bytes()
    proc = run_script(LIFEFORM_OVER_SAVE, answer)
    assert_checks(proc, LIFEFORM_OVER_SAVE_CHECKS[answer])
    if answer == "cancel":
        assert path.read_bytes() == before  # Cancel, then quit from the menu: the save is kept
    else:
        data = read_ok(data_dir)  # Yes started fresh; quitting autosaved it over the old save
        assert data["seed"] == 9 != saved["seed"]


# M4: a failed autosave on Esc stays in the sim and shows the toast a failed Cmd+S shows. The
# store is pointed under a file (argv[1]), so it cannot make its folder; then it is fixed.
ESC_SAVE_FAILS = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from panda3d.core import KeyboardButton as K

    from alife.__main__ import parse_args
    from alife.app import App
    from alife.save import SaveStore
    from alife.screens.menu import MenuScreen
    from alife.ui_text import SAVE_FAILED

    blocker = Path(sys.argv[1])
    app = App(parse_args(["--lifeform", "demo", "--seed", "4"]))
    A = K.ascii_key


    def check(name, ok, detail=""):
        print("CHECK", name, "ok" if ok else "FAIL", detail, flush=True)


    def dev():
        return app.ursina.win.get_input_device(0)


    def press(button, raw=False):
        if raw:
            dev().raw_button_down(button)
        dev().button_down(button)
        if raw:
            dev().raw_button_up(button)
        dev().button_up(button)


    def cmd_s():
        dev().button_down(K.meta())
        dev().button_down(K.lmeta())
        press(A("s"), raw=True)
        dev().button_up(K.lmeta())
        dev().button_up(K.meta())


    def scenario():
        yield 20
        s = app.screen
        good = app.store
        app.store = SaveStore(blocker / "saves")
        press(K.escape())
        yield 3
        check("esc_stays_in_sim", app.screen is s, type(app.screen).__name__)
        toast = s.toast.message if s.toast.enabled else None
        prefix = SAVE_FAILED.format(reason="")
        shown = toast is not None and toast.startswith(prefix) and len(toast) > len(prefix)
        check("esc_failure_toast", shown, toast)
        tick = s.session.tick
        yield 5
        check("sim_keeps_running", s.session.tick > tick)
        cmd_s()
        yield 3
        check("cmd_s_same_toast", s.toast.enabled and s.toast.message == toast, s.toast.message)
        app.store = good
        press(K.escape())
        yield 3
        check("esc_then_menu", isinstance(app.screen, MenuScreen))
        check("esc_then_saved", good.read("demo").status == "ok")
        app.quit()


    steps = scenario()
    wait = [0]


    def drive(task):
        if wait[0] > 0:
            wait[0] -= 1
            return task.cont
        try:
            wait[0] = next(steps)
        except StopIteration:
            return task.done
        return task.cont


    app.ursina.taskMgr.add(drive, "esc-save-fails")
    sys.exit(app.run())
    """
)

ESC_SAVE_FAILS_CHECKS = {
    "esc_stays_in_sim",
    "esc_failure_toast",
    "sim_keeps_running",
    "cmd_s_same_toast",
    "esc_then_menu",
    "esc_then_saved",
}


@pytest.mark.gui
def test_failed_esc_autosave_stays_in_sim(tmp_path: Path, data_dir: Path) -> None:  # M4
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, so no saves folder can be made under it")
    proc = run_script(ESC_SAVE_FAILS, str(blocker))
    assert_checks(proc, ESC_SAVE_FAILS_CHECKS)
    lines = proc.stderr.splitlines()
    assert len([ln for ln in lines if ln.startswith(AUTOSAVE_FAILED)]) == 1, proc.stderr  # Esc
    assert len([ln for ln in lines if ln.startswith("alife: save failed")]) == 1  # Cmd+S
    assert read_ok(data_dir)["seed"] == 4  # the Esc after the fix saved
