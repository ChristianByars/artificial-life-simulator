"""Controls (F5, M2): key bindings, modifiers and WASD pan (headless), plus one GUI test that
drives the real app with Panda key and mouse events.

Key names are the ones Ursina hands to ``Entity.input`` on this Mac, recorded in F5 by injecting
events into Panda's input device (see ``alife.keys``). GUI test: ``uv run pytest -m gui``.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import get_args

import pytest

from alife import keys
from alife.keys import Action, action_for, describe, dialog_action, held_mods, pan_vector
from alife.orbit import DEFAULT_POSE

NONE: frozenset[str] = frozenset()


# --- key -> action (headless) ----------------------------------------------------------------


def test_speed_aliases():  # Review Focus 5: "+" arrives as "=" (US layout, macOS)
    assert {"+", "="} <= set(keys.SPEED_UP_KEYS)
    for key in keys.SPEED_UP_KEYS:
        assert action_for(key, NONE) == "speed_up"
        assert action_for(key, frozenset({"shift"})) == "speed_up"  # Shift+= is "+" on US keys


def test_speed_down_aliases():
    assert "-" in keys.SPEED_DOWN_KEYS
    for key in keys.SPEED_DOWN_KEYS:
        assert action_for(key, NONE) == "speed_down"


def test_save_needs_cmd_or_ctrl():  # Review Focus 5, decision U1
    assert action_for("s", frozenset({"command"})) == "save"
    assert action_for("s", frozenset({"control"})) == "save"
    assert action_for("s", frozenset({"command", "shift"})) == "save"
    assert action_for("s", set()) is None
    assert action_for("s", frozenset({"shift"})) is None
    assert action_for("s", frozenset({"alt"})) is None


def test_save_fires_once_for_cmd_s():
    # Cmd+S reaches Entity.input as the raw "s" and then as "meta-s" (Ursina strips only the
    # control-, shift- and alt- prefixes). Only the first may save, or one press saves twice.
    events = ["meta", "lmeta", "s", "meta-s", "s up", "lmeta up", "meta up"]
    mods = frozenset({"command"})
    assert [action_for(e, mods) for e in events].count("save") == 1


@pytest.mark.parametrize(
    ("key", "action"),
    [
        ("space", "pause"),
        (".", "step"),
        ("i", "info"),
        ("r", "reset"),
        ("n", "new"),
        ("escape", "menu"),
        ("f", "focus"),
    ],
)
def test_single_key_bindings(key, action):
    assert action_for(key, NONE) == action
    assert action_for(key, frozenset({"shift"})) == action  # Shift does not change a key


def test_every_action_is_bound_once():
    assert set(keys.BINDINGS) == set(get_args(Action))
    bound = [key for names in keys.BINDINGS.values() for key in names]
    assert len(bound) == len(set(bound))


def test_unknown_key_none():
    assert action_for("q", frozenset()) is None


@pytest.mark.parametrize("key", ["space up", "r up", "r hold", "= hold", "left mouse down", ""])
def test_releases_repeats_and_clicks_are_not_actions(key):
    assert action_for(key, NONE) is None


@pytest.mark.parametrize("mod", ["command", "control"])
@pytest.mark.parametrize("key", ["r", "n", "space", "=", "f", "escape"])
def test_cmd_or_ctrl_chords_fire_no_single_key_action(key, mod):
    # Cmd+R, Ctrl+Space and the like belong to the OS or to later shortcuts; only Cmd/Ctrl+S
    # is ours (judgment call in the F5 report).
    assert action_for(key, frozenset({mod})) is None


# --- confirm dialog keys -----------------------------------------------------------------


def test_dialog_keys():
    assert dialog_action("enter") == "confirm"  # numpad Enter arrives as "enter" too
    assert dialog_action("escape") == "cancel"
    for key in ("space", "r", "n", "y", "enter up", "escape up", "scroll up", "left mouse down"):
        assert dialog_action(key) is None


# --- modifiers and WASD pan (headless) -------------------------------------------------------


def test_held_mods_maps_ursina_names():
    assert held_mods({"meta": 1}) == {"command"}
    assert held_mods({"lmeta": 1}) == {"command"}
    assert held_mods({"rmeta": 1}) == {"command"}
    assert held_mods({"left control": 1}) == {"control"}
    assert held_mods({"control": 1}) == {"control"}
    assert held_mods({"shift": 1, "right alt": 1}) == {"shift", "alt"}
    assert held_mods({"meta": 0, "shift": 0, "w": 1}) == frozenset()


def test_held_mods_ignores_prefixed_names():
    # Seen stuck at 1 in held_keys after Cmd+S and Cmd+click: Ursina records the down event
    # "meta-s" but the release arrives as "s up". They must not count as Command held.
    assert held_mods({"meta-s": 1, "meta-mouse1": 1}) == frozenset()


def test_modifier_alias_table_covers_every_modifier():
    assert set(keys.MODIFIER_KEYS) == {"shift", "control", "alt", "command"}
    for mod, names in keys.MODIFIER_KEYS.items():
        for name in names:
            assert held_mods({name: 1}) == {mod}, name


def test_pan_blocked_with_cmd():
    assert pan_vector({"s": 1, "command": 1}) == (0, 0)
    assert pan_vector({"w": 1}) == (0, 1)


@pytest.mark.parametrize("name", [*keys.MODIFIER_KEYS["command"], *keys.MODIFIER_KEYS["control"]])
def test_pan_blocked_for_every_cmd_and_ctrl_name(name):
    assert pan_vector({"s": 1, "w": 1, "a": 1, name: 1}) == (0, 0)


def test_pan_directions():
    assert pan_vector({}) == (0, 0)
    assert pan_vector({"d": 1}) == (1, 0)
    assert pan_vector({"a": 1}) == (-1, 0)
    assert pan_vector({"s": 1}) == (0, -1)
    assert pan_vector({"w": 1, "d": 1}) == (1, 1)
    assert pan_vector({"w": 1, "s": 1}) == (0, 0)
    assert pan_vector({"w": 1, "shift": 1}) == (0, 1)  # Shift does not block panning
    assert pan_vector({"w": 0, "meta": 0}) == (0, 0)
    right, forward = pan_vector({"w": 1})
    assert isinstance(right, float) and isinstance(forward, float)


def test_describe_for_log_keys():
    assert describe("s", frozenset({"command"})) == "'s' command"
    assert describe("space", NONE) == "'space' -"
    assert describe("=", frozenset({"shift", "control"})) == "'=' control+shift"


# --- default camera pose frames the dish (headless maths) ----------------------------------

URSINA_HFOV = 40.0  # ursina/camera.py: camera.fov = 40, horizontal; Panda keeps it on resize
DEFAULT_WINDOW_ASPECT = 1209 / 785  # Ursina's default window on this Mac (F5 probe)
FRAME_MARGIN = 0.95  # the rim must stay inside 95% of the half-frame


def rim_extent(pose: dict[str, list[float]], aspect: float, radius: float = 10.0) -> float:
    """Largest |screen coordinate| of the dish rim, as a fraction of the half-frame."""
    px, py, pz = pose["position"]
    pitch, yaw = (math.radians(v) for v in pose["rotation"][:2])
    fwd = (math.sin(yaw) * math.cos(pitch), -math.sin(pitch), math.cos(yaw) * math.cos(pitch))
    right = (math.cos(yaw), 0.0, -math.sin(yaw))
    up = (  # right x forward, pointing up for a downward-looking camera
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    )
    tan_h = math.tan(math.radians(URSINA_HFOV / 2))
    tan_v = tan_h / aspect
    worst = 0.0
    for i in range(360):
        a = math.tau * i / 360
        v = (radius * math.cos(a) - px, -py, radius * math.sin(a) - pz)
        depth = sum(v[k] * fwd[k] for k in range(3))
        assert depth > 0, "rim point behind the camera"
        x = sum(v[k] * right[k] for k in range(3)) / depth / tan_h
        y = sum(v[k] * up[k] for k in range(3)) / depth / tan_v
        worst = max(worst, abs(x), abs(y))
    return worst


@pytest.mark.parametrize("aspect", [4 / 3, DEFAULT_WINDOW_ASPECT, 16 / 10, 16 / 9])
def test_default_pose_frames_whole_dish(aspect):  # ruling R18
    assert rim_extent(DEFAULT_POSE, aspect) <= FRAME_MARGIN


def test_rim_extent_catches_the_old_pose():
    # The SPEC 4 example pose showed about 17 of 30 walkers (F2): the check must fail it.
    old = {"position": [0, 8, -12], "rotation": [35, 0, 0]}
    assert rim_extent(old, DEFAULT_WINDOW_ASPECT) > 1.0


# --- the real app (GUI: one short, self-closing window) -------------------------------------

GUI_TIMEOUT = 90  # seconds; a hung run fails instead of blocking the suite

# Drives the app through Panda's own input device (button_down / raw_button_down /
# set_pointer_in_window), so events take the real path: ButtonThrower -> Ursina's key renaming
# -> Entity.input -> mouse.input. Letters are sent raw *and* mapped, as the macOS window does;
# Ursina keeps only the raw letter.
CONTROLS = textwrap.dedent(
    """
    import math
    import sys
    from pathlib import Path

    from panda3d.core import KeyboardButton as K, MouseButton as M, Filename, Point2, Point3
    from ursina import Entity, camera, destroy, mouse, scene, window

    from alife.__main__ import parse_args
    from alife.app import App
    from alife.geometry import DISH_RADIUS
    from alife.orbit import DEFAULT_POSE
    from alife.screens.menu import MenuScreen
    from alife.screens.simulation import SimulationScreen
    from alife.ui_text import CONFIRM_NEW, CONFIRM_RESET

    shots = Path(sys.argv[1])
    app = App(parse_args(["--lifeform", "demo", "--seed", "1", "--log-keys"]))
    A = K.ascii_key
    LETTERS = set("abcdefghijklmnopqrstuvwxyz0123456789")


    def check(name, ok, detail=""):
        print("CHECK", name, "ok" if ok else "FAIL", detail, flush=True)


    def dev():
        return app.ursina.win.get_input_device(0)


    def down(button):
        if button.get_name() in LETTERS:
            dev().raw_button_down(button)
        dev().button_down(button)


    def up(button):
        if button.get_name() in LETTERS:
            dev().raw_button_up(button)
        dev().button_up(button)


    def press(button):
        down(button)
        up(button)


    def shot(name):
        app.ursina.win.save_screenshot(Filename.from_os_specific(str(shots / name)))


    def point_at(px, py):
        dev().set_pointer_in_window(px, py)


    def world_to_pixels(p):
        film = Point2()
        local = app.ursina.cam.get_relative_point(scene, Point3(*p))
        inside = camera.lens.project(local, film)
        w, h = window.size
        return inside, (film.x + 1) / 2 * w, (1 - film.y) / 2 * h, film


    def ui_to_pixels(entity):
        p = entity.get_pos(camera.ui)
        w, h = window.size
        return (p.x / window.aspect_ratio + 0.5) * w, (0.5 - p.y) * h


    def hud(s):
        return [t.text for t in s.hud.texts]


    def alive(e):
        return not e.is_empty() and e in scene.entities


    class KeySpy(Entity):
        def __init__(self):
            super().__init__()
            self.seen = []

        def input(self, key):
            self.seen.append(key)


    DISH_SPOTS = ((0, -6), (5, -5), (-5, -5), (6, 0), (-6, 0), (0, 6), (3, 3), (-3, -3))


    def hover_dish(exclude=()):
        # Point at dish spots until one hovers the dish itself (not a walker in front of it).
        for spot in DISH_SPOTS:
            if spot in exclude:
                continue
            _, px, py, _ = world_to_pixels((spot[0], 0, spot[1]))
            point_at(px, py)
            yield 3
            if getattr(mouse.hovered_entity, "is_dish", False):
                return spot, mouse.world_point
        return None, None


    def scenario():
        yield 40  # the window opens and draws
        s = app.screen
        sim = s.session.lifeform.sim
        check("starts_in_sim", isinstance(s, SimulationScreen))
        check("hud_first_line_is_name", hud(s)[:1] == ["Random walkers (dev)"], hud(s))
        check("hud_running_not_paused", "PAUSED" not in hud(s), hud(s))
        ticking = s.session.tick > 0 and any(t.startswith("Tick ") for t in hud(s))
        check("hud_ticks", ticking, hud(s))
        angles = [math.tau * i / 72 for i in range(72)]
        rim = [world_to_pixels((DISH_RADIUS * math.cos(a), 0, DISH_RADIUS * math.sin(a)))
               for a in angles]
        worst = max(max(abs(f.x), abs(f.y)) for inside, _, _, f in rim)
        check("rim_inside_frame", all(inside for inside, _, _, _ in rim))
        check("default_pose_frames_dish", worst <= 0.95, f"worst={worst:.3f}")
        shot("running.png")

        # speed: "=" (US "+" without Shift), "+" (numpad), "-"
        speeds = []
        for b in (A("="), A("+"), A("-"), A("-"), A("-")):
            press(b)
            yield 2
            speeds.append(s.session.speed)
        check("speed_keys", speeds == [2, 4, 2, 1, 0.5], speeds)
        press(A("="))
        yield 2

        # pause and step
        press(K.space())
        yield 2
        check("space_pauses", s.session.paused)
        t0 = s.session.tick
        press(A("."))
        yield 3
        check("dot_steps_one_tick", s.session.tick == t0 + 1, (t0, s.session.tick))
        yield 5
        check("paused_stays_put", s.session.tick == t0 + 1)
        check("hud_paused", "PAUSED" in hud(s), hud(s))
        check("hud_paused_tick", any(f"Tick {t0 + 1} " in t for t in hud(s)), hud(s))
        shot("paused.png")

        # Cmd+S: no pan while Command is held; plain S pans (positive control)
        target = s.orbit_camera.state.target
        down(K.meta()); down(K.lmeta()); down(A("s"))
        yield 8
        check("cmd_s_does_not_pan", s.orbit_camera.state.target == target)
        up(A("s")); up(K.lmeta()); up(K.meta())
        yield 2
        down(A("s"))
        yield 8
        up(A("s"))
        yield 2
        moved = s.orbit_camera.state.target
        check("plain_s_pans", moved[2] < target[2], (target, moved))
        s.orbit_camera.set_pose(DEFAULT_POSE)
        yield 3

        # left click picks the walker nearest the screen centre
        w, h = window.size
        walkers = list(s.session.lifeform.view.entities.values())
        best = min(walkers, key=lambda e: abs(world_to_pixels(e.world_position)[1] - w / 2)
                   + abs(world_to_pixels(e.world_position)[2] - h / 2))
        _, px, py, _ = world_to_pixels(best.world_position)
        point_at(px, py)
        yield 3
        hovered = mouse.hovered_entity
        check("hover_walker", getattr(hovered, "pick_id", None) is not None, repr(hovered))
        press(M.one())
        yield 3
        picked_id = getattr(hovered, "pick_id", None)
        check("click_picks", picked_id is not None and sim.selected == picked_id,
              (sim.selected, picked_id))
        check("hud_selected", f"Selected #{picked_id}" in hud(s), hud(s))

        # F focuses on the picked walker (target on the ground below it)
        press(A("f"))
        yield 3
        tx, ty, tz = s.orbit_camera.state.target
        wx, _, wz = hovered.world_position
        check("f_focuses", abs(tx - wx) < 1e-4 and ty == 0 and abs(tz - wz) < 1e-4,
              ((tx, ty, tz), (wx, wz)))
        s.orbit_camera.focus((0.0, 0.0, 0.0))
        yield 3

        # the picked walker goes away, as a dead cell would (its view disables the entity):
        # the HUD stops showing it and F does nothing
        sim.walkers = [wk for wk in sim.walkers if wk.id != picked_id]
        yield 3
        check("gone_pick_not_shown", not any(t.startswith("Selected") for t in hud(s)), hud(s))
        press(A("f"))
        yield 2
        check("gone_pick_no_focus", s.orbit_camera.state.target == (0.0, 0.0, 0.0))

        # Shift+click on the dish adds a walker where the dish was hit
        spot, dish_hit = yield from hover_dish()
        check("dish_hover_has_world_point", dish_hit is not None)
        count = len(sim.walkers)
        down(K.shift()); down(K.lshift())
        press(M.one())
        up(K.lshift()); up(K.shift())
        yield 3
        added = sim.walkers[-1].position if len(sim.walkers) == count + 1 else None
        check("shift_click_dish_adds", added is not None and dish_hit is not None
              and abs(added.x - dish_hit[0]) < 1e-4 and abs(added.y - dish_hit[2]) < 1e-4,
              (count, len(sim.walkers), added, dish_hit))
        check("shift_click_keeps_pick", sim.selected == picked_id, sim.selected)

        # a plain click on the dish, then on empty space, both clear the pick. (Not on the spot
        # just clicked: the walker added there is in front of the dish, so it would be picked.)
        other, _ = yield from hover_dish(exclude={spot})
        press(M.one())
        yield 3
        check("dish_click_clears_pick", other is not None and sim.selected is None, sim.selected)
        s.session.lifeform.on_pick(picked_id)
        point_at(3, 3)  # top-left window corner: off the dish
        yield 3
        check("corner_hovers_nothing", mouse.hovered_entity is None, repr(mouse.hovered_entity))
        press(M.one())
        yield 3
        check("empty_click_clears_pick", sim.selected is None, sim.selected)

        # R opens the dialog; while it is open, keys reach neither the sim nor the camera
        check("dialog_closed_at_start", s.dialog is None)
        press(A("r"))
        yield 3
        check("r_opens_dialog", s.dialog is not None and s.dialog.message == CONFIRM_RESET)
        point_at(w / 2, h * 0.9)  # off the dialog's buttons
        yield 2
        shot("dialog.png")
        tick, paused, speed = s.session.tick, s.session.paused, s.session.speed
        pose = s.orbit_camera.get_pose()
        spy = KeySpy()  # created after the screen, so Ursina offers it keys after the screen
        for b in (K.space(), A("."), A("="), A("n"), A("f"), M.wheel_up(), M.wheel_down()):
            press(b)
            yield 1
        down(A("w"))
        yield 6
        up(A("w"))
        dev().button_down(M.three())  # right-drag would orbit
        point_at(w / 2 + 200, h * 0.9)
        yield 4
        dev().button_up(M.three())
        yield 2
        check("dialog_ignores_keys", (s.session.tick, s.session.paused, s.session.speed)
              == (tick, paused, speed), (tick, paused, speed))
        check("dialog_blocks_camera", s.orbit_camera.get_pose() == pose)
        check("dialog_eats_keys", spy.seen == [], spy.seen)
        destroy(spy)
        check("dialog_still_reset", s.dialog is not None and s.dialog.message == CONFIRM_RESET)
        press(K.escape())
        yield 3
        check("esc_cancels_dialog", s.dialog is None and app.screen is s)
        check("cancel_keeps_tick", s.session.tick == tick)

        # R then Enter: back to tick 0, still paused, fresh walkers, pick dropped
        s.session.lifeform.on_pick(picked_id)
        old_walkers = list(s.session.lifeform.view.entities.values())
        old_root = s.view_root
        press(A("r"))
        yield 2
        press(K.enter())
        yield 3
        check("enter_resets", s.dialog is None and s.session.tick == 0)
        check("reset_stays_paused", s.session.paused)
        check("hud_paused_after_reset", "PAUSED" in hud(s), hud(s))
        check("reset_drops_pick", not any(t.startswith("Selected") for t in hud(s)), hud(s))
        check("old_view_destroyed", old_root.is_empty() and not any(alive(e) for e in old_walkers))
        new_walkers = s.session.lifeform.view.entities
        fresh = len(new_walkers) == 30 and all(alive(e) for e in new_walkers.values())
        check("view_rebuilt", fresh, len(new_walkers))
        press(A("f"))  # nothing picked: F does nothing
        yield 2
        check("f_without_pick_noop", s.orbit_camera.state.target == (0.0, 0.0, 0.0))

        # N, then click Yes: new seed, tick 0
        seed = s.session.seed
        press(A("n"))
        yield 3
        check("n_opens_dialog", s.dialog is not None and s.dialog.message == CONFIRM_NEW)
        bx, by = ui_to_pixels(s.dialog.yes_button)
        point_at(bx, by)
        yield 3
        check("hover_yes", mouse.hovered_entity is s.dialog.yes_button, repr(mouse.hovered_entity))
        press(M.one())
        yield 3
        check("yes_starts_new", s.dialog is None and s.session.seed != seed and s.session.tick == 0)

        # resume, then a toast (F6 shows "Saved" through it)
        press(K.space())
        yield 3
        check("space_resumes", not s.session.paused)
        s.toast.show("Saved")
        yield 4
        check("toast_shown", s.toast.enabled and s.toast.message == "Saved")
        shot("toast.png")

        # Esc goes back to the menu and removes the screen's UI
        ui_root = s.ui_root
        press(K.escape())
        yield 3
        check("esc_to_menu", isinstance(app.screen, MenuScreen))
        check("ui_destroyed", ui_root.is_empty())
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


    app.ursina.taskMgr.add(drive, "controls")
    sys.exit(app.run())
    """
)

CONTROLS_CHECKS = {
    "starts_in_sim",
    "hud_first_line_is_name",
    "hud_running_not_paused",
    "hud_ticks",
    "default_pose_frames_dish",
    "rim_inside_frame",
    "speed_keys",
    "space_pauses",
    "dot_steps_one_tick",
    "paused_stays_put",
    "hud_paused",
    "hud_paused_tick",
    "cmd_s_does_not_pan",
    "plain_s_pans",
    "hover_walker",
    "click_picks",
    "hud_selected",
    "f_focuses",
    "gone_pick_not_shown",
    "gone_pick_no_focus",
    "dish_hover_has_world_point",
    "shift_click_dish_adds",
    "shift_click_keeps_pick",
    "dish_click_clears_pick",
    "corner_hovers_nothing",
    "empty_click_clears_pick",
    "dialog_closed_at_start",
    "r_opens_dialog",
    "dialog_ignores_keys",
    "dialog_blocks_camera",
    "dialog_eats_keys",
    "dialog_still_reset",
    "esc_cancels_dialog",
    "cancel_keeps_tick",
    "enter_resets",
    "reset_stays_paused",
    "hud_paused_after_reset",
    "reset_drops_pick",
    "old_view_destroyed",
    "view_rebuilt",
    "f_without_pick_noop",
    "n_opens_dialog",
    "hover_yes",
    "yes_starts_new",
    "space_resumes",
    "toast_shown",
    "esc_to_menu",
    "ui_destroyed",
}


def run_controls(shots: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", CONTROLS, str(shots)],
        capture_output=True,
        text=True,
        timeout=GUI_TIMEOUT,
    )


def colours(path: Path) -> set[tuple[int, int, int]]:
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return {c for _count, c in rgb.getcolors(maxcolors=rgb.width * rgb.height)}


def has_colour(found: set[tuple[int, int, int]], rgb: tuple[int, int, int], tol: int = 2) -> bool:
    return any(all(abs(a - b) <= tol for a, b in zip(c, rgb, strict=True)) for c in found)


@pytest.mark.gui
def test_controls_in_app(tmp_path, data_dir):
    from alife.ui import confirm, hud

    proc = run_controls(tmp_path)
    both = proc.stdout + proc.stderr
    assert proc.returncode == 0, both[-4000:]
    assert "error occurred while compiling GLSL" not in both
    checks = {
        parts[1]: parts[2]
        for parts in (line.split(maxsplit=3) for line in proc.stdout.splitlines())
        if parts and parts[0] == "CHECK"
    }
    assert checks.keys() == CONTROLS_CHECKS, proc.stdout[-4000:]
    failed = sorted(name for name, result in checks.items() if result != "ok")
    assert not failed, (failed, [ln for ln in proc.stdout.splitlines() if "FAIL" in ln])
    # --log-keys printed each event with the modifiers held (Cmd+S came out as "'s' command").
    assert "'s' command" in proc.stdout
    assert "'space' -" in proc.stdout
    assert "'left mouse down' shift" in proc.stdout
    # The HUD and the dialog reach the screen; PAUSED is drawn in its own colour.
    running = colours(tmp_path / "running.png")
    paused = colours(tmp_path / "paused.png")
    dialog = colours(tmp_path / "dialog.png")
    assert not has_colour(running, hud.PAUSED_RGB)
    assert has_colour(paused, hud.PAUSED_RGB)
    assert has_colour(dialog, confirm.PANEL_RGB)
    assert not has_colour(running, confirm.PANEL_RGB)
    assert has_colour(dialog, confirm.YES_RGB, tol=4)
