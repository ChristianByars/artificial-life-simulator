"""CLI, app shell and shader ports (F2, M1).

Tests marked ``gui`` open a short, self-closing window: ``uv run pytest -m gui``.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from alife.__main__ import main, parse_args

FLAGS = ("--smoke", "--screenshot", "--lifeform", "--continue", "--seed", "--log-keys")
GLSL_ERROR = "error occurred while compiling GLSL"
GL_ERROR = ":display:gsg:glgsg(error)"  # Panda3D writes these to stdout
GUI_TIMEOUT = 60  # seconds; a run that hangs fails instead of blocking the suite


# --- argument parsing (headless) -------------------------------------------------------------


def test_help_lists_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in FLAGS:
        assert flag in out


def test_parse_args_values():
    args = parse_args(
        ["--lifeform", "demo", "--seed", "7", "--smoke", "2.5", "--screenshot", "a.png"]
    )
    assert args.lifeform == "demo"
    assert args.seed == 7
    assert args.smoke == 2.5
    assert args.screenshot == Path("a.png")
    assert args.continue_id is None and args.log_keys is False
    args = parse_args(["--continue", "demo", "--log-keys"])
    assert args.continue_id == "demo" and args.log_keys is True and args.lifeform is None


def test_parse_args_defaults():
    args = parse_args([])
    assert (args.lifeform, args.continue_id, args.seed, args.smoke, args.screenshot) == (
        None,
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "abc"])
def test_smoke_rejects_bad_seconds(value, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--smoke", value])
    assert exc.value.code == 2


def test_lifeform_and_continue_are_exclusive(capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--lifeform", "demo", "--continue", "demo"])
    assert exc.value.code == 2


@pytest.fixture
def no_window(monkeypatch):
    """Replace ``alife.app`` so any attempt to build the App (and open a window) fails."""
    fake = types.ModuleType("alife.app")

    class App:
        def __init__(self, *_a, **_k):
            raise AssertionError("App was built: a window would have opened")

    fake.App = App
    monkeypatch.setitem(sys.modules, "alife.app", fake)


@pytest.mark.parametrize("flag", ["--lifeform", "--continue"])
def test_unknown_lifeform_exits_2(flag, capsys, no_window):
    assert main([flag, "nope"]) == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "nope" in text
    assert "demo" in text  # the known ids are listed


# --- shader ports (headless: imports Ursina, opens nothing) ---------------------------------


def _uniforms(src: str) -> set[str]:
    return set(re.findall(r"^\s*uniform\s+\w+\s+(\w+)\s*;", src, flags=re.MULTILINE))


def _stock_and_ports():
    from ursina.shaders.text_shader import text_shader
    from ursina.shaders.unlit_shader import unlit_shader
    from ursina.shaders.unlit_with_fog_shader import unlit_with_fog_shader

    from alife.ui import shaders

    return [
        (unlit_with_fog_shader, shaders.UNLIT_WITH_FOG),
        (unlit_shader, shaders.UNLIT),
        (text_shader, shaders.TEXT),
    ]


def test_ports_are_glsl_150():
    for _stock, port in _stock_and_ports():
        assert port.vertex.lstrip().startswith("#version 150"), port.name
        assert port.fragment.lstrip().startswith("#version 150"), port.name


def test_ports_keep_stock_uniforms_and_inputs():
    for stock, port in _stock_and_ports():
        assert _uniforms(port.vertex) == _uniforms(stock.vertex), port.name
        assert _uniforms(port.fragment) == _uniforms(stock.fragment), port.name
        assert port.default_input.keys() == stock.default_input.keys(), port.name
        assert port.continuous_input.keys() == stock.continuous_input.keys(), port.name


# --- window watchdog (headless: stub windows, imports Ursina, opens nothing) -----------------


class StubWindow:
    """Stands in for a Panda window. As in Panda, "open" and "valid" are separate: is_valid()
    is also False for a zero-size window, which is still open."""

    def __init__(self, is_open: bool = True, valid: bool = True) -> None:
        self.open = is_open
        self.valid = valid

    def is_valid(self) -> bool:
        return self.valid

    def get_properties(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(get_open=lambda: self.open)


def watchdog(win: StubWindow | None):
    """Drive App._watch_window against a stub; returns (tick, exits, fake_app)."""
    from alife.app import App

    exits: list[bool] = []
    ursina = types.SimpleNamespace(win=win, userExit=lambda: exits.append(True))
    fake_app = types.SimpleNamespace(ursina=ursina, _window_seen_open=False)
    return (lambda: App._watch_window(fake_app, None)), exits, fake_app


def test_watchdog_keeps_running_for_open_zero_size_window():
    from direct.task import Task

    win = StubWindow()
    tick, exits, _ = watchdog(win)
    assert tick() == Task.cont
    win.valid = False  # e.g. shrunk to 0 px content height, or reported as 0x0 when minimised
    for _ in range(5):
        assert tick() == Task.cont
    assert exits == []


def test_watchdog_quits_when_window_closes_or_goes_away():
    from direct.task import Task

    win = StubWindow()
    tick, exits, fake_app = watchdog(win)
    tick()
    win.open = False
    assert tick() == Task.done and exits == [True]

    tick, exits, fake_app = watchdog(StubWindow())
    tick()
    fake_app.ursina.win = None
    assert tick() == Task.done and exits == [True]


def test_watchdog_waits_until_window_first_opens():
    tick, exits, _ = watchdog(StubWindow(is_open=False, valid=False))  # still opening
    for _ in range(5):
        tick()
    assert exits == []


# --- the app (GUI: each test opens a short, self-closing window) -----------------------------


def run_app(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alife", *args],
        capture_output=True,
        text=True,
        timeout=GUI_TIMEOUT,
    )


def assert_clean(proc: subprocess.CompletedProcess[str]) -> None:
    both = proc.stdout + proc.stderr
    assert proc.returncode == 0, both[-3000:]
    assert GLSL_ERROR not in both, both[-3000:]
    assert GL_ERROR not in both, both[-3000:]


def colours(path: Path) -> set[tuple[int, int, int]]:
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return {c for _count, c in rgb.getcolors(maxcolors=rgb.width * rgb.height)}


def colour_counts(path: Path) -> dict[tuple[int, int, int], int]:
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return {c: n for n, c in rgb.getcolors(maxcolors=rgb.width * rgb.height)}


def has_colour(found: set[tuple[int, int, int]], rgb: tuple[int, int, int], tol: int = 2) -> bool:
    return any(all(abs(a - b) <= tol for a, b in zip(c, rgb, strict=True)) for c in found)


def mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))


def to_rgb255(c) -> tuple[int, int, int]:
    return tuple(round(v * 255) for v in (c[0], c[1], c[2]))


@pytest.mark.gui
def test_smoke_menu(data_dir):
    assert_clean(run_app("--smoke", "2"))


@pytest.mark.gui
def test_smoke_demo(data_dir):
    assert_clean(run_app("--lifeform", "demo", "--seed", "1", "--smoke", "2"))


@pytest.mark.gui
def test_no_shader_errors(data_dir):
    proc = run_app("--lifeform", "demo", "--smoke", "2")
    assert GLSL_ERROR not in proc.stderr
    assert GLSL_ERROR not in proc.stdout  # Panda3D's glgsg errors go to stdout
    assert GL_ERROR not in proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stderr[-3000:]


@pytest.mark.gui
def test_screenshot_menu(tmp_path, data_dir):
    from alife import app
    from alife.screens import menu

    out = tmp_path / "menu.png"
    assert_clean(run_app("--screenshot", str(out)))
    assert out.exists()
    found = colours(out)
    assert len(found) > 1
    # Not a black or blank frame: the background, a card and the white title are all drawn.
    assert has_colour(found, app.BACKGROUND_RGB)
    assert has_colour(found, menu.CARD_RGB)
    assert has_colour(found, (255, 255, 255), tol=8)
    # Text colour reaches the screen: Text carries it as vertex colour (p3d_Color).
    assert has_colour(found, menu.SUMMARY_RGB)
    # Alpha blending works: the white name text's anti-aliased edges mix with the card.
    # (Without blending every text pixel is either the text colour or the card colour.)
    assert any(
        has_colour(found, mix(menu.CARD_RGB, (255, 255, 255), t), tol=6) for t in (0.4, 0.5, 0.6)
    )


@pytest.mark.gui
def test_screenshot_demo(tmp_path, data_dir):
    from ursina import color

    from alife.lifeforms.demo import view

    out = tmp_path / "demo.png"
    assert_clean(run_app("--lifeform", "demo", "--seed", "1", "--screenshot", str(out)))
    found = colours(out)
    assert len(found) > 1
    assert has_colour(found, to_rgb255(view.WALKER_COLOR))  # walkers are drawn
    assert has_colour(found, to_rgb255(color.dark_gray))  # so is the dish (demo/view.py)


INTEGRATION = textwrap.dedent(
    """
    import sys

    from panda3d.core import Filename, SamplerState, Texture
    from ursina import Entity, Mesh, Text, application, camera, color, destroy, scene, window
    from ursina.shaders.text_shader import text_shader
    from ursina.shaders.unlit_shader import unlit_shader
    from ursina.shaders.unlit_with_fog_shader import unlit_with_fog_shader

    from alife.__main__ import parse_args
    from alife.app import App
    from alife.screens.menu import MenuScreen
    from alife.screens.simulation import SimulationScreen
    from alife.ui import iter_entities, shaders

    STOCK = (unlit_with_fog_shader, unlit_shader, text_shader)
    RED_HALF = (255, 0, 0, 128)
    GREEN = (0, 255, 0, 255)
    app = App(parse_args(["--lifeform", "demo", "--seed", "1"]))
    app.add_quit_hook(lambda: print("QUIT HOOK RAN", flush=True))
    seen = {}


    def check(name, ok):
        print("CHECK", name, "ok" if ok else "FAIL", flush=True)


    def no_stock_shaders(tag):
        bad = [e for e in iter_entities() if any(e.shader is s for s in STOCK)]
        check(f"{tag}_no_stock_shaders", not bad)
        if bad:
            print("STOCK USERS", [(e.name, e.shader.name) for e in bad], flush=True)


    def step(task):
        n = seen["n"] = seen.get("n", 0) + 1
        if n == 20:
            s = app.screen
            check("starts_in_sim", isinstance(s, SimulationScreen))
            check("sim_ticks", s.session.tick > 0)
            walkers = list(s.session.lifeform.view.entities.values())
            check("walkers_built", len(walkers) == 30)
            check("walkers_use_fog_port", all(w.shader is shaders.UNLIT_WITH_FOG for w in walkers))
            # Stock Ursina gives the overlay unlit_with_fog_shader (the UI swap in entity.py's
            # shader setter only runs on a second assignment), so its port is the fog port.
            check("overlay_uses_port", camera.overlay.shader is shaders.UNLIT_WITH_FOG)
            check("default_shader_is_port", Entity.default_shader is shaders.UNLIT_WITH_FOG)
            # Glyph margin wide enough for the text shader's taps (shaders.PRC_SETTINGS).
            font = next(e for e in iter_entities() if isinstance(e, Text)).font
            check("font_margin_fits_taps", font.get_texture_margin() >= 4)
            no_stock_shaders("sim")
            seen["walkers"] = walkers
            seen["root"] = s.root
            # Render probes over the dish, captured at frame 23 (see test_app_integration).
            # 1) Blending: a half-transparent red quad must mix with the dish behind it.
            blend = Entity(parent=camera.ui, model="quad", scale=0.1, z=-5)
            blend.color = color.rgba32(*RED_HALF)
            # 2) Depth: an invisible quad drawn first, in front, must not hide the green one.
            hole = Entity(parent=camera.ui, model="quad", scale=0.1, x=0.3, z=-20)
            hole.color = color.rgba32(0, 0, 0, 0)
            hole.render_queue = 0
            green = Entity(parent=camera.ui, model="quad", scale=0.1, x=0.3, z=-10)
            green.color = color.rgba32(*GREEN)
            green.render_queue = 1
            # 3) Font-page edge: the text port on an alpha-only texture set up like a Panda
            #    font page (border colour (1,1,1,0)), with UVs running past the page on every
            #    side. Only the centre texel is opaque. Off-page samples must read transparent.
            page = Texture("page_probe")
            page.setup_2d_texture(8, 8, Texture.T_unsigned_byte, Texture.F_alpha)
            texels = bytearray(64)
            texels[4 * 8 + 4] = 255
            page.set_ram_image(bytes(texels))
            page.set_wrap_u(SamplerState.WM_border_color)
            page.set_wrap_v(SamplerState.WM_border_color)
            page.set_border_color((1, 1, 1, 0))
            page.set_minfilter(SamplerState.FT_linear)
            page.set_magfilter(SamplerState.FT_linear)
            quad = Mesh(
                vertices=[(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)],
                triangles=[(0, 1, 2), (0, 2, 3)],
                uvs=[(-1, -1), (2, -1), (2, 2), (-1, 2)],
                colors=[color.magenta] * 4,
            )
            # shader= in the constructor: assigning entity.shader later on a UI entity gets
            # swapped to unlit_shader by Ursina's Entity.shader_setter.
            edge = Entity(
                parent=camera.ui, model=quad, scale=0.2, x=-0.35, z=-5, shader=shaders.TEXT
            )
            edge.model.set_texture(page, 1)
            seen["probes"] = (blend, hole, green, edge)
        elif n == 23:
            app.ursina.win.save_screenshot(Filename.from_os_specific(sys.argv[1]))
            for probe in seen["probes"]:
                destroy(probe)
            # Record any key Ursina still hands to its built-in handlers.
            spied = seen["spied"] = []
            for e in (window.input_entity, window.exit_button, application.hot_reloader):
                e.input = lambda key, name=e.name: spied.append((name, key))
            app.ursina.input("f10")  # stock: next render mode (wireframe)
            app.ursina.input("f11")  # stock: toggle fullscreen
            app.ursina.input("f12")  # stock: show the editor UI
            app.ursina.input("f5")  # stock: hot-reload the code
            app.ursina.input("escape")
        elif n == 27:
            check("render_mode_untouched", window.render_mode == "default")
            check("builtin_keys_ignored", not seen["spied"])
            # window.fullscreen reads True for a windowed start (Ursina bug); ask Panda.
            check("still_windowed", not app.ursina.win.get_properties().get_fullscreen())
            check("editor_ui_hidden", not window.editor_ui.enabled)
            check("hot_reloader_off", not application.hot_reloader.enabled)
            check("escape_shows_menu", isinstance(app.screen, MenuScreen))
            check("root_destroyed", seen["root"] not in scene.entities)
            check("walkers_destroyed", not any(w in scene.entities for w in seen["walkers"]))
            check("menu_has_demo_card", "demo" in app.screen.cards)
            texts = [e for e in iter_entities() if type(e).__name__ == "Text" and e.enabled]
            ported = bool(texts) and all(t.shader is shaders.TEXT for t in texts)
            check("menu_text_uses_text_port", ported)
            no_stock_shaders("menu")
            app.screen.cards["demo"].start_button.on_click()
            app.screen.dialog.confirm()  # Esc autosaved, so New asks first (R20)
        elif n == 31:
            s = app.screen
            check("start_button_starts_sim", isinstance(s, SimulationScreen))
            check("cli_seed_reused", s.session.seed == 1)
            app.quit()
            return task.done
        return task.cont


    app.ursina.taskMgr.add(step, "integration")
    sys.exit(app.run())
    """
)


INTEGRATION_CHECKS = {
    "starts_in_sim",
    "sim_ticks",
    "walkers_built",
    "walkers_use_fog_port",
    "overlay_uses_port",
    "default_shader_is_port",
    "font_margin_fits_taps",
    "sim_no_stock_shaders",
    "render_mode_untouched",
    "builtin_keys_ignored",
    "still_windowed",
    "editor_ui_hidden",
    "hot_reloader_off",
    "escape_shows_menu",
    "root_destroyed",
    "walkers_destroyed",
    "menu_has_demo_card",
    "menu_text_uses_text_port",
    "menu_no_stock_shaders",
    "start_button_starts_sim",
    "cli_seed_reused",
}


@pytest.mark.gui
def test_app_integration(tmp_path, data_dir):
    from ursina import color

    proc = subprocess.run(
        [sys.executable, "-c", INTEGRATION, str(tmp_path / "probes.png")],
        capture_output=True,
        text=True,
        timeout=GUI_TIMEOUT,
    )
    assert_clean(proc)
    checks = dict(
        line.split()[1:3] for line in proc.stdout.splitlines() if line.startswith("CHECK ")
    )
    assert checks.keys() == INTEGRATION_CHECKS, proc.stdout[-3000:]
    failed = sorted(name for name, result in checks.items() if result != "ok")
    assert not failed, (failed, proc.stdout[-3000:])
    assert "QUIT HOOK RAN" in proc.stdout  # the quit path ran the hooks (F6 autosaves there)
    found = colours(tmp_path / "probes.png")
    found_counts = colour_counts(tmp_path / "probes.png")
    dish = to_rgb255(color.dark_gray)
    # Half-transparent quads blend (needs m-dual false: Panda's M_dual split relies on
    # fixed-function alpha tests the core profile lacks, so the quad would come out solid red).
    assert has_colour(found, mix(dish, (255, 0, 0), 128 / 255), tol=3)
    # Invisible pixels write no depth (the ports' discard), so the green quad shows.
    assert has_colour(found, (0, 255, 0))
    # Font-page edge: the opaque centre texel shows (the probe rendered) ...
    magenta = [c for c in found_counts if has_colour({c}, (255, 0, 255), tol=10)]
    assert magenta, "page-edge probe did not render"
    # ... but the off-page area, about 8/9 of the quad, stays clear. With hardware filtering
    # the alpha-swizzled border colour reads as alpha 1 and paints it solid (thousands of px).
    assert sum(found_counts[c] for c in magenta) < 400


WINDOW_GONE = textwrap.dedent(
    """
    import sys

    from alife.__main__ import parse_args
    from alife.app import App

    app = App(parse_args([]))
    app.add_quit_hook(lambda: print("QUIT HOOK RAN", flush=True))
    frames = [0]


    def step(task):
        frames[0] += 1
        if frames[0] == 10:
            # Panda sometimes closes the window without sending the window-event that makes
            # ShowBase call userExit (seen in F2 with a background window). Simulate that: stop
            # ShowBase hearing the event, then quit as usual.
            app.ursina.ignore("window-event")
            app.quit()
            return task.done
        return task.cont


    app.ursina.taskMgr.add(step, "close-window")
    sys.exit(app.run())
    """
)


@pytest.mark.gui
def test_quits_when_window_gone_without_event(data_dir):
    proc = subprocess.run(
        [sys.executable, "-c", WINDOW_GONE], capture_output=True, text=True, timeout=20
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    assert "QUIT HOOK RAN" in proc.stdout
