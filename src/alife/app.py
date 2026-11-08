"""The app shell: one Ursina window, the clock, and the current screen (SPEC 3.1). GUI-only.

Quit path: the window's close button, Cmd+Q, ``--smoke``, ``--screenshot`` and :meth:`App.quit` all
close the window (Cmd+Q is the app menu's Quit item: ``NSApp terminate:``, which Panda's
``applicationShouldTerminate`` turns into closing every window, then cancels). Panda's window-event
then makes ShowBase call ``userExit`` -> ``exitFunc`` (:meth:`App._on_exit`: autosave, quit hooks,
then close the screen) -> ``sys.exit()``. Panda does not always send that event (seen with a
background window), so a per-frame check calls ``userExit`` itself once the window is gone.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

from direct.task import Task  # the module: Task.done / Task.cont are the status ints
from panda3d.core import Filename, GraphicsWindow, WindowProperties, loadPrcFileData
from ursina import Entity, Ursina, application, color, held_keys, window

from alife import keys
from alife.lifeforms import Lifeform, registry
from alife.save import SaveError, restore, store_for_run
from alife.screens.menu import TITLE, MenuScreen
from alife.screens.simulation import SimulationScreen
from alife.session import Session, new_seed
from alife.ui import shaders

BACKGROUND_RGB = (18, 20, 26)  # our assumption: window clear colour (style, not biology)
SCREENSHOT_FRAME = 60  # --screenshot saves the 60th frame drawn with the window open
# Ursina's default icon path is relative and Panda cannot resolve it on macOS (it prints an
# error each run), so pass the same file by absolute path.
ICON = application.package_folder / "textures" / "ursina.ico"


class App:
    """Owns the Ursina app and the current screen. ``run()`` returns the exit code."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lifeforms: dict[str, type[Lifeform]] = registry()
        # <user data dir>/saves, or ALIFE_DATA_DIR. --smoke / --screenshot without ALIFE_DATA_DIR
        # get a throwaway temp folder, so they never overwrite the user's saves (R21).
        test_run = args.smoke is not None or args.screenshot is not None
        self.store = store_for_run(test_run)
        if test_run:
            print(f"alife: test run, saves go to {self.store.root}", file=sys.stderr, flush=True)
        self.screen: MenuScreen | SimulationScreen | None = None
        self.exit_code = 0
        self._quit_hooks: list[Callable[[], None]] = []
        self._exiting = False
        self._frames_open = 0
        self._window_seen_open = False

        shaders.configure_panda()  # gl-version 3 2 and m-dual false, before the window exists
        # Ursina(vsync=...) is ignored; only a PRC setting made before Ursina() works (SPIKE B).
        loadPrcFileData("", "sync-video true")
        self.ursina = Ursina(
            title=TITLE,
            icon=str(ICON),
            fullscreen=False,  # development_mode=False would otherwise default to fullscreen
            borderless=False,
            development_mode=False,  # no hot reloader, editor UI or debug counters
            editor_ui_enabled=False,
            show_ursina_splash=False,
        )
        shaders.install()
        # Ursina sizes the UI lens only when the aspect ratio changes after the window opens.
        # When the window opens at its requested size that never happens, the lens stays 1 x 1,
        # and the UI is drawn 20 times too large: off-screen. Size it once now.
        window.update_aspect_ratio()
        window.color = color.rgba32(*BACKGROUND_RGB)
        self._disable_ursina_builtins()
        self.ursina.exitFunc = self._on_exit
        # Created before any screen, so Ursina offers it every key first (screens eat keys).
        self.key_logger = KeyLogger() if args.log_keys else None

    # screens

    def show_menu(self) -> None:
        self._close_screen()
        self.screen = MenuScreen(self)

    def start(self, lifeform_id: str, seed: int | None = None) -> None:
        """Start ``lifeform_id`` fresh. Without ``seed``: ``--seed`` if given, else a new one."""
        if seed is None:
            seed = self.args.seed if self.args.seed is not None else new_seed()
        session = Session(self.lifeforms[lifeform_id], seed)  # before closing: may raise
        self._close_screen()
        self.screen = SimulationScreen(self, session)

    def resume(self, lifeform_id: str) -> bool:
        """Continue ``lifeform_id`` from its save: state, tick, speed and camera (SPEC §4).

        A save that is missing, damaged or does not restore never crashes the app: print one line
        to stderr and show the menu, whose card then shows the error. The file is left as it is.
        """
        try:
            result = self.store.read(lifeform_id)
            if result.status != "ok":
                raise SaveError(result.message)
            session, pose = restore(result.data, self.lifeforms)
        except (SaveError, OSError) as exc:  # OSError: e.g. the saves folder cannot be read
            print(f"alife: cannot continue {lifeform_id!r}: {exc}", file=sys.stderr, flush=True)
            self.show_menu()
            return False
        self._close_screen()
        self.screen = SimulationScreen(self, session, camera_pose=pose)
        return True

    def _close_screen(self) -> None:
        if self.screen is not None:
            screen, self.screen = self.screen, None
            screen.close()

    # running and quitting

    def run(self) -> int:
        if self.args.lifeform is not None:
            self.start(self.args.lifeform)
        elif self.args.continue_id is not None:
            self.resume(self.args.continue_id)  # falls back to the menu
        else:
            self.show_menu()
        tasks = self.ursina.taskMgr
        tasks.add(self._watch_window, "alife-watch-window")
        if self.args.smoke is not None:
            tasks.doMethodLater(self.args.smoke, self._smoke_task, "alife-smoke")
        if self.args.screenshot is not None:
            tasks.add(self._screenshot_task, "alife-screenshot")
        try:
            self.ursina.run()
        except SystemExit as exc:  # the quit path ends in sys.exit() (ShowBase.finalizeExit)
            if exc.code not in (None, 0):
                raise
        return self.exit_code

    def quit(self) -> None:
        """Quit exactly as the window's close button does: ask Panda to close the window."""
        win = self.ursina.win
        if win is not None and win.is_valid():
            props = WindowProperties()
            props.set_open(False)
            win.request_properties(props)
        else:
            self.ursina.userExit()

    def add_quit_hook(self, hook: Callable[[], None]) -> None:
        """Run ``hook`` on every quit path (window close, Cmd+Q, ``--smoke``, ``quit()``), after
        the autosave."""
        self._quit_hooks.append(hook)

    def _autosave(self) -> None:
        """Save a running simulation on quit. ``autosave`` never raises: quitting goes on."""
        if isinstance(self.screen, SimulationScreen):
            self.screen.autosave()

    def _on_exit(self) -> None:
        if self._exiting:
            return
        self._exiting = True
        for step in [self._autosave, *self._quit_hooks, self._close_screen]:
            try:
                step()
            except Exception:  # one failing hook must not stop the others, or the exit
                traceback.print_exc()
                self.exit_code = 1

    def _watch_window(self, task: Task.Task) -> int:
        """Quit once the window is gone, even if Panda sent no window-event for it.

        Without the event ShowBase never calls ``userExit``, and the app would keep running with
        no window and without its quit hooks. Only "closed" counts: an open window that Panda
        reports as not valid (zero size, e.g. shrunk to no content height) keeps the app running.
        """
        if _window_open(self.ursina.win):
            self._window_seen_open = True
        elif self._window_seen_open:
            self.ursina.userExit()
            return Task.done
        return Task.cont

    def _smoke_task(self, task: Task.Task) -> int:
        self.quit()
        return Task.done

    def _screenshot_task(self, task: Task.Task) -> int:
        if not _window_open(self.ursina.win):
            return Task.cont
        self._frames_open += 1
        if self._frames_open < SCREENSHOT_FRAME:
            return Task.cont
        self._save_screenshot(Path(self.args.screenshot))
        self.quit()
        return Task.done

    def _save_screenshot(self, path: Path) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # The UI is drawn in this window too, so this captures the 3D scene and the UI. (M0's
        # empty captures came from the overlay bug and from capturing before the first frames.)
        if not self.ursina.win.save_screenshot(Filename.from_os_specific(str(path))):
            print(f"alife: could not save a screenshot to {path}", file=sys.stderr)
            self.exit_code = 1

    def _disable_ursina_builtins(self) -> None:
        """Turn off Ursina's built-in keys that quit, reload code or open editor UI."""
        # window.input_entity: F10 / Shift+F10 render mode, F11 fullscreen, F12 editor UI,
        # Ctrl+F10 editor camera.
        _switch_off(window.input_entity)
        # window.exit_button: Ctrl+Shift+Alt+Q and Alt+F4 call sys.exit() directly, which would
        # skip the quit hooks. Off, and routed to our quit path in case it is ever shown.
        _switch_off(window.exit_button)
        window.exit_button.on_click = self.quit
        # Hot reloader: F5 code, F6 textures, F7 models, F8 shaders, F9 auto-reload, Ctrl+R.
        _switch_off(application.hot_reloader)
        application.hot_reloader.hotreload = False
        # Editor UI: cog button and menu (reload and editor-camera buttons), FPS and counters.
        for entity in (window.cog_button, window.cog_menu, window.editor_ui):
            _switch_off(entity)


class KeyLogger(Entity):
    """``--log-keys``: print each input event as Ursina names it, with the modifiers held.

    One line per event (``'s' command``), so the user can check the key names in ``alife.keys``
    on their own keyboard.
    """

    def __init__(self) -> None:
        super().__init__(name="key_logger")

    def input(self, key: str) -> None:
        print(keys.describe(key, keys.held_mods(held_keys)), flush=True)


def _window_open(win: GraphicsWindow | None) -> bool:
    """Open by Panda's own flag. Not ``is_valid()``: that is also False for a zero-size window,
    which is still open (only ``quit()`` needs it, to know a close request can be served)."""
    return win is not None and win.get_properties().get_open()


def _switch_off(entity: Entity) -> None:
    entity.ignore_input = True
    entity.enabled = False
