import subprocess
import sys
import textwrap

# Sim code never imports ursina/panda3d/direct (SPEC 3.2). Walk every alife module
# except the GUI-only ones and check none of them pulled those in.
SCRIPT = textwrap.dedent(
    """
    import importlib, pkgutil, sys

    import alife

    GUI_EXACT = {"alife.__main__", "alife.app", "alife.camera"}
    GUI_PREFIX = ("alife.screens", "alife.ui")

    def is_gui(name):
        return (
            name in GUI_EXACT
            or name.split(".")[-1] == "view"
            or any(name == p or name.startswith(p + ".") for p in GUI_PREFIX)
        )

    imported = []

    def walk(path, prefix):
        for info in pkgutil.iter_modules(path, prefix):
            if is_gui(info.name):
                continue
            mod = importlib.import_module(info.name)
            imported.append(info.name)
            if info.ispkg:
                walk(mod.__path__, info.name + ".")

    walk(alife.__path__, "alife.")
    bad = sorted(
        m for m in sys.modules
        if any(m == p or m.startswith(p + ".") for p in ("ursina", "panda3d", "direct"))
    )
    print("IMPORTED", *imported)
    print("BAD", *bad)
    """
)


def run_walk() -> tuple[set[str], list[str]]:
    proc = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = dict(
        line.split(" ", 1) if " " in line else (line, "") for line in proc.stdout.splitlines()
    )
    return set(lines["IMPORTED"].split()), lines["BAD"].split()


def test_sim_modules_do_not_import_gui_libs():
    imported, bad = run_walk()
    # Not vacuous: the walk really reached the sim modules.
    assert {"alife.geometry", "alife.lifeforms", "alife.lifeforms.demo.sim"} <= imported
    assert bad == []
