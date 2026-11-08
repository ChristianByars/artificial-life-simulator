"""Key bindings (SPEC 5, decision U1): Ursina key names to actions. Headless: no Ursina import.

Names are what Ursina 8.3 hands to ``Entity.input`` on macOS. Sources:

- ``ursina/input_handler.py`` (``Keys``: 'escape', 'enter', 'left mouse down', 'scroll up',
  'left shift', 'left control', 'left alt', ...) and ``ursina/main.py`` (``_input_name_changes``
  renames Panda's 'lshift', 'mouse1', 'wheel_up'; ``input()`` strips the 'control-', 'shift-' and
  'alt-' prefixes, but not 'meta-').
- Panda3D 1.10's ``KeyboardButton`` names, which Ursina passes through unchanged: 'space', '.',
  '=', '+', '-', and 'meta' / 'lmeta' / 'rmeta' for Command.
- F5 probe (real events injected into Panda's input device, see the F5 report): Cmd+S arrives as
  'meta', 'lmeta', 's', 'meta-s', 's up', 'lmeta up', 'meta up'. Shift+= arrives as 'shift',
  'left shift', '='. Letters come from Panda's raw (layout-free) events; other keys from the
  layout. The macOS window names a key as if no modifier were held (Panda's
  ``cocoaGraphicsWindow.mm`` uses ``UCKeyTranslate`` with modifier state 0), so Shift+= is '='.
  The keypad sends '+', '-', '.' and 'enter' (Panda 1.10 has no separate keypad names).

Run the app with ``--log-keys`` to see the names on another keyboard or system.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Literal

Action = Literal[
    "pause", "step", "speed_up", "speed_down", "info", "save", "reset", "new", "menu", "focus"
]
DialogAction = Literal["confirm", "cancel"]

# Our modifier names, as used in ``mods`` sets.
SHIFT = "shift"
CONTROL = "control"
ALT = "alt"
COMMAND = "command"

# held_keys names that mean each modifier is down. The generic names ('shift', 'control', 'alt',
# 'meta') come with every press; the left/right names come too (Ursina renames 'lshift' to
# 'left shift', but leaves 'lmeta' as it is). 'command' is our own name: Ursina never sends it,
# but a held map written in our names (tests) works too. Prefixed names such as 'meta-s' are
# deliberately absent: Ursina records their press but the release arrives as 's up', so they
# stay at 1 in held_keys for the rest of the run.
MODIFIER_KEYS: dict[str, tuple[str, ...]] = {
    SHIFT: ("shift", "left shift", "right shift"),
    CONTROL: ("control", "left control", "right control"),
    ALT: ("alt", "left alt", "right alt"),
    COMMAND: ("command", "meta", "lmeta", "rmeta"),
}

# Holding one of these makes a key a shortcut chord: only Cmd/Ctrl+S (save) is ours; Cmd+R,
# Ctrl+Space and the rest fire nothing, and WASD stops panning (so Cmd+S does not pan back).
CHORD_MODIFIERS: frozenset[str] = frozenset({COMMAND, CONTROL})

PAUSE_KEYS = ("space",)
STEP_KEYS = (".",)  # the keypad's decimal key also arrives as '.'
SPEED_UP_KEYS = ("+", "=")  # '=' is the US '+' key without (and, on macOS, with) Shift; '+' keypad
SPEED_DOWN_KEYS = ("-",)  # main keyboard and keypad
INFO_KEYS = ("i",)
SAVE_SHORTCUT_KEYS = ("s",)  # with Command or Control only (U1); plain S pans
RESET_KEYS = ("r",)
NEW_KEYS = ("n",)
MENU_KEYS = ("escape",)
FOCUS_KEYS = ("f",)

BINDINGS: dict[Action, tuple[str, ...]] = {
    "pause": PAUSE_KEYS,
    "step": STEP_KEYS,
    "speed_up": SPEED_UP_KEYS,
    "speed_down": SPEED_DOWN_KEYS,
    "info": INFO_KEYS,
    "save": SAVE_SHORTCUT_KEYS,
    "reset": RESET_KEYS,
    "new": NEW_KEYS,
    "menu": MENU_KEYS,
    "focus": FOCUS_KEYS,
}
_ACTION_BY_KEY: dict[str, Action] = {
    key: action for action, names in BINDINGS.items() for key in names
}

# Confirm dialog (Esc cancels, Enter confirms; the keypad's Enter is 'enter' too).
CONFIRM_KEYS = ("enter",)
CANCEL_KEYS = ("escape",)

LEFT_CLICK = "left mouse down"  # also what Shift+click and Ctrl+click arrive as

# WASD pan: key -> (right, forward).
PAN_KEYS: dict[str, tuple[int, int]] = {"d": (1, 0), "a": (-1, 0), "w": (0, 1), "s": (0, -1)}


def held_mods(held: Mapping[str, float]) -> frozenset[str]:
    """Our names for the modifiers held in ``held`` (Ursina's ``held_keys``)."""
    return frozenset(
        mod for mod, names in MODIFIER_KEYS.items() if any(held.get(name, 0) for name in names)
    )


def action_for(key: str, mods: Set[str]) -> Action | None:
    """The action for one input event ``key`` with ``mods`` held, or None."""
    chord = not CHORD_MODIFIERS.isdisjoint(mods)
    if key in SAVE_SHORTCUT_KEYS:
        return "save" if chord else None
    if chord:
        return None
    return _ACTION_BY_KEY.get(key)


def dialog_action(key: str) -> DialogAction | None:
    """What ``key`` does while a confirm dialog is open: confirm, cancel, or nothing."""
    if key in CONFIRM_KEYS:
        return "confirm"
    if key in CANCEL_KEYS:
        return "cancel"
    return None


def pan_vector(held: Mapping[str, float]) -> tuple[float, float]:
    """(right, forward) from W A S D, each -1..1; zero while Command or Control is held."""
    if not CHORD_MODIFIERS.isdisjoint(held_mods(held)):
        return 0.0, 0.0
    right = forward = 0.0
    for key, (dx, dy) in PAN_KEYS.items():
        amount = float(held.get(key, 0))
        right += dx * amount
        forward += dy * amount
    return right, forward


def describe(key: str, mods: Set[str]) -> str:
    """One ``--log-keys`` line: the key as Ursina names it, then the modifiers held."""
    return f"{key!r} {'+'.join(sorted(mods)) or '-'}"
