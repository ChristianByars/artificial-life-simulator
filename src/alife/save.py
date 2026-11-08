"""Atomic save store with validation, and building / restoring a save (SPEC §4).

One slot per life form: ``<user_data_dir>/saves/<lifeform_id>.json``.
A bad file never raises from ``read``; it returns a ``LoadResult`` whose
``message`` is ready to show on the life form's menu card.
``build_save`` turns a session and camera pose into a save; ``restore`` turns a save back into a
session and pose, and raises only ``SaveError``.
Headless module: no Ursina/Panda3D imports.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import platformdirs

import alife
from alife import orbit
from alife.session import Session

if TYPE_CHECKING:
    from alife.lifeforms import Lifeform

SCHEMA_VERSION = 1

# The 11 top-level keys of a save file (SPEC §4).
SAVE_KEYS: tuple[str, ...] = (
    "schema_version",
    "app_version",
    "lifeform_id",
    "saved_at",
    "seed",
    "rng_state",
    "params",
    "tick",
    "speed",
    "camera",
    "state",
)

LoadStatus = Literal["ok", "missing", "corrupt", "unknown_schema", "wrong_lifeform"]

# Card-ready messages, one per status.
MSG_OK = ""
MSG_MISSING = "No save yet."
MSG_CORRUPT = "Save file is damaged."
MSG_UNKNOWN_SCHEMA = "Save is from a newer version."
MSG_WRONG_LIFEFORM = "Save belongs to another life form."
# A file that reads fine but does not restore (stale state, bad values), or a saves folder that
# cannot be read.
MSG_RESTORE_FAILED = "Save can't be loaded."

Pose = dict[str, list[float]]


class SaveError(Exception):
    """A save could not be written (the previous file is left intact) or restored."""

    @property
    def reason(self) -> str:
        """A few words for a toast: the OS error text, the cause's type, or the message."""
        cause = self.__cause__
        if isinstance(cause, OSError) and cause.strerror:
            return cause.strerror
        if cause is not None:
            return type(cause).__name__
        return str(self)


@dataclass(frozen=True)
class LoadResult:
    status: LoadStatus
    data: dict | None
    message: str


def saves_dir() -> Path:
    """`ALIFE_DATA_DIR` (tests) or the platform user-data dir, plus `/saves` (M8)."""
    base = os.environ.get("ALIFE_DATA_DIR")
    if not base:
        base = platformdirs.user_data_dir("alife", appauthor=False)
    return Path(base) / "saves"


def store_for_run(test_run: bool) -> SaveStore:
    """The store for one app run. A test run (``--smoke``, ``--screenshot``) without
    ``ALIFE_DATA_DIR`` gets a fresh temp folder, so it never writes the real saves (R21);
    autosave still runs there."""
    if test_run and not os.environ.get("ALIFE_DATA_DIR"):
        return SaveStore(Path(tempfile.mkdtemp(prefix="alife-")) / "saves")
    return SaveStore()


def _reject_constant(name: str) -> Any:
    # json accepts NaN/Infinity by default; a save with them is damaged.
    raise ValueError(f"non-finite number in save: {name}")


class SaveStore:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else saves_dir()

    def path(self, lifeform_id: str) -> Path:
        return self.root / f"{lifeform_id}.json"

    def exists(self, lifeform_id: str) -> bool:
        return self.path(lifeform_id).is_file()

    def write(self, lifeform_id: str, data: dict) -> Path:
        """Write atomically: temp file in the same dir, fsync, then `os.replace`."""
        target = self.path(lifeform_id)
        tmp_name: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=target.parent, prefix=f".{lifeform_id}.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, target)
        except (OSError, TypeError, ValueError, RecursionError) as exc:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            raise SaveError(f"could not write save '{lifeform_id}': {exc}") from exc
        return target

    def read(self, lifeform_id: str) -> LoadResult:
        """Check order: missing, corrupt, unknown_schema, wrong_lifeform, ok."""
        path = self.path(lifeform_id)
        if not path.is_file():
            return LoadResult("missing", None, MSG_MISSING)
        try:
            with path.open(encoding="utf-8") as f:
                obj = json.load(f, parse_constant=_reject_constant)
        except (OSError, ValueError, RecursionError):
            # ValueError covers JSONDecodeError and UnicodeDecodeError.
            return LoadResult("corrupt", None, MSG_CORRUPT)
        if not isinstance(obj, dict) or any(k not in obj for k in SAVE_KEYS):
            return LoadResult("corrupt", None, MSG_CORRUPT)
        version = obj["schema_version"]
        if isinstance(version, bool) or version != SCHEMA_VERSION:
            return LoadResult("unknown_schema", None, MSG_UNKNOWN_SCHEMA)
        if obj["lifeform_id"] != lifeform_id:
            return LoadResult("wrong_lifeform", None, MSG_WRONG_LIFEFORM)
        return LoadResult("ok", obj, MSG_OK)


# --- building and restoring a save ----------------------------------------------------------


def build_save(session: Session, camera_pose: Mapping[str, Any], now: datetime) -> dict:
    """The save for ``session`` with the camera at ``camera_pose``, with every ``SAVE_KEYS`` key.

    ``saved_at`` is ISO 8601 to the second with a UTC offset: an aware ``now`` keeps its own,
    a naive one is taken as local time and gets the local offset.
    """
    snap = session.snapshot()
    stamp = now if now.tzinfo is not None else now.astimezone()
    data = {
        "schema_version": SCHEMA_VERSION,
        "app_version": alife.__version__,
        "lifeform_id": snap["lifeform_id"],
        "saved_at": stamp.isoformat(timespec="seconds"),
        "seed": snap["seed"],
        "rng_state": snap["rng_state"],
        "params": snap["params"],
        "tick": snap["tick"],
        "speed": snap["speed"],
        "camera": {
            "position": [float(v) for v in camera_pose["position"]],
            "rotation": [float(v) for v in camera_pose["rotation"]],
        },
        "state": snap["state"],
    }
    return {key: data[key] for key in SAVE_KEYS}


def restore(data: Any, registry: Mapping[str, type[Lifeform]]) -> tuple[Session, Pose]:
    """Rebuild the session and camera pose from a save (``SaveStore.read`` data).

    Raises ``SaveError`` for anything wrong with the save, including an unknown life form and
    any exception from the life form's ``set_state``, so a bad save never crashes the app.
    ``restore`` itself does not modify ``data``. No view is built: the session is headless.
    """
    try:
        return _restore(data, registry)
    except SaveError:
        raise
    except Exception as exc:  # a life form's set_state may raise anything for stale state
        raise SaveError(f"could not restore save: {type(exc).__name__}: {exc}") from exc


def _restore(data: Any, registry: Mapping[str, type[Lifeform]]) -> tuple[Session, Pose]:
    if not isinstance(data, Mapping):
        raise SaveError(f"save is not an object: {type(data).__name__}")
    missing = [key for key in SAVE_KEYS if key not in data]
    if missing:
        raise SaveError(f"save is missing {', '.join(missing)}")
    version = data["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise SaveError(f"unknown schema_version {version!r}")
    lifeform_id = data["lifeform_id"]
    if not isinstance(lifeform_id, str) or lifeform_id not in registry:
        raise SaveError(f"unknown life form {lifeform_id!r}")
    tick = data["tick"]
    if not _is_int(tick) or tick < 0:
        raise SaveError(f"tick must be a non-negative integer, got {tick!r}")
    if not _is_int(data["seed"]):
        raise SaveError(f"seed must be an integer, got {data['seed']!r}")
    speed = data["speed"]
    if not _is_real(speed) or not math.isfinite(speed):
        raise SaveError(f"speed must be a finite number, got {speed!r}")
    if not isinstance(data["params"], Mapping):
        raise SaveError("params must be an object")
    pose = _camera_pose(data["camera"])
    session = Session.from_snapshot(registry[lifeform_id], dict(data))
    return session, pose


def _camera_pose(camera: Any) -> Pose:
    """The saved pose as floats: a 3-number position and a 2- or 3-number rotation, finite."""
    if not isinstance(camera, Mapping):
        raise SaveError(f"bad camera pose: {camera!r}")
    position, rotation = camera.get("position"), camera.get("rotation")
    if not (_is_numbers(position, (3,)) and _is_numbers(rotation, (2, 3))):
        raise SaveError(f"bad camera pose: {camera!r}")
    pose = {"position": [float(v) for v in position], "rotation": [float(v) for v in rotation]}
    if not all(math.isfinite(v) for v in (*pose["position"], *pose["rotation"])):
        raise SaveError(f"camera pose is not finite: {camera!r}")
    orbit.from_pose(pose)  # the camera accepts it (raises ValueError otherwise)
    return pose


def _is_numbers(value: object, lengths: tuple[int, ...]) -> bool:
    return isinstance(value, list) and len(value) in lengths and all(_is_real(v) for v in value)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
