"""Atomic save store with validation (SPEC §4).

One slot per life form: ``<user_data_dir>/saves/<lifeform_id>.json``.
A bad file never raises from ``read``; it returns a ``LoadResult`` whose
``message`` is ready to show on the life form's menu card.
Headless module: no Ursina/Panda3D imports.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import platformdirs

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


class SaveError(Exception):
    """A save could not be written. The previous save file is left intact."""


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
