"""Life-form contract and registry (SPEC 3.4).

Each life form is a subpackage of ``alife.lifeforms`` that exposes ``LIFEFORM``, a class
implementing :class:`Lifeform`. The registry finds them; nothing else needs editing to add one.
Package ``__init__`` modules must stay headless: import their view inside ``build_view``.
"""

from __future__ import annotations

import importlib
import pkgutil
import random
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ursina import Entity

    from alife.geometry import Vec2

OUR_ASSUMPTION = "our assumption"


@dataclass(frozen=True)
class InfoNote:
    """One note for the info panel. ``assumed`` marks notes that are our own assumption."""

    title: str
    text: str
    assumed: bool = False


@dataclass(frozen=True)
class Source:
    """A source model or paper (SPEC 1.1)."""

    label: str
    citation: str
    url: str | None = None


@dataclass(frozen=True)
class Param:
    """A tunable parameter: default/min/max and where the value comes from."""

    label: str
    default: float
    min: float
    max: float
    source: str  # a citation, or OUR_ASSUMPTION
    unit: str = ""

    def __post_init__(self) -> None:
        # Written as a positive test so NaN fails it too.
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"Param {self.label!r}: need min <= default <= max, "
                f"got {self.min}, {self.default}, {self.max}"
            )
        if not self.source.strip():
            raise ValueError(f"Param {self.label!r}: source must not be empty")

    def clamp(self, v: float) -> float:
        return float(min(self.max, max(self.min, v)))


class Lifeform(Protocol):
    id: str  # stable key, e.g. "cell"; used in save files
    name: str  # "Cell colony"
    summary: str  # one line for the menu card
    biology: list[InfoNote]  # notes for the info panel (title + text)
    sources: list[Source]  # source models + citations (SPEC 1.1)
    params: dict[str, Param]  # tunable parameters: default/min/max, source or "our assumption"

    # simulation (headless)
    def init(self, rng: random.Random, params: dict) -> None: ...
    def step(self, dt: float) -> None: ...  # advance one fixed tick
    def stats(self) -> dict[str, str | float]: ...  # shown in HUD
    def get_state(self) -> dict: ...  # JSON-serializable
    def set_state(self, state: dict) -> None: ...

    # view (Ursina)
    def build_view(self, root: Entity) -> None: ...
    def sync_view(self) -> None: ...  # called once per frame
    def on_pick(self, entity_id: int | None) -> None: ...  # clicked a 3D object
    def on_dish_click(self, world_pos: Vec2, mods: set) -> None: ...


def registry() -> dict[str, type[Lifeform]]:
    """Discover life-form subpackages, keyed by id and ordered by display name."""
    found: dict[str, type[Lifeform]] = {}
    for info in pkgutil.iter_modules(__path__):
        if not info.ispkg:
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as exc:  # a broken life form must not take the menu down
            print(
                f"warning: skipping life form package {info.name!r}: {exc!r}",
                file=sys.stderr,
            )
            continue
        cls = getattr(module, "LIFEFORM", None)
        if cls is None:
            continue
        if cls.id in found:
            raise ValueError(f"duplicate life form id {cls.id!r} (package {info.name!r})")
        found[cls.id] = cls
    return dict(sorted(found.items(), key=lambda kv: kv[1].name))


def default_params(cls: type[Lifeform]) -> dict[str, float]:
    return {key: float(p.default) for key, p in cls.params.items()}
