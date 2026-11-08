"""Simulation session: one life form, one seeded RNG, and a fixed-timestep clock (SPEC 3.4, 3.5).

Headless: never import ursina here. The app calls ``advance(frame_dt)`` once per frame and
draws what the life form holds afterwards.
"""

from __future__ import annotations

import math
import random

from alife.lifeforms import Lifeform, default_params

TICK_RATE = 60  # SPEC 3.5: base ticks per second
TICK_DT = 1 / TICK_RATE
SPEEDS = (0.25, 0.5, 1, 2, 4, 8)  # SPEC 3.5: 0.25x to 8x
DEFAULT_SPEED = 1.0
MAX_TICKS_PER_FRAME = 16  # our assumption: cap so a slow frame cannot snowball
_EPSILON = 1e-9  # tolerance for float error when the accumulator should be exactly one tick


def new_seed() -> int:
    """A fresh random seed in [0, 2**31)."""
    return random.SystemRandom().randrange(2**31)


def rng_state_to_json(state: tuple) -> list:
    """``random.getstate()`` as JSON-friendly lists (version, 625 ints, gauss_next)."""
    version, internal, gauss_next = state
    return [version, list(internal), gauss_next]


def rng_state_from_json(obj: list) -> tuple:
    """Inverse of :func:`rng_state_to_json`; the result goes to ``Random.setstate``."""
    version, internal, gauss_next = obj
    return (version, tuple(internal), gauss_next)


def _nearest_speed(value: object) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_SPEED
    if not math.isfinite(v):
        return DEFAULT_SPEED if math.isnan(v) else float(SPEEDS[-1 if v > 0 else 0])
    return float(min(SPEEDS, key=lambda s: abs(s - v)))


class Session:
    def __init__(self, lifeform_cls: type[Lifeform], seed: int, params: dict | None = None) -> None:
        merged = default_params(lifeform_cls)
        for key, value in (params or {}).items():
            if key not in lifeform_cls.params:
                raise ValueError(f"unknown parameter {key!r} for life form {lifeform_cls.id!r}")
            merged[key] = value
        self.lifeform_cls = lifeform_cls
        self.seed = seed
        self.params: dict[str, float] = {
            key: lifeform_cls.params[key].clamp(value) for key, value in merged.items()
        }
        self.tick = 0
        self.speed: float = DEFAULT_SPEED
        self.paused = False
        self._accumulator = 0.0
        self.rng = random.Random(seed)
        self.lifeform = self._build()

    def _build(self) -> Lifeform:
        lifeform = self.lifeform_cls()
        lifeform.init(self.rng, self.params)
        return lifeform

    @property
    def sim_time(self) -> float:
        return self.tick * TICK_DT

    # clock

    def advance(self, frame_dt: float) -> int:
        """Run the ticks owed for one rendered frame; return how many ran.

        If more than ``MAX_TICKS_PER_FRAME`` are owed (a stall), run the cap and drop the rest.
        """
        if self.paused or not math.isfinite(frame_dt) or frame_dt < 0:
            return 0
        self._accumulator += frame_dt * self.speed
        owed = int((self._accumulator + _EPSILON) / TICK_DT)
        if owed > MAX_TICKS_PER_FRAME:
            owed = MAX_TICKS_PER_FRAME
            self._accumulator = 0.0
        else:
            self._accumulator = max(0.0, self._accumulator - owed * TICK_DT)
        for _ in range(owed):
            self._tick()
        return owed

    def _tick(self) -> None:
        self.lifeform.step(TICK_DT)
        self.tick += 1

    def step_once(self) -> bool:
        """Run exactly one tick, only while paused."""
        if not self.paused:
            return False
        self._tick()
        return True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def speed_up(self) -> None:
        self._move_speed(1)

    def speed_down(self) -> None:
        self._move_speed(-1)

    def _move_speed(self, delta: int) -> None:
        i = SPEEDS.index(_nearest_speed(self.speed))
        self.speed = float(SPEEDS[min(len(SPEEDS) - 1, max(0, i + delta))])

    # restart

    def reset(self) -> None:
        """Replay from the start: same seed and params, tick 0, fresh life form.

        Speed and pause are kept: they are playback controls, not simulation state.
        """
        self.tick = 0
        self._accumulator = 0.0
        self.rng = random.Random(self.seed)
        self.lifeform = self._build()

    def new(self, seed: int) -> None:
        self.seed = seed
        self.reset()

    # save / load

    def snapshot(self) -> dict:
        return {
            "lifeform_id": self.lifeform_cls.id,
            "seed": self.seed,
            "rng_state": rng_state_to_json(self.rng.getstate()),
            "params": dict(self.params),
            "tick": self.tick,
            "speed": float(self.speed),
            "state": self.lifeform.get_state(),
        }

    @classmethod
    def from_snapshot(cls, lifeform_cls: type[Lifeform], snap: dict) -> Session:
        """Rebuild a session from ``snapshot()`` output (after a JSON round trip).

        Random(seed) -> init -> set_state -> setstate on the same Random object the life
        form holds, so the replay continues exactly.
        """
        session = cls(lifeform_cls, int(snap["seed"]), snap["params"])
        session.lifeform.set_state(snap["state"])
        session.rng.setstate(rng_state_from_json(snap["rng_state"]))
        session.tick = int(snap["tick"])
        session.speed = _nearest_speed(snap["speed"])
        return session
