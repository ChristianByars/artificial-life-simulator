"""One cell's cell-cycle clock: the Tyson 1991 cdc2-cyclin oscillator (SPEC §6.1).

Source: Tyson JJ (1991), "Modeling the cell division cycle: cdc2 and cyclin interactions",
PNAS 88(16):7328-32; BioModels BIOMD0000000005, bundled in models/ (see models/SOURCE.md).
Every rate constant and initial value is read from that file. Model time is minutes.

Backend (SPIKE.md D2): each cell owns one libroadrunner instance. The model is compiled once;
every instance starts from that template's saveStateS() blob. Each advance is one
rr.oneStep with the integrator reset, so a cell's future depends only on its time, species
and parameters, and copy and restore are bitwise exact (plan M12).
"""

from __future__ import annotations

import functools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import roadrunner

MODEL_PATH = Path(__file__).parent / "models" / "BIOMD0000000005.xml"

# Active MPF above this shows as Mitosis: about half the mean MPF peak of 0.193.
MPF_THRESHOLD = 0.1  # our assumption (SPIKE.md D4)

# Model minutes per sim second, so one 35.60-min cycle takes about 60 s on screen.
DEFAULT_TIME_SCALE = 35.60 / 60  # our assumption (SPIKE.md D3)

_MPF = "[M]"  # active MPF, p-cyclin:cdc2 (Tyson's M)
_CYCLIN = "[YT]"  # total cyclin, YT = Y + YP + M + pM (an assignment rule in the model)
# Cyclin synthesis rate k1aa = 0.015 /min (Tyson 1991). The model's rate constants are SBML
# reaction-local parameters, which libroadrunner can't set; promotion renames them globally.
_SYNTHESIS = "Reaction6_k1aa"


@dataclass(frozen=True)
class _Template:
    blob: bytes  # saveStateS() of the compiled model at its initial state
    species_ids: tuple[str, ...]  # C2, CP, M, pM, Y, YP
    initial_species: tuple[float, ...]  # amounts; the compartment volume is 1
    param_ids: tuple[str, ...]  # the ten promoted rate constants
    published_params: tuple[float, ...]
    synthesis_index: int


@functools.cache
def _template() -> _Template:
    sbml = roadrunner.RoadRunner.getParamPromotedSBML(MODEL_PATH.read_text(encoding="utf-8"))
    rr = roadrunner.RoadRunner(sbml)
    model = rr.model
    param_ids = tuple(model.getGlobalParameterIds())
    return _Template(
        blob=rr.saveStateS(),
        species_ids=tuple(model.getFloatingSpeciesIds()),
        initial_species=tuple(model.getFloatingSpeciesAmounts().tolist()),
        param_ids=param_ids,
        published_params=tuple(model.getGlobalParameterValues().tolist()),
        synthesis_index=param_ids.index(_SYNTHESIS),
    )


class CellCycle:
    """A cell's Tyson 1991 model instance. Make one with fresh(), copy() or from_dict()."""

    def __init__(
        self,
        t: float,
        species: Sequence[float],
        params: Sequence[float],
        synthesis_scale: float,
    ) -> None:
        template = _template()
        rr = roadrunner.RoadRunner()
        rr.loadStateS(template.blob)
        rr.model.setFloatingSpeciesAmounts(list(species))
        rr.model.setGlobalParameterValues(list(params))
        rr.model.setTime(t)  # loadStateS doesn't restore time (SPIKE.md C4)
        self._rr = rr
        self._params = tuple(params)  # base values, before synthesis scaling
        self.set_synthesis_scale(synthesis_scale)

    @classmethod
    def fresh(cls) -> CellCycle:
        """A new cell at the model's published initial state: t = 0, synthesis scale 1."""
        template = _template()
        return cls(0.0, template.initial_species, template.published_params, 1.0)

    @property
    def t(self) -> float:
        """Model time in minutes."""
        return self._rr.model.getTime()

    @property
    def mpf(self) -> float:
        """Active MPF (species M)."""
        return self._rr.model[_MPF]

    @property
    def cyclin(self) -> float:
        """Total cyclin (YT)."""
        return self._rr.model[_CYCLIN]

    @property
    def synthesis_scale(self) -> float:
        return self._scale

    @property
    def params(self) -> dict[str, float]:
        """Base rate constants by promoted id, before synthesis scaling."""
        return dict(zip(_template().param_ids, self._params, strict=True))

    def set_synthesis_scale(self, s: float) -> None:
        """Scale cyclin synthesis: effective k1aa = base k1aa × s, s clamped to [0, 1].

        Nutrient coupling is our assumption (SPEC §6.1): starved cells slow down and stop.
        """
        if not math.isfinite(s):
            raise ValueError(f"synthesis scale must be finite, got {s!r}")
        self._scale = min(max(s, 0.0), 1.0)
        i = _template().synthesis_index
        self._rr.model.setGlobalParameterValues([i], [self._params[i] * self._scale])

    def advance(self, dt_model: float) -> None:
        """Integrate dt_model minutes in one step (one sim tick is TICK_DT × time_scale)."""
        if not 0.0 < dt_model < math.inf:
            raise ValueError(f"dt_model must be positive and finite, got {dt_model!r}")
        # reset=True restarts the integrator from the current state: nothing else carries over.
        self._rr.oneStep(self.t, dt_model, reset=True)

    def copy(self, rng: random.Random | None = None, spread: float = 0.0) -> CellCycle:
        """A daughter with this cell's time, species and synthesis scale.

        spread = 0 keeps this cell's parameters exactly. With spread > 0, each parameter is
        its published value × (1 + U(-spread, spread)), drawn from rng in sorted-id order, so
        daughters drift out of sync (our assumption, SPEC §6.1). Spreading around the published
        values, not the parent's, keeps a lineage on Tyson 1991's parameters (ruling R13).
        """
        if not 0.0 <= spread < 1.0:
            raise ValueError(f"spread must be in [0, 1), got {spread!r}")
        params = self._params
        if spread > 0.0:
            if rng is None:
                raise ValueError("a spread copy needs the session's rng")
            template = _template()
            ids = template.param_ids
            factor = {pid: 1 + rng.uniform(-spread, spread) for pid in sorted(ids)}
            params = tuple(
                v * factor[pid] for pid, v in zip(ids, template.published_params, strict=True)
            )
        return CellCycle(self.t, self._species(), params, self._scale)

    def state_vector(self) -> tuple[float, ...]:
        """(t, C2, CP, M, pM, Y, YP): model time, then species amounts in model order."""
        return (self.t, *self._species())

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot; from_dict() rebuilds it bitwise."""
        species = dict(zip(_template().species_ids, self._species(), strict=True))
        return {
            "t": self.t,
            "species": species,
            "params": self.params,
            "synthesis_scale": self._scale,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CellCycle:
        """Rebuild a to_dict() snapshot.

        Raises ValueError if a value is missing, non-finite or not a number, or if the
        species or parameter ids don't match this model (a stale save).
        """
        template = _template()
        try:
            t = _finite(d["t"], "t")
            species = _by_id(d["species"], template.species_ids, "species")
            params = _by_id(d["params"], template.param_ids, "params")
            scale = _finite(d["synthesis_scale"], "synthesis_scale")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"bad cell-cycle state: {exc!r}") from exc
        return cls(t, species, params, scale)

    def _species(self) -> tuple[float, ...]:
        return tuple(self._rr.model.getFloatingSpeciesAmounts().tolist())


def _by_id(values: Mapping[str, Any], ids: tuple[str, ...], what: str) -> tuple[float, ...]:
    if set(values) != set(ids):
        raise ValueError(f"{what} ids {sorted(values)} don't match the model's {sorted(ids)}")
    return tuple(_finite(values[k], f"{what}[{k}]") for k in ids)


def _finite(value: Any, what: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{what} is not finite: {value!r}")
    return x
