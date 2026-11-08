# SPIKE (M0): throwaway experiment, not product code.
"""Hand-coded Tyson 1991 cdc2/cyclin ODEs, derived from BIOMD0000000005.

Equations and parameter values are read off the model's reaction rate laws
(see `getAntimony()` output captured in cycle_spike.py / README.md), not
re-derived from the paper. State vector order matches
`roadrunner.RoadRunner.model.getFloatingSpeciesIds()` for this SBML file:

    y = [C2, CP, M, pM, Y, YP]

where (BioModels display names in parens):
    C2  = cdc2k            (free, unphosphorylated cdc2)
    CP  = cdc2k-P          (free, phosphorylated cdc2)
    M   = p-cyclin_cdc2    (active MPF: cyclin-cdc2 complex, cdc2 dephosphorylated)
    pM  = p-cyclin_cdc2-P  (inactive MPF: complex, cdc2 phosphorylated)
    Y   = cyclin           (free, unphosphorylated cyclin)
    YP  = p-cyclin         (free, phosphorylated cyclin -> degraded)

Reactions (cell compartment volume = 1, so the `cell*` factor in the SBML
kinetic laws is dropped here):
    v1 = k6 * M                           # M -> C2 + YP
    v2 = k8notP * C2                      # C2 -> CP
    v3 = k9 * CP                          # CP -> C2
    v4 = k3 * CP * Y                      # CP + Y -> pM
    v5 = k5notP * M                       # M -> pM
    v6 = k1aa                             # -> Y   (cyclin synthesis, our "synthesis param")
    v7 = k2 * Y                           # Y ->
    v8 = k7 * YP                          # YP ->
    v9 = pM * (k4prime + k4 * (M / CT) ** 2)   # pM -> M
where CT = C2 + CP + M + pM (total cdc2, conserved, = 1 at these initial
conditions).

Source: Tyson JJ, "Modeling the cell division cycle: cdc2 and cyclin
interactions", PNAS 1991; 88(16):7328-32. Parameter values below are the
curated BioModels BIOMD0000000005 values (local SBML parameters), cited in
cycle_spike.py's species/parameter table. All six are the published model's
own constants, not our assumptions.
"""

from __future__ import annotations

import numpy as np

# BIOMD0000000005 parameter values (local SBML parameters; see SOURCE.md /
# cycle_spike.py table). These are the published model's own values, cited
# from the bundled SBML file -- not chosen by us.
DEFAULT_PARAMS: dict[str, float] = {
    "k6": 1.0,  # Reaction1: M -> C2 + YP
    "k8notP": 1_000_000.0,  # Reaction2: C2 -> CP
    "k9": 1000.0,  # Reaction3: CP -> C2
    "k3": 200.0,  # Reaction4: CP + Y -> pM
    "k5notP": 0.0,  # Reaction5: M -> pM
    "k1aa": 0.015,  # Reaction6: -> Y  (cyclin synthesis rate)
    "k2": 0.0,  # Reaction7: Y ->
    "k7": 0.6,  # Reaction8: YP ->
    "k4": 180.0,  # Reaction9: pM -> M (quadratic term)
    "k4prime": 0.018,  # Reaction9: pM -> M (base term)
}

# Index order for the plain-Python and numpy state vectors.
SPECIES_ORDER = ("C2", "CP", "M", "pM", "Y", "YP")


def rhs(y: list[float], p: dict[str, float] = DEFAULT_PARAMS) -> list[float]:
    """dy/dt for y = [C2, CP, M, pM, Y, YP]. Plain Python, no numpy."""
    c2, cp, m, pm, yy, yp = y
    ct = c2 + cp + m + pm

    v1 = p["k6"] * m
    v2 = p["k8notP"] * c2
    v3 = p["k9"] * cp
    v4 = p["k3"] * cp * yy
    v5 = p["k5notP"] * m
    v6 = p["k1aa"]
    v7 = p["k2"] * yy
    v8 = p["k7"] * yp
    v9 = pm * (p["k4prime"] + p["k4"] * (m / ct) ** 2) if ct != 0 else 0.0

    dc2 = v1 - v2 + v3
    dcp = v2 - v3 - v4
    dm = -v1 - v5 + v9
    dpm = v4 + v5 - v9
    dy = v6 - v7 - v4
    dyp = v1 - v8

    return [dc2, dcp, dm, dpm, dy, dyp]


def rhs_batch(Y: np.ndarray, P: dict[str, np.ndarray | float] = DEFAULT_PARAMS) -> np.ndarray:
    """Vectorized dy/dt for a batch of cells.

    Y: shape (n, 6), columns [C2, CP, M, pM, Y, YP].
    P: dict of parameter name -> scalar or shape-(n,) array (per-cell
       parameters, e.g. a per-cell cyclin synthesis rate k1aa).
    Returns: shape (n, 6) array of derivatives, same column order.
    """
    c2 = Y[:, 0]
    cp = Y[:, 1]
    m = Y[:, 2]
    pm = Y[:, 3]
    yy = Y[:, 4]
    yp = Y[:, 5]
    ct = c2 + cp + m + pm
    ct_safe = np.where(ct == 0, 1.0, ct)

    v1 = P["k6"] * m
    v2 = P["k8notP"] * c2
    v3 = P["k9"] * cp
    v4 = P["k3"] * cp * yy
    v5 = P["k5notP"] * m
    v6 = np.broadcast_to(P["k1aa"], m.shape).astype(float)
    v7 = P["k2"] * yy
    v8 = P["k7"] * yp
    v9 = pm * (P["k4prime"] + P["k4"] * (m / ct_safe) ** 2)

    dc2 = v1 - v2 + v3
    dcp = v2 - v3 - v4
    dm = -v1 - v5 + v9
    dpm = v4 + v5 - v9
    dy = v6 - v7 - v4
    dyp = v1 - v8

    return np.stack([dc2, dcp, dm, dpm, dy, dyp], axis=1)


def rk4_step(y: list[float], dt: float, p: dict[str, float] = DEFAULT_PARAMS) -> list[float]:
    """One fixed-step classical RK4 step, plain Python."""
    k1 = rhs(y, p)
    y2 = [y[i] + 0.5 * dt * k1[i] for i in range(6)]
    k2 = rhs(y2, p)
    y3 = [y[i] + 0.5 * dt * k2[i] for i in range(6)]
    k3 = rhs(y3, p)
    y4 = [y[i] + dt * k3[i] for i in range(6)]
    k4 = rhs(y4, p)
    return [y[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(6)]


def rk4_step_batch(
    Y: np.ndarray, dt: float, P: dict[str, np.ndarray | float] = DEFAULT_PARAMS
) -> np.ndarray:
    """One fixed-step classical RK4 step for a batch of cells, numpy."""
    k1 = rhs_batch(Y, P)
    k2 = rhs_batch(Y + 0.5 * dt * k1, P)
    k3 = rhs_batch(Y + 0.5 * dt * k2, P)
    k4 = rhs_batch(Y + dt * k3, P)
    return Y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def initial_state() -> list[float]:
    """Model's published initial condition, in SPECIES_ORDER."""
    # C2=0, CP=0.75, M=0, pM=0.25, Y=0, YP=0 (from the SBML file).
    return [0.0, 0.75, 0.0, 0.25, 0.0, 0.0]
