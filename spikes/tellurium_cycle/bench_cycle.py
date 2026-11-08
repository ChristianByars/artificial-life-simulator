# SPIKE (M0): throwaway experiment, not product code.
"""Timing script for SPEC.md Sec 8 test C (C3, C6) and the C7 cost side.

The controller runs this alone, later, for real per-tick timing numbers
(this machine runs another spike concurrently right now, so any numbers
from this session are not meaningful -- see common-rules.md). This script
only needs to be proven to run and to emit the right shape of output.

Usage:
    uv run python spikes/tellurium_cycle/bench_cycle.py \
        --mode {rr-simulate,rr-onestep,py-rk4,np-rk4,create,copy} \
        --cells 300 --ticks 600 --warmup 60 [--every K] --out PATH.json

Modes:
    rr-simulate  advance each active cell with rr.simulate(t, t+dt, 2)
                 (integrator restarts every call)
    rr-onestep   advance each active cell with rr.oneStep(t, dt)
                 (integrator continuity preserved)
    py-rk4       advance each active cell with tyson_ode.rhs/rk4_step,
                 plain Python, one cell at a time
    np-rk4       advance ALL active cells in one batched numpy RK4 step
                 via tyson_ode.rhs_batch/rk4_step_batch
    create       time building one RoadRunner instance from the bundled
                 SBML file (full pipeline: SBML -> Antimony -> compiled
                 instance, since production cells need the promoted,
                 settable cyclin-synthesis parameter -- see C2/C5 in
                 cycle_spike.py for why the plain SBML load alone isn't
                 enough). Reports the first instance and the median of
                 the next 20, and flags whether the next-20 are much
                 faster (a sign libroadrunner is caching the compiled
                 model, not recompiling LLVM IR each time).
    copy         time cloning one instance's state onto a new instance,
                 via the cheapest *exact* method found in C4:
                 saveStateS()/loadStateS() for species values, plus a
                 manual setTime() because C4 found loadStateS does NOT
                 restore model time on its own.

IMPORTANT per C7 (cycle_spike.py): plain explicit RK4 is numerically
UNSTABLE at the tick-scale dt this script defaults to (the model is
stiff -- Reaction2's k8notP=1e6 term). py-rk4/np-rk4 here measure only
the *computational cost* of the RK4 arithmetic at that step size, not a
claim that the fallback produces correct biology at that dt. A real
fallback-2 implementation would need a stiff/implicit method (different,
likely higher, cost) or many RK4 substeps per tick (likely prohibitive --
see cycle_spike.py's stability scan). Flagged in the M0-C report as a
probable SPEC Sec 6.1 fallback-2 revision, not silently assumed away.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tyson_ode as tode  # noqa: E402

REPO_ROOT = HERE.parent.parent
MODEL_PATH = REPO_ROOT / "src/alife/lifeforms/cell/models/BIOMD0000000005.xml"

# our assumption: default dt_model, from cycle_spike.py's measured period
# (~35.6 model-time-units) via dt_model = (1/60) * (period_min/60).
DEFAULT_DT_MODEL = 35.601 / 3600.0


def build_promoted_antimony_string() -> str:
    import tellurium as te

    rr = te.loadSBMLModel(str(MODEL_PATH))
    return rr.getAntimony()


def new_rr_instance(antimony_str: str):
    import tellurium as te

    return te.loadAntimonyModel(antimony_str)


def stats_ms(times_s: list[float]) -> dict:
    arr = np.asarray(times_s) * 1000.0
    return {
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "n": int(arr.size),
    }


def active_indices(n: int, every: int, tick: int) -> list[int]:
    if every <= 1:
        return list(range(n))
    return [i for i in range(n) if tick % every == i % every]


def mode_create(args: argparse.Namespace) -> dict:
    antimony_str = build_promoted_antimony_string()

    t0 = time.perf_counter()
    _ = new_rr_instance(antimony_str)
    first_s = time.perf_counter() - t0

    next_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = new_rr_instance(antimony_str)
        next_times.append(time.perf_counter() - t0)
    median_next20_s = float(np.median(next_times))

    return {
        "mode": "create",
        "first_instance_ms": first_s * 1000.0,
        "median_next20_ms": median_next20_s * 1000.0,
        "appears_cached": bool(median_next20_s < first_s * 0.5),
    }


def mode_copy(args: argparse.Namespace) -> dict:
    antimony_str = build_promoted_antimony_string()
    src = new_rr_instance(antimony_str)
    src.reset()
    t = 0.0
    for _ in range(50):
        src.oneStep(t, args.dt)
        t += args.dt

    times = []
    for _ in range(min(args.ticks, 50)):
        t0 = time.perf_counter()
        blob = src.saveStateS()
        dst = new_rr_instance(antimony_str)
        dst.loadStateS(blob)
        dst.model.setTime(src.model.getTime())  # C4 finding: not restored by loadStateS
        times.append(time.perf_counter() - t0)

    return {"mode": "copy", **stats_ms(times)}


def mode_rr(args: argparse.Namespace, use_onestep: bool) -> dict:
    antimony_str = build_promoted_antimony_string()
    cells = [new_rr_instance(antimony_str) for _ in range(args.cells)]
    for c in cells:
        c.reset()

    total_ticks = args.warmup + args.ticks
    tick_times = []
    t = 0.0
    for tick in range(total_ticks):
        idx = active_indices(args.cells, args.every, tick)
        step_dt = args.dt * args.every
        t0 = time.perf_counter()
        for i in idx:
            c = cells[i]
            if use_onestep:
                c.oneStep(t, step_dt)
            else:
                c.simulate(t, t + step_dt, 2)
        elapsed = time.perf_counter() - t0
        if tick >= args.warmup:
            tick_times.append(elapsed)
        t += args.dt

    return {"mode": "rr-onestep" if use_onestep else "rr-simulate", **stats_ms(tick_times)}


def mode_py_rk4(args: argparse.Namespace) -> dict:
    cells_y = [tode.initial_state() for _ in range(args.cells)]

    total_ticks = args.warmup + args.ticks
    tick_times = []
    for tick in range(total_ticks):
        idx = active_indices(args.cells, args.every, tick)
        step_dt = args.dt * args.every
        t0 = time.perf_counter()
        for i in idx:
            cells_y[i] = tode.rk4_step(cells_y[i], step_dt)
        elapsed = time.perf_counter() - t0
        if tick >= args.warmup:
            tick_times.append(elapsed)

    return {"mode": "py-rk4", **stats_ms(tick_times)}


def mode_np_rk4(args: argparse.Namespace) -> dict:
    Y = np.array([tode.initial_state() for _ in range(args.cells)])

    total_ticks = args.warmup + args.ticks
    tick_times = []
    for tick in range(total_ticks):
        step_dt = args.dt * args.every
        t0 = time.perf_counter()
        if args.every <= 1:
            Y = tode.rk4_step_batch(Y, step_dt)
        else:
            mask = (np.arange(args.cells) % args.every) == (tick % args.every)
            if mask.any():
                Y[mask] = tode.rk4_step_batch(Y[mask], step_dt)
        elapsed = time.perf_counter() - t0
        if tick >= args.warmup:
            tick_times.append(elapsed)

    return {"mode": "np-rk4", **stats_ms(tick_times)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        required=True,
        choices=["rr-simulate", "rr-onestep", "py-rk4", "np-rk4", "create", "copy"],
    )
    p.add_argument("--cells", type=int, default=300)
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--every", type=int, default=1, help="SPEC Sec 6.1 fallback 1: update 1/K of cells per tick")
    p.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT_MODEL,
        help="model-time advanced per tick before --every spreading (our assumption default)",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dispatch = {
        "create": mode_create,
        "copy": mode_copy,
        "rr-simulate": lambda a: mode_rr(a, use_onestep=False),
        "rr-onestep": lambda a: mode_rr(a, use_onestep=True),
        "py-rk4": mode_py_rk4,
        "np-rk4": mode_np_rk4,
    }
    result = dispatch[args.mode](args)
    result["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
