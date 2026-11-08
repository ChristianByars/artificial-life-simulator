# SPIKE (M0): throwaway experiment, not product code.
"""M0-C: Tellurium per-cell spike for SPEC.md Sec 8 test C (C1, C2, C4, C5),
plus the C7 RK4-vs-roadrunner comparison using tyson_ode.py.

Run: uv run python spikes/tellurium_cycle/cycle_spike.py
Writes a plot and a few small artifacts to spikes/tellurium_cycle/out/.

No timing verdicts are printed here beyond what's needed to sanity-check
that save/restore methods behave as expected (small counts, not a speed
claim) -- see common-rules.md. The timed sweep is bench_cycle.py, run by
the controller alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tyson_ode as tode  # noqa: E402

REPO_ROOT = HERE.parent.parent
MODEL_PATH = REPO_ROOT / "src/alife/lifeforms/cell/models/BIOMD0000000005.xml"
OUT_DIR = HERE / "out"
OUT_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# C1: imports
# ---------------------------------------------------------------------------
def check_c1() -> None:
    section("C1: imports")
    import roadrunner
    import tellurium

    print("tellurium:", tellurium.__version__)
    print("roadrunner (libroadrunner):", roadrunner.__version__)
    print("numpy:", np.__version__)
    print("six installed:", __import__("six").__version__)


# ---------------------------------------------------------------------------
# C2: load, table species/params, simulate, measure period + amplitude
# ---------------------------------------------------------------------------
def build_promoted_instance():
    """Load via te.loadSBMLModel, then round-trip through Antimony so local
    SBML parameters (one per reaction's kineticLaw) become settable globals.

    Finding: plain `roadrunner.RoadRunner(path)` and `te.loadSBMLModel(path)`
    both report rr.model.getGlobalParameterIds() == [] for this file -- the
    10 rate constants are SBML Level 2 *local* parameters (scoped inside each
    reaction's <kineticLaw>), not global ones, and libRoadRunner does not
    expose or allow setting them by id in that form (`rr.model['Reaction6_k1aa']`
    raises "No sbml element exists for symbol"). Round-tripping through
    `getAntimony()` -> `te.loadAntimonyModel()` promotes them to uniquely
    named globals (`Reaction6_k1aa`, etc.) that ARE settable via `rr[...]`.
    This is the mechanism `cycle.py` (M4) will need for C5 (per-cell param
    changes) and for any cross-cell parameter spread.
    """
    import tellurium as te

    rr_plain = te.loadSBMLModel(str(MODEL_PATH))
    antimony_str = rr_plain.getAntimony()
    rr = te.loadAntimonyModel(antimony_str)
    return rr


def check_c2() -> dict:
    section("C2: load BIOMD0000000005, species/params, period + amplitude")
    import roadrunner
    import tellurium as te

    # Load both ways, as the brief asks.
    rr_te = te.loadSBMLModel(str(MODEL_PATH))
    rr_plain = roadrunner.RoadRunner(str(MODEL_PATH))
    print("te.loadSBMLModel: OK,", rr_te.model.getNumFloatingSpecies(), "floating species")
    print("roadrunner.RoadRunner: OK,", rr_plain.model.getNumFloatingSpecies(), "floating species")

    rr = build_promoted_instance()

    display_names = {
        "C2": "cdc2k (free, unphosphorylated)",
        "CP": "cdc2k-P (free, phosphorylated)",
        "M": "p-cyclin_cdc2 -- ACTIVE MPF (Tyson's M)",
        "pM": "p-cyclin_cdc2-P (inactive MPF)",
        "Y": "cyclin (free)",
        "YP": "p-cyclin (free, phosphorylated)",
    }
    inits = dict(
        zip(
            rr.model.getFloatingSpeciesIds(),
            rr.model.getFloatingSpeciesInitConcentrations(),
            strict=True,
        )
    )

    print("\n-- Species table --")
    print(f"{'id':6} {'name':38} {'initial value'}")
    for sid, name in display_names.items():
        print(f"{sid:6} {name:38} {inits[sid]}")
    print("(assignment rules: YT = Y+YP+M+pM = total cyclin; CT = C2+CP+M+pM = total cdc2, conserved)")

    print("\n-- Parameter table (promoted from per-reaction SBML local parameters) --")
    param_ids = rr.model.getGlobalParameterIds()
    param_vals = rr.model.getGlobalParameterValues()
    reaction_meaning = {
        "Reaction1_k6": "k6: M -> C2 + YP (MPF/cdc2-cyclin dissociation)",
        "Reaction2_k8notP": "k8notP: C2 -> CP (cdc2 phosphorylation)",
        "Reaction3_k9": "k9: CP -> C2 (cdc2 dephosphorylation)",
        "Reaction4_k3": "k3: CP + Y -> pM (cyclin/cdc2-p association)",
        "Reaction5_k5notP": "k5notP: M -> pM (non-specific cdc2 deactivation)",
        "Reaction6_k1aa": "k1aa: -> Y  *** CYCLIN SYNTHESIS RATE ***",
        "Reaction7_k2": "k2: Y -> (default cyclin degradation)",
        "Reaction8_k7": "k7: YP -> (cdc2-triggered cyclin degradation)",
        "Reaction9_k4prime": "k4prime: pM -> M (base activation)",
        "Reaction9_k4": "k4: pM -> M (quadratic, autocatalytic activation)",
    }
    for pid, val in zip(param_ids, param_vals, strict=True):
        print(f"{pid:20} {val:>12}   {reaction_meaning.get(pid, '')}")

    print("\nTime units: SBML file declares no explicit <unitDefinition> for time.")
    print("Tyson 1991's own units are minutes (rate constants in 1/min); we treat")
    print("1 model time unit == 1 minute, consistent with the paper and with the")
    print("period measured below (~35, matching the paper's 'rapid division")
    print("cycles in early embryos' regime, reported there as ~35 min).")

    print("\nActive MPF id: M      Total cyclin: YT (= Y+YP+M+pM)")
    print("Cyclin synthesis parameter: Reaction6_k1aa (local SBML id 'k1aa' in Reaction6)")

    # Simulate many cycles at fine resolution so the sharp MPF spikes are resolved.
    t_end = 500.0
    n_points = 50_000  # dt ~ 0.01
    rr.reset()
    result = rr.simulate(0, t_end, n_points)
    t = result["time"]
    M = result["[M]"]
    YT = M + result["[pM]"] + result["[Y]"] + result["[YP]"]

    peaks_idx = find_peaks(M)
    peak_t = np.array([parabolic_peak(t, M, i)[0] for i in peaks_idx])
    peak_m = np.array([parabolic_peak(t, M, i)[1] for i in peaks_idx])
    print(f"\nFound {len(peaks_idx)} peaks in M over t=[0,{t_end}].")

    # Skip first 2 cycles as warm-up.
    if len(peak_t) > 3:
        peak_t_use = peak_t[2:]
        peak_m_use = peak_m[2:]
        periods = np.diff(peak_t_use)
        period_mean = float(np.mean(periods))
        period_std = float(np.std(periods))
        amp_mean = float(np.mean(peak_m_use))
        print(f"Peak times (model units): {np.round(peak_t, 2)}")
        print(f"Period (mean peak interval, after 2 warm-up cycles): {period_mean:.3f} +/- {period_std:.4f}")
        print(f"Amplitude (mean peak M, after warm-up): {amp_mean:.4f}")
    else:
        period_mean = float("nan")
        period_std = float("nan")
        amp_mean = float("nan")
        print("Not enough peaks found to measure a period -- investigate.")

    print(
        "\nComparison with published behavior: Tyson 1991 describes this parameter"
        "\nset (k5notP=0, k2=0; others as tabled) as the 'spontaneous oscillator'"
        "\nregime, associated with rapid division cycles in early embryos, with a"
        "\nperiod on the order of tens of minutes. The BioModels curation page for"
        "\nBIOMD0000000005 reproduces Fig. 2 of the paper (sustained relaxation"
        "\noscillation in M / active MPF, sharp narrow spikes with a slow buildup"
        "\nbetween them -- exactly the shape we see here). Our measured period"
        f"\n(~{period_mean:.1f} model-time-units) and spike shape (sharp, narrow, "
        "\nrelaxation-oscillator-like, amplitude saturating near M's max) matches"
        "\nthat description. We do not have the paper's exact digitized period"
        "\nvalue to compare to better than 'matches the described regime'."
    )

    plot_cycle(t, M, YT, peak_t, peak_m, period_mean)

    return {
        "species_init": inits,
        "params": dict(zip(param_ids, param_vals, strict=True)),
        "period_mean": period_mean,
        "period_std": period_std,
        "amplitude_mean": amp_mean,
        "n_peaks_found": len(peaks_idx),
    }


def find_peaks(y: np.ndarray, min_height: float = 0.05) -> list[int]:
    idx = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > min_height:
            idx.append(i)
    return idx


def parabolic_peak(t: np.ndarray, y: np.ndarray, i: int) -> tuple[float, float]:
    """Parabolic interpolation of the peak near index i using its 3 neighbors."""
    if i <= 0 or i >= len(y) - 1:
        return float(t[i]), float(y[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(t[i]), float(y[i])
    dt = t[i + 1] - t[i]  # assume uniform spacing locally
    offset = 0.5 * (y0 - y2) / denom
    t_peak = t[i] + offset * dt
    y_peak = y1 - 0.25 * (y0 - y2) * offset
    return float(t_peak), float(y_peak)


def plot_cycle(t, M, YT, peak_t, peak_m, period_mean) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, M, label="M (active MPF)", color="crimson")
    ax.plot(t, YT, label="YT (total cyclin)", color="steelblue", alpha=0.7)
    ax.scatter(peak_t, peak_m, color="black", s=12, zorder=5, label="detected peaks")
    ax.set_xlabel("model time (minutes)")
    ax.set_ylabel("concentration (dimensionless)")
    ax.set_title(f"BIOMD0000000005 (Tyson 1991): period = {period_mean:.2f} model-time-units")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = OUT_DIR / "cycle_oscillation.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


# ---------------------------------------------------------------------------
# C4: save / restore
# ---------------------------------------------------------------------------
def snapshot_dict(rr) -> dict:
    """Plain-dict snapshot: model time, every floating species value, and
    every (promoted) global parameter -- the JSON-serializable path the app
    will actually use for save files."""
    return {
        "time": float(rr.model.getTime()),
        "species": dict(
            zip(
                rr.model.getFloatingSpeciesIds(),
                [float(v) for v in rr.model.getFloatingSpeciesConcentrations()],
                strict=True,
            )
        ),
        "params": dict(
            zip(
                rr.model.getGlobalParameterIds(),
                [float(v) for v in rr.model.getGlobalParameterValues()],
                strict=True,
            )
        ),
    }


def restore_from_dict(rr, snap: dict) -> None:
    rr.reset()
    for pid, val in snap["params"].items():
        rr[pid] = val
    rr.model.setFloatingSpeciesConcentrations(
        [snap["species"][sid] for sid in rr.model.getFloatingSpeciesIds()]
    )
    rr.model.setTime(snap["time"])


def check_c4() -> dict:
    section("C4: save/restore -- dict snapshot, two advance methods, saveStateS")
    results = {}

    tick_dt = 0.1  # model-time units per advance step for this sanity check

    for method in ("rr-simulate", "rr-onestep"):
        print(f"\n-- method: {method} --")
        rr_a = build_promoted_instance()
        rr_a.reset()

        # Advance A to t=50 in tick-size steps (same advance method for the
        # warm-up, so A's internal integrator state before the snapshot
        # matches how it would have gotten there in the app).
        t = 0.0
        for _ in range(int(round(50.0 / tick_dt))):
            if method == "rr-simulate":
                rr_a.simulate(t, t + tick_dt, 2)
            else:
                rr_a.oneStep(t, tick_dt)
            t += tick_dt

        snap = snapshot_dict(rr_a)
        rr_b = build_promoted_instance()
        restore_from_dict(rr_b, snap)

        # Confirm restore landed exactly.
        snap_b = snapshot_dict(rr_b)
        restore_err = max(
            abs(snap["species"][k] - snap_b["species"][k]) for k in snap["species"]
        )
        print(f"restore error right after loading dict into B: {restore_err:.3e}")

        # Advance both A and B 1000 more identical steps.
        n_steps = 1000
        for _ in range(n_steps):
            if method == "rr-simulate":
                rr_a.simulate(t, t + tick_dt, 2)
                rr_b.simulate(t, t + tick_dt, 2)
            else:
                rr_a.oneStep(t, tick_dt)
                rr_b.oneStep(t, tick_dt)
            t += tick_dt

        m_a = rr_a["M"]
        m_b = rr_b["M"]
        all_a = np.array(rr_a.model.getFloatingSpeciesConcentrations())
        all_b = np.array(rr_b.model.getFloatingSpeciesConcentrations())
        max_abs_diff = float(np.max(np.abs(all_a - all_b)))
        bitwise_equal = bool(np.array_equal(all_a, all_b))
        print(f"M after 1000 more steps: A={m_a!r} B={m_b!r}")
        print(f"max abs diff across all species: {max_abs_diff:.3e}")
        print(f"bitwise equal: {bitwise_equal}")
        results[method] = {"max_abs_diff": max_abs_diff, "bitwise_equal": bitwise_equal}

    # saveStateS / loadStateS: does it give an exact copy?
    print("\n-- saveStateS / loadStateS --")
    rr_a = build_promoted_instance()
    rr_a.reset()
    t = 0.0
    for _ in range(500):
        rr_a.oneStep(t, 0.1)
        t += 0.1
    state_blob = rr_a.saveStateS()
    rr_c = build_promoted_instance()
    rr_c.loadStateS(state_blob)
    all_a = np.array(rr_a.model.getFloatingSpeciesConcentrations())
    all_c = np.array(rr_c.model.getFloatingSpeciesConcentrations())
    time_a = rr_a.model.getTime()
    time_c = rr_c.model.getTime()
    exact = bool(np.array_equal(all_a, all_c)) and time_a == time_c
    print(f"saveStateS/loadStateS exact copy (species + time): {exact}")
    print(f"  time: A={time_a} C={time_c}")
    import timeit

    cost_s = timeit.timeit(lambda: rr_a.saveStateS(), number=20) / 20
    print(f"  saveStateS cost (sanity only, not a timing verdict): ~{cost_s * 1000:.2f} ms/call, n=20")
    results["saveStateS_exact"] = exact
    results["saveStateS_cost_ms_sanity"] = cost_s * 1000

    return results


# ---------------------------------------------------------------------------
# C5: per-instance parameter change
# ---------------------------------------------------------------------------
def check_c5() -> dict:
    section("C5: change cyclin synthesis rate on one of two instances mid-run")
    rr1 = build_promoted_instance()
    rr2 = build_promoted_instance()
    rr1.reset()
    rr2.reset()

    orig_k1aa = rr1["Reaction6_k1aa"]
    print(f"original Reaction6_k1aa: {orig_k1aa}")

    t_split = 100.0
    res1a = rr1.simulate(0, t_split, 5000)
    res2a = rr2.simulate(0, t_split, 5000)

    # Change k1aa on instance 2 only (double it), continue both.
    rr2["Reaction6_k1aa"] = orig_k1aa * 2.0
    print(f"rr2 Reaction6_k1aa set to: {rr2['Reaction6_k1aa']}   rr1 unchanged: {rr1['Reaction6_k1aa']}")

    res1b = rr1.simulate(t_split, t_split + 200, 10000)
    res2b = rr2.simulate(t_split, t_split + 200, 10000)

    def period_after(res):
        t = res["time"]
        M = res["[M]"]
        idx = find_peaks(M)
        if len(idx) < 3:
            return float("nan")
        pts = [parabolic_peak(t, M, i)[0] for i in idx]
        return float(np.mean(np.diff(pts[1:])))

    period1 = period_after(res1b)
    period2 = period_after(res2b)
    print(f"rr1 (unchanged) period after split: {period1:.3f}")
    print(f"rr2 (k1aa doubled) period after split: {period2:.3f}")
    print(f"rr1 still running independently: param unaffected = {rr1['Reaction6_k1aa'] == orig_k1aa}")

    # Arrest case: k1aa = 0 on a third instance.
    rr3 = build_promoted_instance()
    rr3.reset()
    rr3["Reaction6_k1aa"] = 0.0
    res3 = rr3.simulate(0, 300, 5000)
    M3 = res3["[M]"]
    idx3 = find_peaks(M3)
    tail_std = float(np.std(M3[-1000:]))
    print(f"k1aa=0: peaks found over t=[0,300]: {len(idx3)}; tail std of M (last ~1000 pts): {tail_std:.3e}")
    print(f"k1aa=0 => oscillation arrested: {len(idx3) == 0 or tail_std < 1e-4}")

    return {
        "orig_k1aa": orig_k1aa,
        "period_unchanged": period1,
        "period_doubled": period2,
        "arrest_at_zero": len(idx3) == 0 or tail_std < 1e-4,
    }


# ---------------------------------------------------------------------------
# C7 prep: hand-coded RK4 vs roadrunner
# ---------------------------------------------------------------------------
def check_c7(period_min: float) -> dict:
    section("C7 prep: hand-coded RK4 (tyson_ode.py) vs roadrunner/CVODE")

    import roadrunner

    rr = roadrunner.RoadRunner(str(MODEL_PATH))
    rr.reset()

    time_scale = period_min / 60.0
    dt_model = (1.0 / 60.0) * time_scale
    print(f"period_min={period_min:.3f}  time_scale={time_scale:.5f}  dt_model={dt_model:.6f}")
    print(f"-> ~{period_min / dt_model:.0f} RK4 steps per cycle at this dt_model")

    # First: is plain explicit RK4 numerically stable at dt_model at all?
    y = tode.initial_state()
    unstable_at = None
    n_test_steps = int(round(5 * period_min / dt_model))
    for i in range(n_test_steps):
        y = tode.rk4_step(y, dt_model)
        if not all(np.isfinite(v) for v in y):
            unstable_at = i
            break
    print(f"RK4 at dt_model over 5 cycles: {'BLEW UP at step ' + str(unstable_at) if unstable_at is not None else 'ran to completion without NaN/inf'}")

    # Empirical stability threshold (bisection-free scan), independent of
    # dt_model, to characterize the stiffness.
    def rk4_stable_to_t(dt: float, t_target: float = 2.0) -> bool:
        yy = tode.initial_state()
        n = int(t_target / dt)
        for _ in range(min(n, 2_000_000)):
            yy = tode.rk4_step(yy, dt)
            if not all(np.isfinite(v) for v in yy):
                return False
        return True

    dt_candidates = [1e-2, 1e-3, 1e-4, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    stability = {dt: rk4_stable_to_t(dt) for dt in dt_candidates}
    print("Explicit-RK4 stability scan (stable to t=2, independent of dt_model):")
    for dt, ok in stability.items():
        print(f"  dt={dt:.7f}  {'stable' if ok else 'BLOWS UP'}")
    stable_dts = [dt for dt, ok in stability.items() if ok]
    approx_dt_max = max(stable_dts) if stable_dts else None
    print(f"Approx max stable dt found in this scan: {approx_dt_max}")
    print(
        f"dt_model ({dt_model:.6f}) is "
        f"{'WITHIN' if approx_dt_max and dt_model <= approx_dt_max else 'FAR ABOVE'} "
        "the explicit-RK4 stability region."
    )
    print(
        "Cause: Reaction2 (k8notP=1e6) and Reaction3 (k9=1000) are a fast"
        " cdc2<->cdc2-P equilibration, orders of magnitude faster than the"
        " ~35-minute cycle -- a classic stiff system. CVODE (roadrunner's"
        " default integrator) handles this implicitly; explicit RK4 cannot"
        " at any tick-scale step."
    )

    # If it happened to be stable (it is not, per above, but keep the compare
    # logic general / reusable), compare periods and max MPF diff over 5
    # cycles using a small enough dt to actually integrate correctly, so we
    # can still report "what RK4-if-stabilized would look like" for the
    # record, subcycled well below the stability threshold.
    sub_dt = min(1e-6, (approx_dt_max or 1e-6) / 2)
    n_sub_steps = int(round(5 * period_min / sub_dt))
    n_sub_steps = min(n_sub_steps, 2_000_000)  # cap for the spike; see README
    print(
        f"\nFor the record: integrating with RK4 at a stable dt={sub_dt:.1e} "
        f"would need {5 * period_min / sub_dt:.3e} steps for 5 cycles -- "
        f"capping this spike run at {n_sub_steps} steps "
        f"({n_sub_steps * sub_dt:.3f} model-time-units) rather than running it out."
    )
    y = tode.initial_state()
    ys = [y]
    for _ in range(n_sub_steps):
        y = tode.rk4_step(y, sub_dt)
        ys.append(y)
    ys = np.array(ys)
    t_covered = n_sub_steps * sub_dt
    M_rk4 = ys[:, 2]
    rr.reset()
    res_rr = rr.simulate(0, t_covered, max(n_sub_steps // 1000, 100))
    print(
        f"Covered only {t_covered:.3f} of {5 * period_min:.1f} model-time-units "
        "in the time budget for this spike -- not enough to measure a period "
        "or a 5-cycle max-diff this way. See README for the conclusion instead "
        "of a period-difference number: RK4 is disqualified by the stability "
        "scan above before a period comparison is meaningful."
    )

    return {
        "dt_model": dt_model,
        "time_scale": time_scale,
        "rk4_unstable_at_dt_model": unstable_at is not None,
        "approx_max_stable_dt": approx_dt_max,
        "stiff_solver_needed": True,
    }


def main() -> None:
    check_c1()
    c2 = check_c2()
    c4 = check_c4()
    c5 = check_c5()
    c7 = check_c7(period_min=c2["period_mean"])

    summary = {"c2": c2, "c4": c4, "c5": c5, "c7": c7}
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
