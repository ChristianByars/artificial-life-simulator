# M0-C: Tellurium per-cell spike

Throwaway spike for SPEC.md Sec 8 test C. See
`docs/superpowers/runs/plan-1/reports/m0-c.md` for the full writeup; this
file is just "how to run it."

## Files

- `cycle_spike.py` -- C1 (imports), C2 (load/table/period/plot), C4
  (save/restore), C5 (per-instance parameter change). Run:
  `uv run python spikes/tellurium_cycle/cycle_spike.py`
- `tyson_ode.py` -- hand-coded Tyson 1991 ODEs (`rhs`, `rhs_batch`, RK4
  steppers), derived from BIOMD0000000005's reaction rate laws and
  verified to match `roadrunner`'s own derivative evaluation bit-for-bit
  at two different states. Used by `cycle_spike.py`'s C7 section and by
  `bench_cycle.py`'s `py-rk4`/`np-rk4` modes.
- `bench_cycle.py` -- timing script (C3, C6, C7 cost side). The controller
  runs this alone later for real numbers; see its own docstring for modes
  and usage.
- `out/` -- gitignored outputs (plot, JSON summaries, sanity-run JSON).

## Headline finding (read before using `bench_cycle.py --mode py-rk4/np-rk4`)

Explicit fixed-step RK4 is **numerically unstable** at the tick-scale step
size (`dt_model ~= 0.0099` model-time-units, from `time_scale = period_min
/ 60`). The model is stiff: Reaction2's `k8notP = 1e6` (cdc2
phosphorylation) and Reaction3's `k9 = 1000` are ~5 orders of magnitude
faster than the ~35-minute cycle. `cycle_spike.py`'s stability scan found
RK4 only stays finite below `dt ~= 2e-6`, i.e. about 4,800x smaller than
`dt_model` -- roughly 17,500,000 RK4 steps per cycle, not ~3,600. SPEC.md
Sec 6.1's fallback 2 ("export the model's equations... integrate them
ourselves") should specify an implicit/stiff method (e.g.
`scipy.integrate.solve_ivp(method="Radau"/"LSODA")`, or a semi-implicit
scheme), not RK4, if it's ever implemented. Flagged to the controller;
not silently changed here.

`bench_cycle.py --mode py-rk4/np-rk4` still runs and still measures a real
per-tick wall-clock cost (useful for comparing raw arithmetic cost against
`rr-simulate`/`rr-onestep`), but the trajectory it produces at that `dt`
is biologically meaningless -- expect (and ignore) numpy overflow warnings
in `np-rk4`.

## Parameter access gotcha

Plain `roadrunner.RoadRunner(path)` and `te.loadSBMLModel(path)` both
report zero global parameters for this file -- the model's 10 rate
constants are SBML Level 2 **local** parameters, one per reaction's
`<kineticLaw>`, and libRoadRunner does not expose them for
get/set-by-id in that form. Round-tripping through Antimony
(`rr.getAntimony()` then `te.loadAntimonyModel(that_string)`) promotes
them to uniquely-named globals (`Reaction6_k1aa`, etc.) that *are*
settable via `rr['Reaction6_k1aa'] = value`. `cycle.py` (M4) will need
this for per-cell cyclin-synthesis coupling to energy.
