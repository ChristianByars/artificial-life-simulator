# M0 spike findings

SPEC.md §8, M0. Run 2026-11-07 on the Mac that M4's "300 cells at ≥ 60 FPS" targets.

**Hardware:** Apple M5 Pro, 18-core CPU, 20-core GPU, 48 GB. Built-in Retina display 3024 × 1964.
OpenGL 4.1 / GLSL 4.10 (Metal-backed). Python 3.12.13, Ursina 8.3.0, Panda3D 1.10.16,
libroadrunner 2.11.0, tellurium 2.2.13.1.

**Timing method:** run serially in the controller session with no agents or tests running. Load average 2.8–4.8 (other apps open; Safari's GPU process at ~30% CPU). Render: vsync off, 1600 × 1000-point window, settle 150 frames, then 300 sampled frames of frame-to-frame wall time, 3 runs per setting. Tables show the mean of the run means and the median of the run p95s. Cell cycle: 300 cells, 60 warm-up and 600 sampled ticks, 3 runs.

## Decisions

Status: **approved** by the user at the M0 gate, 2026-11-07.

| # | Decision | Proposal | Evidence |
|---|---|---|---|
| D1 | Tissue Forge | Not used. Own 2D physics (§6.1). App stays on Python 3.12 + uv. | A not run (below). User: "If we don't need it or if there is another alternative, then thats fine." |
| D2 | Cell-cycle backend | **One libroadrunner instance per cell, advanced with `rr.oneStep(t, dt)` every tick.** New cells come from a template state: `RoadRunner()` + `loadStateS(blob)` + `setTime()`. No §6.1 fallback needed. | C3: 1.00 ms mean / 1.13 ms p95 for 300 cells (budget 5 ms). Same period as `simulate` (35.604 min over 8 cycles at the tick size). New instance from template 1.1 ms vs 22 ms from Antimony. |
| D3 | `time_scale` default *(our assumption)* | **0.593** model-minutes per screen-second, so one 35.6-min cycle ≈ 60 s on screen | C2 period 35.60 min; §6.1 "a cycle takes about a minute on screen" |
| D4 | `MPF_THRESHOLD` *(our assumption)* | **0.1** (half the mean MPF peak of 0.193) | Active MPF stays above 0.1 for 3.6% of each cycle: ~2.2 s of a 60 s on-screen cycle shown as *Mitosis*. (0.05 → 4.1 s; 0.15 → 1.1 s.) |
| D5 | Render approach | **`instanced`**: one draw per object kind, per-cell data in buffer textures. Add a per-instance colour/scale channel for phase tint and division/death; draw the selected cell as one extra normal entity. | At 300 cells: p95 4.6 ms vs `naive` 19.2 ms. Largest count at p95 ≤ 16.7 ms: ~1,625 vs ~275. |
| D6 | Mesa | Not used. A colony is one list of cells stepped in order; Mesa's scheduler and grid add nothing. | §6.1 needs no agent framework |

## A. Tissue Forge: not run

Skipped by decision before the spike (D1). Tissue Forge needs its own conda env on Python ≤ 3.11.
§6.1 already specifies our own 2D physics (Brownian motion, repulsion, dish rim), and every
validation test in §6.1 is written against that physics.

**Revisit when** a full-3D life form (tissue, organism) is specced and needs a cell-mechanics
engine.

## B. Ursina render test

Script: `ursina_render/render_spike.py` (README there for how to run it). Scene: a dish plus N
cells. Each cell has a translucent rim-lit membrane, a nucleus, 3 mitochondria and 8 ribosome
specks, all moving every frame.

| Check | Result |
|---|---|
| B1 300 translucent cells with interiors, ≥ 60 FPS | **Pass with `instanced`** (p95 4.6 ms). `naive` fails (p95 19.2 ms). |
| B2 Rim-light shader on OpenGL 4.1 | **Pass**, after two fixes (below). Rim brightens toward the silhouette; checked on window captures. |
| B3 Click picks a cell | **Pass.** Ray pick against an invisible collision sphere per cell: 6/6 probes at 20 cells; 5/6 at 300, where the one miss returned the nearer cell in front, which is the right answer for a click. |

**Three render modes tried:**

| Mode | Nodes at 300 cells | Startup at 300 | Notes |
|---|---|---|---|
| `naive` | 3,900 (one `Entity` per object) | ~1 s | Simplest. Per-cell state (selection, phase tint) is just entity properties. |
| `merged` | 600 (interior baked with `Entity.combine()`) | **~31 s** | ~0.1 s per cell to combine, so every division would stall a frame. Interiors render white (vertex colours lost). **Rejected.** |
| `instanced` | 4 in total (one draw per object kind) | ~1 s | Per-instance positions in a buffer texture. Per-cell colour or selection needs extra shader channels. |

**Frame time (ms), render only.** The spike moves every object each frame in Python; sim cost is
not included.

| Cells | `naive` mean | `naive` p95 | `instanced` mean | `instanced` p95 |
|---:|---:|---:|---:|---:|
| 0 | 0.30 | 0.56 | (spike crashes on an empty buffer) | |
| 100 | 4.30 | 5.46 | 0.85 | 2.41 |
| 200 | 10.82 | 12.68 | 1.36 | 2.94 |
| 225 | 11.89 | 13.17 | | |
| 250 | 13.42 | 14.14 | | |
| **275** | 14.99 | **15.81** | | |
| 285 | 16.64 | 17.29 | | |
| 300 | 18.35 | 19.21 | 2.07 | 4.59 |
| 400 | 31.61 | 33.25 | 2.71 | 6.40 |
| 600 | 58.76 | 62.41 | 3.93 | 7.89 |
| 1,000 | 156.44 | 164.12 | 6.25 | 11.57 |
| 1,500 | | | 9.20 | 14.96 |
| **1,625** | | | 9.98 | **16.10** |
| 1,750 | | | 10.72 | 17.18 |
| 2,000 | | | 12.25 | 18.69 |
| 3,000 | | | 18.12 | 24.56 |

Largest count with p95 ≤ 16.7 ms: **`naive` ~275, `instanced` ~1,625.** A repeat of `naive`
at 300 gave 18.51 / 19.28. `merged` at 300: 3.74 / 5.71, but rejected (startup). Rim shader cost
is small: at 300 cells, without it `naive` p95 is 17.89 (vs 19.21) and `instanced` 4.48 (vs 4.59).

**Must carry into product code:**

1. `loadPrcFileData('', 'gl-version 3 2')` before `Ursina()`. Without it macOS gives OpenGL 2.1 /
   GLSL 1.20.
2. That core profile rejects Ursina 8.3's bundled `#version 130/140` shaders. The worst case is
   `camera.overlay`, a full-window clear quad Ursina always creates (it gets
   `unlit_with_fog_shader`). It draws **over the whole window**. Product code needs GLSL 150 ports of
   `unlit_with_fog_shader` (the default), `unlit_shader` and `text_shader`, which the HUD and menus need.
   *Corrected in Plan 1 F2:* compiled ports alone still leave the window black. Ursina gives every
   model `TransparencyAttrib.M_dual`, which Panda splits with fixed-function alpha tests the core
   profile lacks, so the overlay writes RGBA (0,0,0,0). The fix is `m-dual false` before `Ursina()`
   plus `discard` at alpha ≤ 0 in every shader (`src/alife/ui/shaders.py`). A missing `p3d_Color`
   reads white, not black.
3. Ursina 8's `color.rgba()` takes 0–1 floats; use `color.rgba32()` for 0–255.
4. `vsync=` passed to `Ursina()` is ignored; set `sync-video` with `loadPrcFileData` first.
5. Picking ray from-mask `BitMask32.allOn()`; a `CollisionPlane` must not inherit a rotated
   visual parent's transform.

**Screenshots** work in-app (`--screenshot`, offscreen buffer) and from outside
(`screencapture -l <CGWindowID>`). The spike's original "unsolved screenshot" finding was these
scene bugs, not the capture. *Plan 1 F2:* `base.win.save_screenshot()` also works and captures 3D
plus UI (the offscreen buffer captures only the 3D camera); skip the first frames after launch.

## C. Tellurium per-cell test

Script: `tellurium_cycle/cycle_spike.py` (checks) and `bench_cycle.py` (timing). Model:
`src/alife/lifeforms/cell/models/BIOMD0000000005.xml` (Tyson 1991, CC0; see `SOURCE.md` there).

| Check | Result |
|---|---|
| C1 Install with uv on Apple Silicon | **Pass.** tellurium 2.2.13.1 needs `six` declared separately (dev group). |
| C2 Oscillates, matches published behaviour | **Pass, qualitative.** Period 35.60 min, MPF peaks ≈ 0.19: a slow cyclin sawtooth with sharp narrow MPF spikes, Tyson 1991's spontaneous-oscillator regime. No digitised period from the paper to compare to; `test_cycle_reference` compares against a standalone Tellurium run, as §6.1 says. |
| C3 300 instances, one tick, ≤ 5 ms | **Pass.** `oneStep`: 1.00 ms mean, 1.13 ms p95. `simulate`: 3.14 ms mean, 3.26 ms p95. |
| C4 Save/restore onto a fresh instance | **Pass, bit-exact,** copying time + species + parameters as a dict. `saveStateS`/`loadStateS` copies species but **not time**; call `setTime()` too. |
| C5 Per-instance cyclin synthesis change mid-run | **Pass.** Doubling `k1aa` on one instance gives period 18.42 min on that instance only; `k1aa = 0` stops the cycle (starvation arrest). Needs one Antimony round trip so the SBML local parameters become settable globals. |

**Per-tick cost for 300 cells (ms)**, 600 ticks × 3 runs:

| Method | Mean | p95 | Max |
|---|---:|---:|---:|
| `rr.oneStep`, every cell every tick | 1.00 | 1.13 | 1.37 |
| `rr.simulate(t, t+dt, 2)`, every cell every tick | 3.14 | 3.26 | 23.0 (one outlier per run) |
| `oneStep`, 1/6 of cells per tick (fallback 1) | 0.20 | 0.26 | 0.32 |
| `simulate`, 1/6 of cells per tick | 0.57 | 0.61 | 16.7 |
| RK4, plain Python *(unstable, cost only)* | 0.43 | 0.45 | 0.49 |
| RK4, batched numpy *(unstable, cost only)* | 0.057 | 0.061 | 0.12 |

**Making a new instance (ms)**, median of 20:

| Method | Time |
|---|---:|
| From Antimony, no other instance alive (model recompiled) | 22.1 |
| From Antimony, another instance alive (compiled model reused) | 6.7 |
| `RoadRunner()` + `loadStateS(template)`; parameters still settable | **1.1** |
| Copy one cell: `saveStateS` + new from Antimony + `loadStateS` + `setTime` | 7.9 (p95 8.4) |

**Stiffness.** One reaction (`k8notP = 1e6`) is about a million times faster than the cycle.
Explicit RK4 blows up within 2 steps at the tick size (`dt_model ≈ 0.0099`); it is only stable
below `dt ≈ 2e-6`. libroadrunner's CVODE handles it.

## Open items

1. **§6.1 fallback 2** ("integrate them ourselves"): plain RK4 cannot work here (stiffness,
   above). If that fallback is ever needed, it has to be an implicit solver (e.g. Rosenbrock with
   the model's 6 × 6 Jacobian, batched in numpy). SPEC.md stays as is; the user: "it was just an
   initial spec to get you started." Not needed while D2 holds.
2. **Tyson match is qualitative** (regime and shape), not a digitised published number.
3. **8× speed budget.** At 8× a frame runs 8 ticks: ~9 ms of cell cycle at 300 cells (p95), plus ~4.6 ms render, before physics and nutrients. The planned tick cap (16 per frame, backlog dropped) keeps the app responsive, but 8× may run slower than 8× at 300 cells. Measure in the full app (Plan 2).
