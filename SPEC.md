# Artificial Life Simulator — Spec

> Source of truth for implementation. Specs are discussed in claude.ai and written here; Claude Code implements from this file.
> Version 0.3 · 2026-11-07
> - 0.3: biology rule (§1.1: every life form names a proven source model and ships a test against known results); Tellurium + Tyson 1991 cell-cycle model now core to the cell colony; M0 adds a Tellurium speed test; roadmap lists source models and tests.
> - 0.2: 3D rendering (Ursina/Panda3D replaces pygame), cells as Python objects instead of arrays, biology packages, M0 test milestone.

## 1. Goal

A desktop app where users launch the app, pick a life form, and watch it simulate in 3D. Users can orbit the camera, pause, change speed, save progress, load it later, and reset.

Secondary goal: **learning biology.** Every life form models a real biological system and explains what it shows in a built-in info panel.

### Non-goals (v1)
- Simulating every molecule. Life forms work at the cell level using proven models (§1.1); molecule-level detail appears only in dedicated life forms (e.g. `molecules`).
- Networking, accounts, cloud saves

### 1.1 Biology rule: proven models only

No invented biology. Every life form must:

1. **Name its source model**: a published model run through a package (e.g. a BioModels entry in Tellurium), or a textbook model with published parameters (e.g. the Hill equation for oxygen binding). Record the citation in the life form's `sources`.
2. **Use published parameters**, with the source noted next to each value. Any value we choose ourselves (e.g. how nutrients scale a rate, or a time-scale factor) is labeled **our assumption** in code and in the info panel.
3. **Ship a validation test** in `tests/validation/` showing the simulation reproduces a known result from the source (a period, a curve, a drift direction). The life form isn't done until it passes.
4. **Show the source in the app**: the info panel names the model and marks which behaviors are simplified or assumed.

Prefer a package when one implements the source model directly and passes M0-style checks (installs on the Mac, steps tick by tick, state can be saved and restored, fast enough). Otherwise implement the textbook model ourselves, held to the same validation test.

## 2. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.12+ |
| Env / deps | `uv` (`pyproject.toml`) |
| 3D window, input, rendering, UI | `ursina` (built on `panda3d`; drop to Panda3D APIs for custom shaders or instancing) |
| Cell-cycle biochemistry | `tellurium` / `libroadrunner` (runs published SBML models; required by the cell colony) |
| Environment fields | `numpy` (nutrient and chemical grids are naturally arrays; the objects rule in §3.3 applies to living entities) |
| Save location | `platformdirs` (user data dir) |
| Tests / lint | `pytest`, `ruff` |
| Packaging (later) | Panda3D `build_apps` or PyInstaller → macOS `.app`, Windows `.exe` |

**Biology packages added per life form** (only when that life form is built):

| Package | Used for |
|---|---|
| `brian2` | Neurons life form (spiking networks) |
| `smoldyn` | Molecules life form; confirm macOS arm64 wheels first |
| `mesa` | Optional agent-model backbone; decide in M0 |
| Tissue Forge | Candidate physics backend; evaluated in M0 |

Target machine: Apple Silicon Mac (M5 Pro). Must also run on Windows.

**Platform constraint:** macOS caps OpenGL at 4.1, so no GPU compute shaders. Simulation runs on the CPU; the GPU is for rendering only.

## 3. Architecture

### 3.1 App screens (state machine)

```
Menu ──pick──▶ Simulation ──Esc──▶ Menu
                 │  ▲
             Space│  │Space
                 ▼  │
               Paused (overlay)
```

- `MenuScreen`: cards, one per registered life form (name, one-line description, Continue button if a save exists).
- `SimulationScreen`: runs the selected life form, owns the 3D scene, HUD and controls.
- One `App` owns the Ursina app, the clock, and the current screen.

### 3.2 Sim and view are separate

**The simulation never imports Ursina.** It is plain Python, so it can be tested headless, saved, and stepped faster than real time.

```
lifeforms/cell/sim.py   ← Cell, Nucleus, Mitochondrion… plain Python objects
lifeforms/cell/view.py  ← maps cell.id → Ursina Entity; syncs transforms each frame
```

The view reads sim objects and updates their 3D entities. It creates entities for new cells and removes them for dead ones.

### 3.3 Entities are objects, not arrays

Model biology the way it reads:

```python
@dataclass
class Cell:
    id: int
    position: Vec2          # on the dish plane
    velocity: Vec2
    radius: float
    energy: float
    age: float
    generation: int
    cycle: CellCycle        # wraps this cell's Tyson 1991 model instance (§6.1)
    nucleus: Nucleus
    mitochondria: list[Mitochondrion]

    @property
    def phase(self) -> CellPhase: ...   # derived from cycle state, not stored

    def step(self, env: Environment, dt: float) -> list[Event]: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Cell": ...
```

- Do **not** vectorize up front. Only rewrite a hot path to arrays or `numba` after profiling shows it misses the frame budget.
- Neighbor queries use a uniform-grid spatial hash, not all-pairs checks.

### 3.4 Life form plugin interface

Each life form is a package in `src/alife/lifeforms/<id>/`, registered automatically. Adding a life form must not require changing the menu, the save system, or the app loop.

```python
class Lifeform(Protocol):
    id: str                   # stable key, e.g. "cell"; used in save files
    name: str                 # "Cell colony"
    summary: str              # one line for the menu card
    biology: list[InfoNote]   # notes for the info panel (title + text)
    sources: list[Source]     # source models + citations (§1.1)
    params: dict[str, Param]  # tunable parameters: default/min/max, source or "our assumption"

    # simulation (headless)
    def init(self, rng: random.Random, params: dict) -> None: ...
    def step(self, dt: float) -> None: ...            # advance one fixed tick
    def stats(self) -> dict[str, str | float]: ...    # shown in HUD
    def get_state(self) -> dict: ...                  # JSON-serializable
    def set_state(self, state: dict) -> None: ...

    # view (Ursina)
    def build_view(self, root: Entity) -> None: ...
    def sync_view(self) -> None: ...                  # called once per frame
    def on_pick(self, entity_id: int | None) -> None: ...      # clicked a 3D object
    def on_dish_click(self, world_pos: Vec2, mods: set) -> None: ...
```

### 3.5 Simulation loop

- **Fixed timestep** (60 ticks/s base), decoupled from frame rate. Speed multiplier runs more ticks per frame (0.25×–8×). Cap ticks per frame so slow frames don't snowball.
- **Deterministic**: all randomness comes from one seeded `random.Random` held by the session (plus a seeded `numpy` generator if a life form needs one). Iterate entities in a stable order (by `id`). Same seed + params + tick count = same state.

## 4. Save, load, reset

| Action | Behavior |
|---|---|
| **Save** (`S`) | Write current session to the life form's save slot. Show toast "Saved". |
| **Autosave** | On returning to the menu and on quit. |
| **Load / Continue** | Menu card shows Continue when a save exists; restores exact state, tick, speed and camera. |
| **Reset** (`R`) | Restart with the **same seed and params** (replay from the start). Asks for confirmation. |
| **New** (`N`) | Restart with a **new seed**. Asks for confirmation. |

Save file: `<user_data_dir>/saves/<lifeform_id>.json`

```json
{
  "schema_version": 1,
  "app_version": "0.3.0",
  "lifeform_id": "cell",
  "saved_at": "2026-11-07T16:00:00-08:00",
  "seed": 12345,
  "rng_state": "...",
  "params": { "...": "..." },
  "tick": 5400,
  "speed": 1.0,
  "camera": { "position": [0, 8, -12], "rotation": [35, 0, 0] },
  "state": { "...": "lifeform-specific, from get_state()" }
}
```

- Entity objects serialize via `to_dict` / `from_dict`. A cell's `cycle` saves its model time, species values and any per-cell parameter changes, and restores them onto a fresh model instance.
- Writes are atomic (temp file, then rename).
- Unknown `schema_version` or corrupt file: show an error on the card and offer New; never crash.

## 5. Controls and HUD

| Input | Action |
|---|---|
| Right-drag | Orbit camera |
| Middle-drag / `W A S D` | Pan |
| Scroll | Zoom |
| Left click | Select (pick) a cell |
| `Shift` + left click | Life-form action on the dish (e.g. drop nutrients) |
| `F` | Focus camera on selected entity |
| `Space` | Pause / resume |
| `.` | Step one tick (while paused) |
| `+` / `-` | Speed up / down |
| `I` | Toggle biology info panel |
| `S` | Save |
| `R` | Reset (same seed) |
| `N` | New (new seed) |
| `Esc` | Back to menu (autosaves) |

HUD (top-left): life form name, tick, sim time, speed, FPS, plus `stats()`. Paused state is clearly shown.

Info panel (right, toggle `I`): the life form's `biology` notes and `sources`, with simplified or assumed behaviors marked. When a cell is selected: its phase, cell-cycle protein levels (cyclin, active MPF) as a small live chart, energy, size, age, generation, and its organelles with what each does.

## 6. Life forms

### 6.1 MVP: Cell colony (`cell`)

**Layout: 3D on a dish.** Cells are 3D objects resting on a petri-dish floor; they move in 2D across the dish. This keeps the math simple and prevents cells from hiding behind each other. Later life forms (blood, tissue) may use full 3D.

**Biology it teaches:** what a eukaryotic cell is made of, how cells take in nutrients, how the cell cycle is controlled by proteins (cyclin and cdc2 forming MPF), division (mitosis), and death when starved.

**Source models:**

| Process | Model | Source |
|---|---|---|
| Cell cycle | cdc2–cyclin oscillator, 6 variables | Tyson 1991, *Modeling the cell division cycle: cdc2 and cyclin interactions*, PNAS. BioModels `BIOMD0000000005` (bundled SBML file; verify license on download) |
| Nutrient spread | Diffusion (Fick's second law), explicit finite differences | Textbook |
| Nutrient uptake | Michaelis–Menten saturation | Textbook |
| Cell motion | Brownian motion + soft-sphere repulsion | Textbook |

**Model, per tick:**
1. **Environment**: nutrient grid on the dish floor (e.g. 128×128) diffuses and slowly regenerates.
2. **Movement**: Brownian motion (diffusion coefficient: our assumption, chosen for visible motion) plus soft repulsion with neighbors via the spatial hash.
3. **Feeding**: Michaelis–Menten uptake from the grid cells under each cell → energy (rate constants: our assumption, scaled to the dish).
4. **Upkeep**: energy spent proportional to size; surplus energy → growth.
5. **Cell cycle**: each cell owns one Tyson 1991 model instance (libRoadRunner), advanced by `dt × time_scale` each tick (`time_scale`: our assumption, so a cycle takes about a minute on screen).
   - **Nutrient coupling (our assumption):** the cyclin synthesis rate scales with the cell's energy, so starved cells slow down and stop cycling.
   - **Phase display:** *Mitosis* while active MPF is above a threshold; *Interphase* otherwise. The model doesn't represent S phase, so G1/S/G2 sub-labels are shown as approximations, marked in the info panel.
6. **Division**: when active MPF falls back below the threshold after a peak (end of mitosis) and a minimum time has passed since the last division. Two daughters each get half the energy and a copy of the parent's model state, with a small random spread in parameters (our assumption) so they drift out of sync. Animate the split (pinch, then separate).
7. **Death**: energy ≤ 0 → the cell shrinks and fades.

**Fallback if per-cell models are too slow** (decided in M0-C), in order:
1. Step the cell-cycle models at a lower rate (e.g. 10 Hz) with a larger `dt`.
2. Export the model's equations from Tellurium and integrate them ourselves in Python. Same published equations and parameters, validated against Tellurium output.

**Validation tests** (`tests/validation/test_cell_*.py`):
- `test_cycle_reference`: one isolated cell's MPF oscillation period matches a standalone Tellurium run of `BIOMD0000000005` within 1% (with `time_scale` applied).
- `test_doubling_time`: with unlimited nutrients, population doubling time matches the cycle period within 10%.
- `test_starvation_arrest`: with zero nutrients, cells stop dividing.
- `test_diffusion_analytic`: a point source of nutrient spreads as the analytic Gaussian.
- `test_brownian_msd`: an isolated cell's mean squared displacement ≈ 4Dt.

**Rendering:**
- Membrane: translucent sphere (slightly squashed on the dish) with a rim-light/fresnel shader so edges glow.
- Inside, visible through the membrane: nucleus (darker sphere), a few mitochondria (small capsules), ribosome specks.
- Nucleus visibly condenses during mitosis.
- Dish floor shows the nutrient field as a soft heatmap texture.
- Selected cell: outline highlight.

**Params:** initial cells, nutrient regen rate, diffusion rate, upkeep cost, `time_scale`. Each param records its source or "our assumption".

**Stats:** population, births, deaths, mean energy, oldest generation, count in interphase vs mitosis.

### 6.2 Roadmap (one package each, after MVP)

Each row's source model and test are the starting point; refine when the life form is specced.

| id | Life form | Source model | Package | Validation test |
|---|---|---|---|---|
| `mitosis` | Single-cell division close-up | Novak & Tyson 1993, M-phase control (BioModels `BIOMD0000000107`) | `tellurium` | Stage order and timing follow the model's MPF dynamics |
| `bacteria` | Swimming bacteria | Run and tumble (Berg & Brown 1972); optional receptor adaptation (Spiro, Parkinson & Othmer 1997, CellML; may need conversion) | — / optional `tellurium` | Net drift up a food gradient; tumbling drops when moving toward food |
| `immune` | White blood cells vs bacteria | Gradient-following chemotaxis toward a diffusing chemoattractant | — | Chemoattractant spread matches analytic diffusion; neutrophils reach the source |
| `blood` | Blood vessel (full 3D) | Poiseuille flow; hemoglobin oxygen binding via the Hill equation (textbook parameters) | — | Velocity profile matches the parabola; saturation matches the Hill curve |
| `neurons` | Small neural network | Izhikevich 2003 spiking neuron model | `brian2` | Regular-spiking parameters reproduce the published firing pattern |
| `molecules` | Inside a synapse | Particle diffusion and binding | `smoldyn` | Mean squared displacement matches the diffusion coefficient |
| `evolution` | Competing species | Mutation–selection (population genetics) | — | Neutral traits drift without trend; a beneficial mutation spreads |

## 7. Project structure

```
Artificial Life Simulator/
├── SPEC.md                  # this file
├── CLAUDE.md                # working memory + pointer to this spec
├── TASKS.md
├── spikes/                  # M0 throwaway experiments + SPIKE.md report
├── pyproject.toml
├── src/alife/
│   ├── __main__.py          # `uv run python -m alife`
│   ├── app.py               # App, main loop, screen switching
│   ├── screens/
│   │   ├── menu.py
│   │   └── simulation.py
│   ├── ui/                  # HUD, info panel, toasts, confirm dialog
│   ├── camera.py            # orbit/pan/zoom, focus, save/restore
│   ├── shaders/             # membrane rim-light, outline
│   ├── save.py              # save/load, atomic writes
│   ├── session.py           # seed, rng, tick, speed, lifeform instance
│   ├── spatial.py           # uniform-grid spatial hash
│   └── lifeforms/
│       ├── __init__.py      # registry + Lifeform protocol
│       └── cell/
│           ├── __init__.py  # registers the life form
│           ├── sim.py       # Cell, organelles, Environment (no Ursina)
│           ├── cycle.py     # CellCycle: wraps a RoadRunner instance
│           ├── models/
│           │   └── BIOMD0000000005.xml   # bundled Tyson 1991 SBML
│           ├── view.py      # Ursina entities for cells and dish
│           └── biology.py   # info panel notes + sources
└── tests/
    ├── test_save_roundtrip.py
    ├── test_determinism.py
    ├── test_cell_sim.py     # headless, no window
    └── validation/          # §1.1 tests against known results
        └── test_cell_*.py
```

## 8. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| M0 | Tests: Tissue Forge, Ursina, Tellurium | `spikes/SPIKE.md` records findings and decisions (below) |
| M1 | Scaffold | `uv run python -m alife` opens a 3D window with the menu; one placeholder life form appears from the registry |
| M2 | Sim loop, camera, controls | Fixed timestep, pause, step, speed, orbit/pan/zoom, picking, HUD all work on the placeholder |
| M3 | Save / load / reset | Save → quit → Continue restores identical state and camera; Reset replays identically (`test_determinism.py`) |
| M4 | Cell colony | Cells feed, grow, cycle (Tyson model), divide and die; Shift+click nutrients; population stays stable for 10 min at default params; 300 cells at ≥ 60 FPS on the Mac; **all §6.1 validation tests pass** |
| M5 | Info panel | `I` panel shows biology notes and sources with assumptions marked; clicking a cell shows its phase, live cyclin/MPF chart, stats and organelles |
| M6 | Packaging | Build runs on macOS without Python installed (including libRoadRunner) |

### M0 test plan (timebox: about 1 hour each)

**A. Tissue Forge** (own conda env; it has been tested on Python ≤ 3.11):
1. Installs and runs a built-in cell example on Apple Silicon?
2. Can our code step it one tick at a time, headless, from our own loop?
3. Can we read cell positions/radii each tick and set state back (for save/load)?
4. Cell division supported?
5. Speed at 300 and 1,000 cells?

**B. Ursina render test** (main env, Python 3.12):
1. 300 translucent cells, each with a nucleus and organelles inside, on a dish, at ≥ 60 FPS.
2. Rim-light shader on the membrane works on macOS OpenGL 4.1.
3. Clicking a cell picks it.

**C. Tellurium per-cell test** (main env, Python 3.12):
1. `tellurium` / `libroadrunner` install with `uv` on Apple Silicon.
2. Load `BIOMD0000000005`; a standalone run oscillates and matches the published behavior.
3. 300 independent model instances, each advanced one tick: time per tick ≤ 5 ms on the Mac?
4. Save/restore: copy one instance's time and species values onto a fresh instance; both produce identical output afterward.
5. Can a parameter (cyclin synthesis rate) be changed per instance mid-run?

**Decisions to record:**
- Tissue Forge: use it as the physics backend (rendered by our Ursina view) only if A2, A3 and A4 pass **and** running Python 3.11 for the whole app is acceptable. Otherwise write our own physics (§6.1).
- Tellurium: per-cell instances if C3–C5 pass; otherwise the §6.1 fallbacks, in order.
- Mesa: adopt only if its agent model clearly saves work.

## 9. Open questions

- Art style: clean, bright 3D look, or microscope look (dark-field background, glow, soft focus)?
- Should params be editable in-app (sliders), or only defaults in v1?
- Multiple save slots per life form, or one slot each?
