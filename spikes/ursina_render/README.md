# Ursina render spike (M0-B)

> **Controller verification (supersedes the screenshot and "recommended mode"
> sections below).** As handed back, the window drew nothing, because of two bugs:
>
> 1. Ursina always creates `camera.overlay`, a full-window clear quad under
>    `camera.ui` that Ursina forces onto `unlit_shader` (`#version 130`). Under
>    `gl-version 3 2` that shader fails, and the quad draws **opaque black over
>    the whole scene**. The "near-black" captures were real, not a permission
>    problem. Fix in the spike: `camera.overlay.enabled = False`. Product code
>    needs GLSL 150 ports of Ursina's UI shaders (`unlit_shader`, `text_shader`).
>    Bare Panda3D on the same context renders correctly.
> 2. Ursina 8's `color.rgba()` takes 0–1 floats; the spike passed 0–255, so every
>    surface was saturated white. Fixed with `color.rgba32()`.
>
> After both fixes, `naive` and `instanced` render the full scene at 300 cells:
> translucent membranes, a visible rim light, nucleus, mitochondria and ribosomes.
> The built-in `--screenshot` now writes correct pixels too. `merged` renders, but
> it has two problems: the combined interiors come out white (the replacement
> default shader ignores vertex colours), and `Entity.combine()` makes startup
> take ~31 s at 300 cells (~0.1 s per cell, a hitch on every division). For
> captures from outside the app, `screencapture -l <CGWindowID>` works, and so
> does a full-screen capture.

Throwaway spike for SPEC.md §8 test B. Tests Ursina 8.3.0 / Panda3D 1.10.16
on this Mac (Apple M5 Pro, macOS, OpenGL capped at 4.1).

## Run it

```
uv run python spikes/ursina_render/render_spike.py \
    --cells 300 --mode naive \
    --frames 300 --settle 150 \
    --out spikes/ursina_render/out/naive_300.json \
    --screenshot spikes/ursina_render/out/naive_300.png \
    --pick-test
```

`--mode` is one of `naive` / `merged` / `instanced`. `--no-shader` swaps the
rim-light membrane shader for a plain translucent color (useful to A/B the
shader's cost once the controller runs the timed sweep). A GUI window opens
on screen, runs `--settle + --frames` frames, then exits on its own with
code 0 — no keyboard/mouse input needed.

Sanity-check each mode with a short run first:

```
uv run python spikes/ursina_render/render_spike.py --cells 300 --mode naive --frames 30 --settle 10
uv run python spikes/ursina_render/render_spike.py --cells 300 --mode merged --frames 30 --settle 10
uv run python spikes/ursina_render/render_spike.py --cells 300 --mode instanced --frames 30 --settle 10
```

Do not treat the `mean_ms`/`p95_ms`/`max_ms` from any run made during this
spike as a timing result — another spike (Tellurium) was running on this
machine at the same time. The JSON's own `note` field says the same. The
controller runs the real timed sweep alone, later, with this same script.

## What's in the JSON output

`{mode, cells, nodes, mean_ms, p95_ms, max_ms, gl_version, glsl_version,
renderer, window_size, shader, draw_calls_estimated, pick_results,
frames_measured, settle_frames, note}`.

`nodes` / `draw_calls_estimated` are the per-mode node/draw-call counts
described below — computed from the mode's design, not measured via
Panda3D's stats API (no PStats client wired up in this spike).

## The three modes

| mode | per-cell nodes | what moves each frame |
|---|---|---|
| `naive` | 13 (membrane + nucleus + 3 mito + 8 ribosomes, each its own `Entity`) | 13 × N `NodePath.position` writes |
| `merged` | 2 (membrane separate; nucleus+organelles baked into one mesh via `Entity.combine()`) | 2 × N `NodePath.position` writes |
| `instanced` | 4 **total**, not per-cell (one draw call per object kind: membrane / nucleus / mito / ribosome) | 4 texture uploads (`Texture.set_ram_image`) per frame, each carrying N / N / 3N / 8N instance positions, sampled in the vertex shader via `texelFetch(samplerBuffer, gl_InstanceID)` |

All three produce the same scene (seeded, same cell count reproduces the
same layout) so they're visually comparable.

Instanced mode worked within the timebox — this was not a given going in.
Panda3D's own bundled `instancing_shader` uses fixed-size uniform arrays
(256 instances max), which doesn't scale to 2,400 ribosome instances at
N=300. Buffer textures (`Texture.setup_buffer_texture`, sampled via
`texelFetch(samplerBuffer, ...)`) avoid that ceiling and compile/run fine
under the GL 4.1 / GLSL 4.10 context this Mac gives us.

## B2: the rim-light shader, and what it took to get *anything* on screen

`rim = pow(1 - max(dot(N, V), 0), p)`; alpha and brightness both rise
toward the silhouette (`RIM_VERT`/`RIM_FRAG` in render_spike.py). It
compiles and is used by all three modes (with a per-instance variant for
`instanced`, since that mode has no per-cell transform to multiply by —
the per-instance world offset comes from the same buffer-texture lookup
described above and is added after the mesh's own model-matrix transform).

Getting this shader (or any GLSL ≥130 shader) to compile at all on this Mac
needed `loadPrcFileData('', 'gl-version 3 2')` **before** `Ursina()` is
constructed. Without it, Panda3D hands back a legacy OpenGL 2.1 / GLSL 1.20
context (confirmed by querying `base.win.getGsg()`); with it, this Mac
gives GL 4.1 / GLSL 4.10 (matches SPEC's "macOS caps OpenGL at 4.1" note —
that's the cap *once you ask for a modern context*, not the default).

That fix breaks Ursina's own bundled shaders, though: `Entity.default_shader`
(`unlit_with_fog_shader`) and friends are `#version 130`/`140`, and a
`gl-version 3 2` request gets a strict **core** profile that refuses
anything below `#version 150`. Every plain-colored `Entity` (dish, nucleus,
organelles, any `--no-shader` membrane) silently gets no shader at all
unless something replaces `Entity.default_shader` with a `#version 150`
equivalent — done here (`build_simple_unlit_shader`), but this is real
product-code work Plan 2 needs too, not just a spike workaround. One
leftover from this: Ursina's mouse-cursor UI entity hardcodes `unlit_shader`
(bypassing `Entity.default_shader` entirely — see `entity.py`'s
`shader_setter`), so a single harmless `#version 130` compile error still
prints to stderr every run. Did not chase further; it has no effect on the
3D scene.

A second real shader bug, also worth carrying forward: an early version of
the default-shader replacement multiplied by `in vec4 p3d_Color` (matching
Ursina's own built-in), but Ursina's procedural primitives (`'cube'`,
`'sphere'`, `'plane'`, our dish `Circle`) don't supply a per-vertex color
array. The unbound attribute read back as `(0,0,0,1)` on this driver,
silently making every entity solid black regardless of `color=`/
`p3d_ColorScale`. Fix: don't reference `p3d_Color` at all unless the mesh
actually provides it.

### Screenshot capture: not solved cleanly — flagging for Plan 2

"Take a screenshot per mode and note what you see" (brief, transparency
section) could not be made reliable here within the timebox, despite three
different approaches:

1. `base.win.saveScreenshot()` on the onscreen window reliably returns an
   all-zero (R=G=B=A=0) image under the `gl-version 3 2` context (checked
   by reading the PNG back with `PNMImage` and sampling pixels, not just
   looking at it).
2. A throwaway camera cloned onto `render`, with its world position/
   rotation copied from `base.cam`, rendered into an offscreen
   `makeTextureBuffer`: worked (real pixel content) with Ursina's
   default, unmoved camera. Once the scene camera was repositioned via
   `camera.position=`/`camera.rotation=` (as the brief's framing
   requires), the *same* copy produced uniform garbage — confirmed the
   copied pos/hpr matched exactly, yet the image didn't.
3. macOS's own `screencapture` CLI against the window's real screen
   region: also came back solid near-black, independent of the
   `gl-version 3 2` workaround (reproduced with it on and off) — most
   likely this automated/headless environment's calling process has no
   Screen Recording permission, which macOS appears to silently satisfy
   with black pixels for any GPU-composited window content rather than
   erroring.

The current code (`save_screenshot_offscreen`) uses a variant of (2):
reparenting the screenshot camera as a **child of `base.cam`** (zero local
offset) rather than copying world pos/hpr onto a `render`-sibling. That
fixed a minimal 2-entity repro (verified by sampling pixels: got the
actual red/lime scene colors back, not garbage) but the *full* scene
(dish + N cells, custom shaders) still screenshots as uniform white. Ran
out of timebox chasing this further. **Net effect: screenshot output from
this spike should not be trusted as evidence of correct rendering.** All
of B1-B3's *other* checks (shader compiles without error; GL/GLSL
version/renderer detection; picking correctness; per-mode node/draw-call
counts) are independent of screenshotting and did pass — see the report.
If Plan 2 wants automated visual regression shots, this needs real
follow-up (try a newer Panda3D build, try without any gl-version forcing
at all on the *onscreen* buffer while reading back via Panda3D's own GL
calls instead of `screencapture`, or accept a human has to eyeball the
window live).

## B3: picking

Two real bugs found and fixed, beyond the headline screenshot issue:

1. **Collide mask mismatch.** The standard mouse-picking recipe sets the
   ray's from-mask to `GeomNode.getDefaultCollideMask()` (bit 20 — the
   mask *visible geometry* collisions use, e.g. `MeshCollider`). A plain
   `CollisionNode`'s default into-mask is bits 0-19. Those don't overlap,
   so the ray registered zero hits against our `CollisionSphere` proxies
   until the from-mask was changed to `BitMask32.allOn()`.
2. **Plane parented under a rotated node.** The dish's "empty space" probe
   uses an infinite `CollisionPlane`. It was first parented under the
   dish's own visual `Entity`, which has `rotation_x=90` (needed to lay
   the `Circle` mesh flat) — a `CollisionPlane`'s `Plane()` is defined in
   its parent's local space, so that rotation carried straight into the
   collision surface and the "dish" hit landed a few units in front of
   the camera instead of on the floor. Fixed by parenting it under a
   fresh, unrotated `Entity` instead.

Picking uses **dedicated invisible proxy colliders** (`CollisionSphere`,
membrane radius, one per cell), co-located with each cell's membrane every
frame, rather than attaching colliders to whatever the active render mode
happens to use. This was a deliberate design choice, not a shortcut forced
by a bug: `instanced` mode has exactly 4 NodePaths total (one per object
kind) — there is no per-cell node in that mode to hang a collider off of.
Using the same proxy mechanism for all three modes keeps picking
comparable across modes and keeps the picking code mode-independent. Cost:
N extra NodePaths with a `CollisionSphere` and no geometry; they're never
traversed except during an actual `pick_at_world_point()` call, so they add
no per-frame render or collision-traversal cost by themselves.

One framing note: the brief's cells spawn uniformly inside
`CELL_SPAWN_RADIUS = 9.5`, but this camera (position `(0,8,-12)`,
`rotation_x 35`, Ursina's default ~40°×23° FOV) does not fit that whole
disc in frame — cells near the spawn disc's edge legitimately project
outside the view frustum. `run_pick_test` filters the 5 sampled cells (and
the empty-dish probe point) to ones confirmed in-view first, so the test
measures picking correctness rather than camera-framing coverage. With
that filter, all 6 probes (5 cells + empty dish) pass.

## Recommendation for Plan 2

- **Start from `merged`, not `naive`,** for the real cell view. At 300
  cells `naive` is 3,900 moving nodes; `merged` is 600. `merged` keeps a
  real per-cell NodePath (needed for individual cell state like phase
  color, selection highlight, per-cell death animation — all things §6.1
  wants) while cutting internals to one draw call per cell. `instanced`'s
  4-draws-total is the cheapest by far, but it currently has no
  per-cell hook for things like "this cell is selected, fade its
  highlight" or per-cell color variation (mitosis-state tinting, etc.)
  without extra per-instance attribute channels (color, scale) beyond the
  position buffer built here — doable, but more shader work than this
  timebox allowed. Worth revisiting once §6.1's per-cell visual states are
  specced concretely; the buffer-texture approach proven here scales to
  that if/when it's needed.
- **Carry the three fixes above into product code verbatim:** `gl-version
  3 2` before `Ursina()`; a GLSL ≥150 replacement for anything used from
  Ursina's bundled shader set (not just the default); `BitMask32.allOn()`
  for picking rays, not `GeomNode.getDefaultCollideMask()`.
- **Do not assume a screenshot proves the shader works.** Verify shader
  correctness via `shader.compiled` + stderr (no compile error) + picking
  math, the way this spike ultimately had to; screenshot-based visual
  checks need more investigation on this machine before they're trustworthy.
