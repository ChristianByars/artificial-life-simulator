# Artificial Life Simulator

A desktop app that simulates life forms, starting with a cell colony, as a way to learn biology (cells, human and animal bodies).

## Spec
**`SPEC.md` is the source of truth for implementation.** It is a local file, kept out of git. Read it before writing code. If code needs to deviate from the spec, flag it rather than silently diverging.

Stack: Python 3.12+, uv, Ursina (on Panda3D) for 3D. Cells are plain Python objects; sim code never imports Ursina. Run with `uv run python -m alife`.
M0 is done (SPEC.md §8): findings and the approved decisions are in `spikes/SPIKE.md`.
Biology rule (SPEC §1.1): proven source models only; label our own assumptions; every life form ships a validation test.

## Terms
| Term | Meaning |
|------|---------|
| ALS | Artificial Life Simulator (this project) |
| A-Life | Artificial life: simulating lifelike systems |
