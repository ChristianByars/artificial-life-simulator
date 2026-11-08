# SPIKE (M0): throwaway experiment, not product code.
"""Ursina / Panda3D render spike (SPEC.md §8, test B).

Answers B1-B3 on this Mac (Apple Silicon, macOS, OpenGL capped at 4.1):
  B1. N translucent cells (membrane + nucleus + organelles) on a dish,
      three render strategies (naive / merged / instanced).
  B2. A rim-light/fresnel membrane shader, hand-written GLSL, checked for
      compile errors and for which GL/GLSL context Panda3D actually gets
      on macOS.
  B3. Picking: project a cell's world position to screen space and run the
      real Panda3D collision-ray picking path at that point.

Usage:
    uv run python spikes/ursina_render/render_spike.py --cells 300 --mode naive \
        --frames 300 --settle 150 --out spikes/ursina_render/out/naive_300.json \
        --screenshot spikes/ursina_render/out/naive_300.png --pick-test

Do not report numbers produced by a sanity run (--frames 30 --settle 10) as a
timing result -- another spike (Tellurium) runs on this machine at the same
time, so no run here is a clean measurement. The controller runs the timed
sweep alone, later, with this same script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import time
from pathlib import Path

from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    GeomEnums,
    OmniBoundingVolume,
    Point2,
    Point3,
    Texture,
    TransparencyAttrib,
    loadPrcFileData,
)

# --------------------------------------------------------------------------
# PRC settings: MUST be set before Ursina() is constructed. Two Ursina/macOS
# quirks found by reading ursina/window.py (ursina 8.3.0):
#
# 1. Window size: Ursina.__init__ -> window._ready() always calls
#    loadPrcFileData('', 'win-size ...') itself, computed from the `size`
#    kwarg (default: monitor size / 1.25). Any win-size we set beforehand is
#    overridden by that later call. -> window size MUST be passed as
#    Ursina(size=...), not set via our own loadPrcFileData.
#
# 2. vsync: Ursina.__init__ -> window.apply_settings() unconditionally runs
#    `self.vsync = True` right after the window is created. The vsync setter
#    has `if not 'base' in sys.modules:` ("before window opened" branch) --
#    that checks for a literal *module* named "base", which never exists, so
#    the condition is always true and it always re-issues
#    loadPrcFileData('sync-video True'), regardless of the Ursina(vsync=...)
#    kwarg we pass in (which is itself never read by _ready() -- dead
#    parameter). Since this happens AFTER the GSG/window already exists,
#    Panda3D's swap-interval was already fixed at window-creation time and
#    this late prc write is a no-op on the live window. Net effect: the only
#    reliable way to get vsync off is to call loadPrcFileData('sync-video 0')
#    BEFORE constructing Ursina(), exactly as the brief says. Passing
#    vsync=False to Ursina() does nothing.
# --------------------------------------------------------------------------
loadPrcFileData("", "sync-video 0")
loadPrcFileData("", "gl-version 3 2")  # see B2 notes in the report/README

from ursina import (  # noqa: E402
    Entity,
    Shader,
    Ursina,
    Vec2,
    Vec3,
    camera,
    color,
    time as ursina_time,
)

SEED = 20261107
DISH_RADIUS = 10.0
CELL_SPAWN_RADIUS = 9.5
MEMBRANE_RADIUS = 0.35
MEMBRANE_Y_SCALE = 0.8
MEMBRANE_ALPHA = 0.35
NUCLEUS_RADIUS = 0.12
CELL_REST_Y = (MEMBRANE_RADIUS * MEMBRANE_Y_SCALE) / 2  # sit the squashed sphere on the dish
MOTION_AMPLITUDE = 0.12
MOTION_FREQ = 0.6

rng = random.Random(SEED)


# --------------------------------------------------------------------------
# Rim-light shader (B2): rim = pow(1 - max(dot(N, V), 0), p); both alpha and
# brightness rise toward the silhouette.
# --------------------------------------------------------------------------
RIM_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
out vec3 world_normal;
out vec3 world_position;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    world_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
}
"""

RIM_FRAG = """#version 150
uniform vec4 p3d_ColorScale;
uniform vec3 camera_world_position;
uniform vec4 base_color;
uniform vec4 rim_color;
uniform float rim_power;
uniform float base_alpha;
in vec3 world_normal;
in vec3 world_position;
out vec4 fragColor;

void main() {
    vec3 N = normalize(world_normal);
    vec3 V = normalize(camera_world_position - world_position);
    float rim = pow(1.0 - max(dot(N, V), 0.0), rim_power);
    vec3 rgb = mix(base_color.rgb, rim_color.rgb, rim) * p3d_ColorScale.rgb;
    float a = clamp(base_alpha + rim * (1.0 - base_alpha), 0.0, 1.0) * p3d_ColorScale.a;
    fragColor = vec4(rgb, a);
}
"""

# Instanced variants: per-instance world position comes from a buffer
# texture (texelFetch(samplerBuffer, gl_InstanceID)), one float3 per
# instance, re-uploaded every frame instead of moving N separate NodePaths.
RIM_VERT_INSTANCED = """#version 150
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform samplerBuffer instance_positions;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
out vec3 world_normal;
out vec3 world_position;

void main() {
    vec3 offset = texelFetch(instance_positions, gl_InstanceID).rgb;
    vec3 shaped = (p3d_ModelMatrix * p3d_Vertex).xyz;
    world_position = shaped + offset;
    gl_Position = p3d_ProjectionMatrix * p3d_ViewMatrix * vec4(world_position, 1.0);
    world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
}
"""

UNLIT_INSTANCED_VERT = """#version 150
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ViewMatrix;
uniform mat4 p3d_ProjectionMatrix;
uniform samplerBuffer instance_positions;
in vec4 p3d_Vertex;

void main() {
    vec3 offset = texelFetch(instance_positions, gl_InstanceID).rgb;
    vec3 shaped = (p3d_ModelMatrix * p3d_Vertex).xyz;
    gl_Position = p3d_ProjectionMatrix * p3d_ViewMatrix * vec4(shaped + offset, 1.0);
}
"""

UNLIT_INSTANCED_FRAG = """#version 150
uniform vec4 p3d_ColorScale;
out vec4 fragColor;

void main() {
    fragColor = p3d_ColorScale;
}
"""


def build_rim_shader() -> Shader:
    return Shader(
        name="rim_light_shader",
        language=Shader.GLSL,
        vertex=RIM_VERT,
        fragment=RIM_FRAG,
        default_input={
            "base_color": color.rgba32(40, 90, 130, 255),
            "rim_color": color.rgba32(190, 230, 255, 255),
            "rim_power": 2.5,
            "base_alpha": MEMBRANE_ALPHA,
            "camera_world_position": Vec3(0, 8, -12),
        },
    )


def build_rim_shader_instanced() -> Shader:
    return Shader(
        name="rim_light_shader_instanced",
        language=Shader.GLSL,
        vertex=RIM_VERT_INSTANCED,
        fragment=RIM_FRAG,
        default_input={
            "base_color": color.rgba32(40, 90, 130, 255),
            "rim_color": color.rgba32(190, 230, 255, 255),
            "rim_power": 2.5,
            "base_alpha": MEMBRANE_ALPHA,
            "camera_world_position": Vec3(0, 8, -12),
        },
    )


# Ursina 8.3.0 ships its Entity.default_shader (unlit_with_fog_shader) and
# several other built-ins (unlit_shader, text_shader, ...) as GLSL #version
# 130/140. Under the gl-version 3 2 core context we force on macOS (needed
# to get past GL 2.1 -- see module docstring / B2 report), a strict core
# profile REJECTS any #version below 150: every plain-colored Entity (no
# explicit shader) fails to compile and renders with no shader bound at all.
# This is a real B2 finding, not a spike-only workaround: Plan 2's product
# code needs the same #version-150-or-higher replacement for any shader it
# relies on from Ursina's bundled set. We patch Entity.default_shader here
# so the rest of the scene (dish, nucleus, organelles, and any --no-shader
# membrane) actually renders instead of silently vanishing.
# NOTE: deliberately does NOT read p3d_Color. A second real bug found here:
# an earlier version multiplied by `in vec4 p3d_Color` (per-vertex color),
# matching Ursina's own unlit_with_fog_shader. Ursina's procedural
# primitives ('cube'/'sphere'/'plane'/our Circle) don't supply a per-vertex
# color array, so that attribute is never bound; an unbound GLSL "in" reads
# as (0,0,0,1) on this driver, silently multiplying every entity's color to
# black regardless of its actual `color=`/p3d_ColorScale -- confirmed via a
# minimal red-cube/lime-ground repro (SOLID BLACK result) that only started
# showing real color once this was dropped to `fragColor = p3d_ColorScale`
# alone. Anything ported from Ursina's bundled shaders needs the same check.
SIMPLE_UNLIT_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""

SIMPLE_UNLIT_FRAG = """#version 150
uniform vec4 p3d_ColorScale;
out vec4 fragColor;

void main() {
    fragColor = p3d_ColorScale;
}
"""


def build_simple_unlit_shader() -> Shader:
    return Shader(name="simple_unlit_shader_150", language=Shader.GLSL, vertex=SIMPLE_UNLIT_VERT, fragment=SIMPLE_UNLIT_FRAG)


def build_unlit_instanced_shader() -> Shader:
    return Shader(
        name="unlit_instanced_shader",
        language=Shader.GLSL,
        vertex=UNLIT_INSTANCED_VERT,
        fragment=UNLIT_INSTANCED_FRAG,
    )


# --------------------------------------------------------------------------
# Cell layout: one local-offset list per organelle kind, shared by all
# three render modes so they're visually equivalent.
# --------------------------------------------------------------------------
def make_cell_layout(cell_rng: random.Random) -> dict:
    """Local offsets (relative to cell center) for one cell's organelles."""
    nucleus_offset = Vec3(0, 0.03, 0)
    mito_offsets = []
    for _ in range(3):
        ang = cell_rng.uniform(0, 6.28318)
        r = cell_rng.uniform(0.08, 0.2)
        mito_offsets.append(Vec3(r * math.cos(ang), cell_rng.uniform(-0.05, 0.08), r * math.sin(ang)))
    ribo_offsets = []
    for _ in range(8):
        # random point inside the membrane radius, biased toward the shell
        ang1 = cell_rng.uniform(0, 6.28318)
        ang2 = cell_rng.uniform(0, 3.14159)
        r = cell_rng.uniform(0.15, 0.3)
        ribo_offsets.append(
            Vec3(
                r * math.sin(ang2) * math.cos(ang1),
                r * math.cos(ang2) * 0.6,
                r * math.sin(ang2) * math.sin(ang1),
            )
        )
    return {"nucleus": nucleus_offset, "mito": mito_offsets, "ribo": ribo_offsets}


def make_cells(n: int) -> list[dict]:
    """n cells: base (x, z) inside CELL_SPAWN_RADIUS, a motion phase, and a
    fixed organelle layout. Seeded, so re-running with the same n reproduces
    the same scene."""
    cells = []
    for i in range(n):
        while True:
            x = rng.uniform(-CELL_SPAWN_RADIUS, CELL_SPAWN_RADIUS)
            z = rng.uniform(-CELL_SPAWN_RADIUS, CELL_SPAWN_RADIUS)
            if x * x + z * z <= CELL_SPAWN_RADIUS * CELL_SPAWN_RADIUS:
                break
        phase = rng.uniform(0, 6.28318)
        layout = make_cell_layout(rng)
        cells.append({"id": i, "base_x": x, "base_z": z, "phase": phase, "layout": layout})
    return cells


def cell_center(cell: dict, sim_time: float) -> Vec3:
    x = cell["base_x"] + MOTION_AMPLITUDE * math.sin(sim_time * MOTION_FREQ + cell["phase"])
    z = cell["base_z"] + MOTION_AMPLITUDE * math.cos(sim_time * MOTION_FREQ + cell["phase"])
    return Vec3(x, CELL_REST_Y, z)


# --------------------------------------------------------------------------
# Scene: dish + camera. Shared by all modes.
# --------------------------------------------------------------------------
def build_dish() -> Entity:
    from ursina.models.procedural.circle import Circle

    dish = Entity(
        model=Circle(48, radius=DISH_RADIUS, mode="ngon"),
        rotation_x=90,
        color=color.rgba32(60, 60, 70, 255),
        collider=None,
    )
    # A flat collision plane for the "empty dish" pick probe (B3). Approximate:
    # real geometry is a radius-10 disc, but every probe point we send here is
    # inside that disc, so an infinite plane at y=CELL_REST_Y is an adequate
    # stand-in and keeps the picking code uniform across all three modes.
    #
    # IMPORTANT: this plane is defined in world-space terms (normal +Y, at
    # y=CELL_REST_Y) and must NOT be parented under `dish` itself -- `dish`
    # has rotation_x=90 (needed to lay the Circle mesh flat), and a
    # CollisionPlane's Plane() is in its parent's local space, so parenting
    # it there would rotate the plane's normal/position right along with
    # the mesh and put the collision surface somewhere else entirely (this
    # was an actual bug here: the "dish" pick hit was landing a few units in
    # front of the camera instead of on the floor). Attach it to a fresh,
    # unrotated Entity instead.
    collider_root = Entity()
    from panda3d.core import CollisionPlane, Plane, Vec3 as PVec3

    plane_node = CollisionNode("dish_collider")
    plane_node.addSolid(CollisionPlane(Plane(PVec3(0, 1, 0), PVec3(0, CELL_REST_Y, 0))))
    plane_node.setTag("pick_id", "dish")
    collider_root.attach_new_node(plane_node)
    return dish


def setup_camera() -> None:
    camera.position = Vec3(0, 8, -12)
    camera.rotation = Vec3(35, 0, 0)


# --------------------------------------------------------------------------
# Mode: naive (13 Entities per cell)
# --------------------------------------------------------------------------
def build_naive(cells: list[dict], shader, no_shader: bool) -> dict:
    membranes = []
    internals = []  # list of 12 entities per cell, flat
    for cell in cells:
        c = cell_center(cell, 0.0)
        membrane = Entity(model="sphere", position=c, scale=(MEMBRANE_RADIUS * 2, MEMBRANE_RADIUS * 2 * MEMBRANE_Y_SCALE, MEMBRANE_RADIUS * 2))
        if no_shader or shader is None:
            membrane.color = color.rgba32(60, 120, 170, int(255 * MEMBRANE_ALPHA))
        else:
            membrane.shader = shader
        membrane.setTransparency(TransparencyAttrib.M_alpha)
        membrane.setBin("transparent", 0)
        membrane.setDepthWrite(False)
        membranes.append(membrane)

        layout = cell["layout"]
        cell_internals = []
        nucleus = Entity(model="sphere", position=c + layout["nucleus"], scale=NUCLEUS_RADIUS * 2, color=color.rgba32(20, 20, 40, 255))
        cell_internals.append(nucleus)
        for off in layout["mito"]:
            mito = Entity(model="sphere", position=c + off, scale=(0.12, 0.05, 0.05), color=color.rgba32(200, 90, 40, 255))
            cell_internals.append(mito)
        for off in layout["ribo"]:
            ribo = Entity(model="sphere", position=c + off, scale=0.025, color=color.rgba32(230, 200, 60, 255))
            cell_internals.append(ribo)
        internals.append(cell_internals)

    def update_fn(sim_time: float) -> None:
        for cell, membrane, cell_internals in zip(cells, membranes, internals):
            c = cell_center(cell, sim_time)
            membrane.position = c
            layout = cell["layout"]
            cell_internals[0].position = c + layout["nucleus"]
            idx = 1
            for off in layout["mito"]:
                cell_internals[idx].position = c + off
                idx += 1
            for off in layout["ribo"]:
                cell_internals[idx].position = c + off
                idx += 1

    nodes = len(cells) * 13
    return {"membranes": membranes, "update": update_fn, "nodes": nodes}


# --------------------------------------------------------------------------
# Mode: merged (membrane Entity + 1 flattened "internals" Entity per cell)
# --------------------------------------------------------------------------
def build_merged(cells: list[dict], shader, no_shader: bool) -> dict:
    membranes = []
    internals_nodes = []
    for cell in cells:
        c = cell_center(cell, 0.0)
        membrane = Entity(model="sphere", position=c, scale=(MEMBRANE_RADIUS * 2, MEMBRANE_RADIUS * 2 * MEMBRANE_Y_SCALE, MEMBRANE_RADIUS * 2))
        if no_shader or shader is None:
            membrane.color = color.rgba32(60, 120, 170, int(255 * MEMBRANE_ALPHA))
        else:
            membrane.shader = shader
        membrane.setTransparency(TransparencyAttrib.M_alpha)
        membrane.setBin("transparent", 0)
        membrane.setDepthWrite(False)
        membranes.append(membrane)

        layout = cell["layout"]
        internals_parent = Entity(position=c)
        Entity(parent=internals_parent, model="sphere", position=layout["nucleus"], scale=NUCLEUS_RADIUS * 2, color=color.rgba32(20, 20, 40, 255))
        for off in layout["mito"]:
            Entity(parent=internals_parent, model="sphere", position=off, scale=(0.12, 0.05, 0.05), color=color.rgba32(200, 90, 40, 255))
        for off in layout["ribo"]:
            Entity(parent=internals_parent, model="sphere", position=off, scale=0.025, color=color.rgba32(230, 200, 60, 255))
        internals_parent.combine(auto_destroy=True)
        internals_nodes.append(internals_parent)

    def update_fn(sim_time: float) -> None:
        for cell, membrane, internals_parent in zip(cells, membranes, internals_nodes):
            c = cell_center(cell, sim_time)
            membrane.position = c
            internals_parent.position = c

    nodes = len(cells) * 2
    return {"membranes": membranes, "update": update_fn, "nodes": nodes}


# --------------------------------------------------------------------------
# Mode: instanced (4 draw calls total: membrane / nucleus / mito / ribosome)
# --------------------------------------------------------------------------
def make_offset_buffer(n: int) -> Texture:
    tex = Texture(f"offsets_{n}_{id(n)}")
    tex.setup_buffer_texture(max(n, 1), Texture.T_float, Texture.F_rgb32, GeomEnums.UH_dynamic)
    return tex


def write_offsets(tex: Texture, positions: list[Vec3]) -> None:
    data = b"".join(struct.pack("<3f", p.x, p.y, p.z) for p in positions)
    tex.set_ram_image(data)


def make_instanced_entity(model_name: str, scale, shader, instance_count: int) -> tuple[Entity, Texture]:
    e = Entity(model=model_name, scale=scale, position=Vec3(0, 0, 0))
    e.shader = shader
    tex = make_offset_buffer(instance_count)
    e.set_shader_input("instance_positions", tex)
    e.set_instance_count(instance_count)
    e.node().setBounds(OmniBoundingVolume())
    e.node().setFinal(True)
    return e, tex


def build_instanced(cells: list[dict], shader, no_shader: bool) -> dict:
    n = len(cells)
    membrane_shader = (build_unlit_instanced_shader() if no_shader else build_rim_shader_instanced())

    membrane, membrane_tex = make_instanced_entity(
        "sphere", (MEMBRANE_RADIUS * 2, MEMBRANE_RADIUS * 2 * MEMBRANE_Y_SCALE, MEMBRANE_RADIUS * 2), membrane_shader, n
    )
    membrane.setTransparency(TransparencyAttrib.M_alpha)
    membrane.setBin("transparent", 0)
    membrane.setDepthWrite(False)
    if no_shader:
        membrane.set_shader_input("p3d_ColorScale", (60 / 255, 120 / 255, 170 / 255, MEMBRANE_ALPHA))

    nucleus_shader = build_unlit_instanced_shader()
    nucleus, nucleus_tex = make_instanced_entity("sphere", NUCLEUS_RADIUS * 2, nucleus_shader, n)
    nucleus.color = color.rgba32(20, 20, 40, 255)

    mito, mito_tex = make_instanced_entity("sphere", (0.12, 0.05, 0.05), build_unlit_instanced_shader(), n * 3)
    mito.color = color.rgba32(200, 90, 40, 255)

    ribo, ribo_tex = make_instanced_entity("sphere", 0.025, build_unlit_instanced_shader(), n * 8)
    ribo.color = color.rgba32(230, 200, 60, 255)

    def positions_for(sim_time: float):
        membrane_pos = []
        nucleus_pos = []
        mito_pos = []
        ribo_pos = []
        for cell in cells:
            c = cell_center(cell, sim_time)
            layout = cell["layout"]
            membrane_pos.append(c)
            nucleus_pos.append(c + layout["nucleus"])
            for off in layout["mito"]:
                mito_pos.append(c + off)
            for off in layout["ribo"]:
                ribo_pos.append(c + off)
        return membrane_pos, nucleus_pos, mito_pos, ribo_pos

    def update_fn(sim_time: float) -> None:
        mp, np_, mip, rp = positions_for(sim_time)
        write_offsets(membrane_tex, mp)
        write_offsets(nucleus_tex, np_)
        write_offsets(mito_tex, mip)
        write_offsets(ribo_tex, rp)

    # seed buffers with the initial (t=0) layout before the first frame renders
    update_fn(0.0)

    nodes = 4  # draw calls; the N/3N/8N per-instance positions move via texture upload, not NodePath transforms
    return {
        "membranes": [membrane],
        "update": update_fn,
        "nodes": nodes,
        "membrane_positions_fn": lambda sim_time: positions_for(sim_time)[0],
    }


# --------------------------------------------------------------------------
# Picking (B3): dedicated invisible sphere-collider proxies, one per cell,
# co-located with the membrane. Needed uniformly across modes because the
# instanced mode has no per-cell NodePath to hang a collider off of.
# --------------------------------------------------------------------------
def build_pick_proxies(cells: list[dict]):
    """Invisible per-cell sphere colliders, co-located with the membrane.

    Used uniformly across all three render modes for the B3 picking test,
    because the instanced mode has no per-cell NodePath to hang a collider
    off of (it's 4 draw calls total, not N). Cost: N extra NodePaths with a
    CollisionSphere solid and no geometry; they never get traversed except
    during an explicit pick_at_world_point() call, so they add no per-frame
    render or traversal cost.
    """
    from panda3d.core import CollisionSphere

    proxy_entities = []
    for cell in cells:
        pe = Entity(position=cell_center(cell, 0.0), model=None)
        cn = CollisionNode(f"pick_proxy_{cell['id']}")
        cn.addSolid(CollisionSphere(0, 0, 0, MEMBRANE_RADIUS))
        cn.setTag("pick_id", f"cell_{cell['id']}")
        pe.attach_new_node(cn)
        proxy_entities.append(pe)
    return proxy_entities


DEBUG_PICK = False


def setup_picking():
    picker_ray = CollisionRay()
    picker_node = CollisionNode("pick_ray")
    picker_node.addSolid(picker_ray)
    # NOTE: NOT GeomNode.getDefaultCollideMask() (bit 20) -- that's the mask
    # visible-geometry collisions use (e.g. MeshCollider). A plain
    # CollisionNode's default into-collide-mask is bits 0-19 (all_on() for
    # the low bits), which does not overlap bit 20 at all, so the ray would
    # silently register zero hits against our CollisionSphere proxies.
    # BitMask32.allOn() matches any into-mask.
    picker_node.setFromCollideMask(BitMask32.allOn())
    picker_node.setIntoCollideMask(0)
    picker_np = base.camera.attach_new_node(picker_node)  # noqa: F821  (base is a Panda3D/Ursina builtin)
    traverser = CollisionTraverser("pick_traverser")
    handler = CollisionHandlerQueue()
    traverser.addCollider(picker_np, handler)

    def pick_at_world_point(world_pos: Vec3):
        cam_space = base.cam.getRelativePoint(base.render, Point3(world_pos.x, world_pos.y, world_pos.z))  # noqa: F821
        p2 = Point2()
        in_view = base.camLens.project(cam_space, p2)  # noqa: F821
        if not in_view:
            return None, None, False
        picker_ray.setFromLens(base.camNode, p2.x, p2.y)  # noqa: F821
        traverser.traverse(base.render)  # noqa: F821
        if handler.getNumEntries() == 0:
            return None, (p2.x, p2.y), True
        handler.sortEntries()
        if DEBUG_PICK:
            for i in range(handler.getNumEntries()):
                e = handler.getEntry(i)
                print("  DEBUG entry", i, e.getIntoNode().getTag("pick_id"), e.getSurfacePoint(base.render))  # noqa: F821
        entry = handler.getEntry(0)
        tag = entry.getIntoNode().getTag("pick_id")
        return (tag if tag else None), (p2.x, p2.y), True

    return pick_at_world_point


def save_screenshot_offscreen(path: Path) -> None:
    """Screenshot capture on this Mac took three tries to get right -- worth
    recording since Plan 2 will hit the same thing:

    1. `base.win.saveScreenshot()` on the onscreen window: under the
       gl-version 3 2 core context, this reliably returns an all-zero
       (R=G=B=A=0) image (confirmed via PNMImage pixel sampling).
    2. A throwaway camera cloned onto `render` (world pos/hpr copied from
       base.cam) rendering into an offscreen `makeTextureBuffer`: worked
       perfectly with Ursina's default, unmoved camera, but once the scene
       camera was repositioned with `camera.position=`/`camera.rotation=`
       (as this spike's brief requires), the SAME copy produced a uniform
       garbage color (confirmed numerically: offcam's world pos/hpr
       matched base.cam's exactly, yet the pixels didn't match a
       known-good render).
    3. macOS's own `screencapture` CLI against the window's screen region:
       also came back uniformly near-black -- turns out this environment
       has no Screen Recording permission for the calling process, which
       macOS silently honors by handing back black for any
       GPU-accelerated window content (confirmed: this happened even with
       NO gl-version forcing at all, i.e. it's independent of our core
       -profile workaround).

    The fix: reparent the screenshot camera as a direct CHILD of base.cam
    (zero local offset) instead of copying its world position/rotation
    onto a sibling under render. Verified by pixel-sampling the output:
    with a plain red box + lime ground test scene this reliably reproduced
    the actual scene colors; the render/pos-copy version never did, for
    reasons not fully run down (something about how the lens or camera
    -relative shader inputs resolve across sibling vs. child cameras,
    most likely) -- noting this as an open question for Plan 2 rather than
    a fully explained one.
    """
    tex = Texture()
    w, h = base.win.getXSize(), base.win.getYSize()  # noqa: F821
    buff = base.win.makeTextureBuffer("screenshot_buffer", w, h, tex, to_ram=True)  # noqa: F821
    offcam = base.makeCamera(buff)  # noqa: F821
    offcam.reparentTo(base.cam)  # noqa: F821  (child of the real camera, NOT a render-sibling with copied pos/hpr)
    offcam.setPosHpr(0, 0, 0, 0, 0, 0)
    offcam.node().setLens(base.camLens)  # noqa: F821
    for _ in range(3):
        base.graphicsEngine.renderFrame()  # noqa: F821
    buff.saveScreenshot(str(path))
    offcam.removeNode()
    base.graphicsEngine.removeWindow(buff)  # noqa: F821


def find_empty_probe_point(cells: list[dict], pick_at_world_point) -> Vec3:
    # Candidates close to dish center, where the camera (FOV ~40x23 deg at
    # this framing) actually has the dish in view -- the brief's radius-9.5
    # spawn disc is wider than the frustum at this distance, so a candidate
    # near the dish edge would just fail "in_view", which is a framing
    # artifact, not a picking failure.
    candidates = [Vec3(x, CELL_REST_Y, z) for x in (-3.0, 3.0, 0.0) for z in (-2.0, 2.0, 4.0)]
    centers = [cell_center(c, 0.0) for c in cells]
    best, best_d = None, -1.0
    for cand in candidates:
        _, _, in_view = pick_at_world_point(cand)
        if not in_view:
            continue
        d = min(((cand.x - c.x) ** 2 + (cand.z - c.z) ** 2) ** 0.5 for c in centers) if centers else 999
        if d > best_d:
            best_d, best = d, cand
    return best if best is not None else candidates[0]


def run_pick_test(cells: list[dict], sim_time: float) -> list[dict]:
    pick_at_world_point = setup_picking()
    results = []

    # Only cells actually on screen can be a meaningful picking probe; the
    # radius-9.5 spawn disc is wider than this camera's frustum at this
    # framing (see README), so filter to in-view cells first.
    in_view_ids = []
    for cell in cells:
        c = cell_center(cell, sim_time)
        _, _, in_view = pick_at_world_point(c)
        if in_view:
            in_view_ids.append(cell["id"])
    sample_pool = in_view_ids if in_view_ids else list(range(len(cells)))
    sample_ids = rng.sample(sample_pool, k=min(5, len(sample_pool)))

    for cid in sample_ids:
        c = cell_center(cells[cid], sim_time)
        got, screen, in_view = pick_at_world_point(c)
        expected = f"cell_{cid}"
        ok = got == expected
        results.append(
            {"probe": "cell", "cell_id": cid, "expected": expected, "got": got, "screen": screen, "in_view": in_view, "pass": ok}
        )
        print(f"PICK cell_{cid} expected={expected} got={got} in_view={in_view} -> {'PASS' if ok else 'FAIL'}")

    empty_point = find_empty_probe_point(cells, pick_at_world_point)
    got, screen, in_view = pick_at_world_point(empty_point)
    ok = got in (None, "dish")
    results.append(
        {"probe": "empty_dish", "cell_id": None, "expected": "dish_or_none", "got": got, "screen": screen, "in_view": in_view, "pass": ok}
    )
    print(f"PICK empty_dish expected=dish_or_None got={got} in_view={in_view} -> {'PASS' if ok else 'FAIL'}")
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Ursina render spike (M0-B)")
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--mode", choices=["naive", "merged", "instanced"], required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--settle", type=int, default=150)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--no-shader", action="store_true")
    parser.add_argument("--pick-test", action="store_true")
    args = parser.parse_args()

    app = Ursina(
        title="ursina_render_spike",
        size=(1600, 1000),
        fullscreen=False,
        borderless=False,
        development_mode=False,
        editor_ui_enabled=False,
        show_ursina_splash=False,
    )

    global base  # noqa: PLW0603  (Ursina/Panda3D expose this as a builtin; make it importable here)
    import builtins

    base = builtins.base

    Entity.default_shader = build_simple_unlit_shader()  # see note above SIMPLE_UNLIT_VERT

    # Controller fix (M0 verification): Ursina always creates camera.overlay, a
    # full-window clear quad under camera.ui that is forced onto unlit_shader
    # (#version 130). The core profile rejects that shader and the quad draws
    # opaque black over the whole scene: the cause of the black window. Hiding it
    # is enough for the spike; product code needs GLSL 150 ports of Ursina's UI
    # shaders (unlit_shader, text_shader) instead.
    camera.overlay.enabled = False

    gsg = base.win.getGsg()
    gl_info = {
        "gl_version": f"{gsg.getDriverVersionMajor()}.{gsg.getDriverVersionMinor()}",
        "glsl_version": f"{gsg.getDriverShaderVersionMajor()}.{gsg.getDriverShaderVersionMinor()}",
        "renderer": gsg.getDriverRenderer(),
        "vendor": gsg.getDriverVendor(),
        "supports_glsl": bool(gsg.getSupportsGlsl()),
    }
    print("GL/GLSL info:", gl_info)

    build_dish()
    setup_camera()

    cells = make_cells(args.cells)

    shader = None if args.no_shader else build_rim_shader()
    builders = {"naive": build_naive, "merged": build_merged, "instanced": build_instanced}
    built = builders[args.mode](cells, shader, args.no_shader)

    pick_proxies = None
    if args.pick_test:
        pick_proxies = build_pick_proxies(cells)

    state = {"phase": "settle", "count": 0, "sim_time": 0.0, "prev": None, "pick_results": None, "done": False}
    frame_times_ms: list[float] = []

    def update() -> None:
        state["sim_time"] += ursina_time.dt
        built["update"](state["sim_time"])
        if pick_proxies is not None:
            for cell, pe in zip(cells, pick_proxies):
                pe.position = cell_center(cell, state["sim_time"])

        if state["phase"] == "settle":
            state["count"] += 1
            if state["count"] >= args.settle:
                state["phase"] = "measure"
                state["count"] = 0
                state["prev"] = time.perf_counter()
            return

        if state["phase"] == "measure":
            now = time.perf_counter()
            if state["prev"] is not None:
                frame_times_ms.append((now - state["prev"]) * 1000.0)
            state["prev"] = now
            state["count"] += 1
            if state["count"] >= args.frames:
                state["phase"] = "finish"
            return

        if state["phase"] == "finish":
            if args.pick_test:
                state["pick_results"] = run_pick_test(cells, state["sim_time"])
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                save_screenshot_offscreen(args.screenshot)
                print(f"screenshot written: {args.screenshot}")
            finish(gl_info, built, args, frame_times_ms, state)
            state["done"] = True
            return

    # Ursina's task manager calls `__main__.update()` every frame (it checks
    # `hasattr(__main__, 'update')` on the module itself -- builtins do NOT
    # satisfy that, only an attribute of the __main__ module does).
    sys.modules["__main__"].update = update

    app.run()
    return 0


def finish(gl_info: dict, built: dict, args, frame_times_ms: list[float], state: dict) -> None:
    n = len(frame_times_ms)
    if n:
        sorted_ms = sorted(frame_times_ms)
        mean_ms = sum(sorted_ms) / n
        p95_ms = sorted_ms[min(int(n * 0.95), n - 1)]
        max_ms = sorted_ms[-1]
    else:
        mean_ms = p95_ms = max_ms = None

    draw_calls_estimated = {
        "naive": args.cells * 13 + 1,
        "merged": args.cells * 2 + 1,
        "instanced": 4 + 1,
    }[args.mode]

    result = {
        "mode": args.mode,
        "cells": args.cells,
        "nodes": built["nodes"],
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "max_ms": max_ms,
        "gl_version": gl_info["gl_version"],
        "glsl_version": gl_info["glsl_version"],
        "renderer": gl_info["renderer"],
        "window_size": [1600, 1000],
        "shader": "none" if args.no_shader else "rim_light",
        "draw_calls_estimated": draw_calls_estimated,
        "pick_results": state["pick_results"],
        "frames_measured": n,
        "settle_frames": args.settle,
        "note": "mean/p95/max from this run are a sanity check only, not a timing verdict (shared machine).",
    }
    print("RESULT:", json.dumps(result, indent=2, default=str))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=str))
        print(f"json written: {args.out}")

    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
