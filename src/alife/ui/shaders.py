"""GLSL 150 ports of Ursina 8.3's ``unlit_with_fog_shader``, ``unlit_shader`` and ``text_shader``.

Why (``spikes/SPIKE.md`` B; ``spikes/ursina_render/README.md``): macOS only gives a modern
OpenGL context (4.1) when asked for ``gl-version 3 2``, and that is a strict core profile that
rejects Ursina's bundled ``#version 130/140`` shaders.

Call :func:`configure_panda` before ``Ursina()`` and :func:`install` right after it.

Each port keeps the stock shader's uniforms, inputs and maths, with three changes, all marked
``F2:`` in the source:

1. ``#version 150`` in both stages.
2. ``discard`` when the final alpha is 0. Ursina gives every model
   ``TransparencyAttrib.M_dual``. Panda draws that as an opaque pass and a blended pass, and
   splits the two with fixed-function alpha tests, which the core profile does not have. Without
   them, ``camera.overlay`` (a full-window ``color.clear`` quad that Ursina always creates) writes
   (0, 0, 0, 0) over the whole frame: the black window M0 found. :func:`configure_panda` sets
   ``m-dual false`` so Panda draws M_dual as plain alpha blending, and the ``discard`` stands in
   for the blended pass's ``alpha > 0`` test, so invisible pixels write no depth.
3. Text only: an outline sample that falls outside the glyph page reads as transparent. Ursina's
   default ``outline_offset`` (10, 10) pushes that sample off the page on purpose. On the core
   profile Panda stores alpha-only font pages as a red channel swizzled to alpha, and the swizzle
   turns the page's border colour (1, 1, 1, 0) into alpha 1, so every glyph drew as a solid box.

Checked in F2: a missing ``p3d_Color`` column reads as white on this driver, so the ports keep
Ursina's vertex-colour term. (The M0 note that it reads black was caused by the overlay bug.)
"""

from __future__ import annotations

import sys

from panda3d.core import ConfigVariableBool, loadPrcFileData
from ursina import Entity, Vec2, Vec3, camera, color
from ursina import shader as ursina_shader
from ursina.shader import Shader
from ursina.shaders.text_shader import text_shader as STOCK_TEXT
from ursina.shaders.unlit_shader import unlit_shader as STOCK_UNLIT
from ursina.shaders.unlit_with_fog_shader import unlit_with_fog_shader as STOCK_UNLIT_WITH_FOG

from alife.ui import iter_entities

PRC_SETTINGS = (
    "gl-version 3 2",  # without it macOS gives OpenGL 2.1 / GLSL 1.20 (SPIKE.md B)
    "m-dual false",  # draw M_dual as alpha blending; see change 2 in the module docstring
)

UNLIT_WITH_FOG_VERT = """#version 150
// F2: GLSL 150 port of Ursina 8.3 unlit_with_fog_shader (vertex); was #version 130.
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uvs;
uniform vec2 texture_scale;
uniform vec2 texture_offset;

in vec4 p3d_Color;
out vec4 vertex_color;
out vec3 vertex_world_position;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uvs = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
    vertex_color = p3d_Color;
    vertex_world_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
}
"""

UNLIT_WITH_FOG_FRAG = """#version 150
// F2: GLSL 150 port of Ursina 8.3 unlit_with_fog_shader (fragment); was #version 140.
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
in vec2 uvs;
out vec4 color;

in vec4 vertex_color;
in vec3 vertex_world_position;
uniform vec3 camera_world_position;
uniform vec4 fog_color;
uniform float fog_start;
uniform float fog_end;

void main() {
    color = texture(p3d_Texture0, uvs) * p3d_ColorScale * vertex_color;
    float distance_to_camera = length(vertex_world_position.xyz - camera_world_position);
    float fog_length = fog_end - fog_start;
    float t = distance_to_camera/fog_length;
    t = clamp(t, 0., 1.);
    color.rgb = mix(color.rgb, fog_color.rgb, t * fog_color.a);
    if (color.a <= 0.0) discard;  // F2: stands in for M_dual's alpha test (module docstring)
}
"""

UNLIT_VERT = """#version 150
// F2: GLSL 150 port of Ursina 8.3 unlit_shader (vertex); was #version 130.
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uvs;
uniform vec2 texture_scale;
uniform vec2 texture_offset;

in vec4 p3d_Color;
out vec4 vertex_color;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uvs = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
    vertex_color = p3d_Color;
}
"""

UNLIT_FRAG = """#version 150
// F2: GLSL 150 port of Ursina 8.3 unlit_shader (fragment); was #version 140. The stock
// shader also declares `in vec3 vertex_world_position`, which its vertex stage never writes
// and nothing reads; dropped.
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
in vec2 uvs;
out vec4 fragColor;

in vec4 vertex_color;

void main() {
    vec4 color = texture(p3d_Texture0, uvs) * p3d_ColorScale * vertex_color;
    if (color.a <= 0.0) discard;  // F2: stands in for M_dual's alpha test (module docstring)
    fragColor = color;
}
"""

# SDF text shader based on https://github.com/wezu/sdf_text, as shipped in Ursina 8.3.
# License: Copyright (c) 2018, wezu (wezu.dev@gmail.com)
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or
# without fee is hereby granted, provided that the above copyright notice and this permission
# notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
# SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE
# AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
TEXT_VERT = """#version 150
// F2: GLSL 150 port of Ursina 8.3 text_shader (vertex); was #version 130.
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uvs;

in vec4 p3d_Color;
out vec4 vertex_color;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uvs = p3d_MultiTexCoord0;
    vertex_color = p3d_Color;
}
"""

TEXT_FRAG = """#version 150
// F2: GLSL 150 port of Ursina 8.3 text_shader (fragment); was #version 140.
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform vec4 outline_color;
uniform vec2 outline_offset;
uniform float outline_power;
in vec2 uvs;
out vec4 FINAL_COLOR;

in vec4 vertex_color;

void main() {
    float dist = texture(p3d_Texture0, uvs).a;
    vec2 width = vec2(0.5-fwidth(dist), 0.5+fwidth(dist));
    float alpha = smoothstep(width.x, width.y, dist);
    //supersampling
    float scale = 0.354; // half of 1/sqrt2 - value from internet(???)
    vec2 duv = scale * (dFdx(uvs) + dFdy(uvs));
    vec4 box = vec4(uvs-duv, uvs+duv);
    alpha +=0.5*(smoothstep(width.x, width.y, texture(p3d_Texture0, box.xy).a)
            +smoothstep(width.x, width.y, texture(p3d_Texture0, box.zw).a)
            +smoothstep(width.x, width.y, texture(p3d_Texture0, box.xw).a)
            +smoothstep(width.x, width.y, texture(p3d_Texture0, box.zy).a));
    alpha /= 3.0; //weighted average, 1*1 + 4*0.5 = 3, so divide by 3
    //outline
    // F2: a sample outside the glyph page reads as transparent (change 3, module docstring).
    vec2 outline_uv = uvs-outline_offset;
    bool off_page = any(lessThan(outline_uv, vec2(0.0))) || any(greaterThan(outline_uv, vec2(1.0)));
    float outline = off_page ? 0.0 : pow(texture(p3d_Texture0, outline_uv).a, outline_power);
    vec4 result = mix(vec4(vertex_color.rgb, outline_color.a * outline), vertex_color, alpha);
    if (result.a <= 0.0) discard;  // F2: stands in for M_dual's alpha test (module docstring)
    FINAL_COLOR = result;
}
"""


def _camera_world_position() -> Vec3:
    return camera.world_position


UNLIT_WITH_FOG = Shader(
    name="unlit_with_fog_shader",
    language=Shader.GLSL,
    vertex=UNLIT_WITH_FOG_VERT,
    fragment=UNLIT_WITH_FOG_FRAG,
    default_input={
        "texture_scale": Vec2(1, 1),
        "texture_offset": Vec2(0.0, 0.0),
        "fog_color": color.clear,
        "fog_start": 10,
        "fog_end": 100,
        "camera_world_position": Vec3.zero,
    },
)
UNLIT_WITH_FOG.continuous_input["camera_world_position"] = _camera_world_position

UNLIT = Shader(
    name="unlit_shader",
    language=Shader.GLSL,
    vertex=UNLIT_VERT,
    fragment=UNLIT_FRAG,
    default_input={
        "texture_scale": Vec2(1, 1),
        "texture_offset": Vec2(0.0, 0.0),
    },
)

TEXT = Shader(
    name="text_shader",
    language=Shader.GLSL,
    vertex=TEXT_VERT,
    fragment=TEXT_FRAG,
    default_input={
        "outline_color": color.white,
        "outline_offset": Vec2(10, 10),
        "outline_power": 1.0,
    },
)

PORTS: dict[int, Shader] = {
    id(STOCK_UNLIT_WITH_FOG): UNLIT_WITH_FOG,
    id(STOCK_UNLIT): UNLIT,
    id(STOCK_TEXT): TEXT,
}


def port_for(shader: object) -> Shader | None:
    """The port that replaces ``shader``, or None if it is not one of Ursina's three."""
    return PORTS.get(id(shader))


def configure_panda() -> None:
    """Panda3D settings the ports need. Call before ``Ursina()``."""
    for line in PRC_SETTINGS:
        loadPrcFileData("", line)


def install() -> None:
    """Make Ursina use the ports everywhere. Call right after ``Ursina()``.

    Afterwards plain entities, UI entities (Buttons included), ``Text`` and ``camera.overlay``
    all use the ports, and so does any entity created later.
    """
    if ConfigVariableBool("m-dual").get_value():
        raise RuntimeError(
            "m-dual is on, so the UI would draw black: "
            "call alife.ui.shaders.configure_panda() before Ursina()"
        )
    for port in (UNLIT_WITH_FOG, UNLIT, TEXT):
        if not port.compiled:  # Text's shader setter assumes a compiled shader
            port.compile()
    # Names bound at import: entity.py's shader setter (default and UI swap), text.py, prefabs.
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "ursina" or name.startswith("ursina.")):
            continue
        for attr, value in list(vars(module).items()):
            port = port_for(value)
            if port is not None:
                setattr(module, attr, port)
    Entity.default_shader = UNLIT_WITH_FOG
    # Shaders named as strings, e.g. Entity.blink(shader='unlit_shader').
    for port in (UNLIT_WITH_FOG, UNLIT, TEXT):
        ursina_shader.imported_shaders[port.name] = port
    # Entities that Ursina() already built: camera.overlay, the editor UI, the hot reloader.
    for entity in iter_entities():
        port = port_for(entity.shader)
        if port is not None:
            entity.shader = port
