"""Ursina UI helpers (GUI-only package: imports Ursina).

Also home to the HUD, info panel, toasts and confirm dialog (F5, M5), and the GLSL 150 shader
ports (``alife.ui.shaders``).
"""

from __future__ import annotations

from collections.abc import Iterator

from panda3d.core import NodePath
from ursina import Entity, camera, destroy, scene


def iter_entities() -> Iterator[Entity]:
    """Every Entity in the 3D scene and the UI, including disabled (stashed) ones."""
    roots: list[NodePath] = [scene]
    ui_render = getattr(camera, "ui_render", None)  # exists once Ursina() has run
    if ui_render is not None:
        roots.append(ui_render)
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.has_python_tag("Entity"):
            entity = node.get_python_tag("Entity")
            if id(entity) not in seen:
                seen.add(id(entity))
                yield entity
        stack.extend(node.get_children())
        stack.extend(node.get_stashed_children())


def destroy_tree(entity: Entity) -> None:
    """Destroy ``entity`` and every Entity below it.

    Ursina's ``destroy`` does not recurse: it detaches the node, but child Entities stay in
    ``scene.entities`` and keep receiving ``update()`` and ``input()``.
    """
    for child in list(entity.children):
        destroy_tree(child)
    destroy(entity)
