"""Shared OpenGL framebuffer and fullscreen-render plumbing."""

from __future__ import annotations

from . import gpumatrix as gm


def attach_output(runtime, owner, width: int | None = None, height: int | None = None) -> None:
    """Attach an output texture and set the viewport using the shared FBO."""
    if width is None:
        width = owner.layout.texture_width
    if height is None:
        height = owner.layout.texture_height
    gm.glViewport(0, 0, width, height)
    gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, runtime.fbo.value)
    gm.glFramebufferTexture2D(
        gm.GL_FRAMEBUFFER,
        gm.GL_COLOR_ATTACHMENT0,
        gm.GL_TEXTURE_2D,
        owner.texture,
        0,
    )


def draw_fullscreen_quad() -> None:
    """Issue the backend's existing immediate-mode fullscreen quad."""
    gm.glBegin(gm.GL_QUADS)
    gm.glVertex2f(-1.0, -1.0)
    gm.glVertex2f(1.0, -1.0)
    gm.glVertex2f(1.0, 1.0)
    gm.glVertex2f(-1.0, 1.0)
    gm.glEnd()
