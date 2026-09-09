"""Context-local OpenGL state suppression for MatrixMan-owned contexts."""

from __future__ import annotations

import functools

from . import gpumatrix as gm


_state = {
    "program": None,
    "active_texture": None,
    "textures": {},
    "framebuffer": None,
    "viewport": None,
    "uniforms": {},
}
_skipped: dict[str, int] = {}


def reset() -> None:
    """Invalidate cached state at a known runtime/context boundary."""
    _state["program"] = None
    _state["active_texture"] = None
    _state["textures"].clear()
    _state["framebuffer"] = None
    _state["viewport"] = None
    _state["uniforms"].clear()


def reset_counters() -> None:
    _skipped.clear()


def snapshot() -> dict[str, int]:
    return dict(_skipped)


def _skip(name: str) -> None:
    _skipped[name] = _skipped.get(name, 0) + 1


def _wrap(name: str, original):
    @functools.wraps(original)
    def wrapped(*args):
        if name == "glDeleteTextures":
            # Deletion implicitly unbinds texture names in OpenGL.  Drop all
            # cached bindings conservatively; the next render re-establishes
            # only the bindings it needs.
            _state["textures"].clear()
            return original(*args)
        if name == "glDeleteFramebuffers":
            _state["framebuffer"] = None
            return original(*args)
        if name == "glUseProgram":
            value = int(args[0])
            if _state["program"] == value:
                _skip(name)
                return None
            _state["program"] = value
        elif name == "glActiveTexture":
            value = int(args[0])
            if _state["active_texture"] == value:
                _skip(name)
                return None
            _state["active_texture"] = value
        elif name == "glBindTexture":
            target, texture = int(args[0]), int(args[1])
            key = (_state["active_texture"], target)
            if _state["textures"].get(key) == texture:
                _skip(name)
                return None
            _state["textures"][key] = texture
        elif name == "glBindFramebuffer":
            value = int(args[1])
            if _state["framebuffer"] == value:
                _skip(name)
                return None
            _state["framebuffer"] = value
        elif name == "glViewport":
            value = tuple(int(item) for item in args)
            if _state["viewport"] == value:
                _skip(name)
                return None
            _state["viewport"] = value
        elif name == "glUniform1i":
            key = (_state["program"], int(args[0]))
            value = int(args[1])
            if _state["uniforms"].get(key) == value:
                _skip(name)
                return None
            _state["uniforms"][key] = value
        return original(*args)

    wrapped._matrixman_state_wrapper = True
    wrapped.original = original
    return wrapped


def install() -> None:
    """Wrap state-changing entry points for the current GL context."""
    reset()
    for name in (
        "glUseProgram", "glActiveTexture", "glBindTexture", "glBindFramebuffer",
        "glViewport", "glUniform1i", "glDeleteTextures", "glDeleteFramebuffers",
    ):
        original = getattr(gm, name, None)
        if original is not None and not getattr(original, "_matrixman_state_wrapper", False):
            setattr(gm, name, _wrap(name, original))
