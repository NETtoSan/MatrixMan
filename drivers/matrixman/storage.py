"""Backward-compatible module alias for OpenGL tensor storage helpers."""

from .backends.opengl import storage as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
