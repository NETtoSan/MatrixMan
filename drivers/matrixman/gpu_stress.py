"""Backward-compatible module alias for the OpenGL stress helper."""

from .backends.opengl import gpu_stress as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
