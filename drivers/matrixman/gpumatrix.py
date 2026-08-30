"""Backward-compatible module alias for the OpenGL matrix helper."""

from .backends.opengl import gpumatrix as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
