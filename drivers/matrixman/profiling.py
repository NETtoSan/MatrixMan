"""Backward-compatible module alias for OpenGL profiling helpers."""

from .backends.opengl import profiling as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
