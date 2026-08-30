"""Backward-compatible module alias for MatrixMan's OpenGL backend."""

from .backends.opengl import backend as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
