"""Thin public façade for the OpenGL MatrixMan backend."""

from . import implementation as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
