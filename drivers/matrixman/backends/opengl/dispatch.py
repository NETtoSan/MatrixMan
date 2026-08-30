"""PyTorch dispatch boundary for the OpenGL MatrixMan backend."""

from . import implementation as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

