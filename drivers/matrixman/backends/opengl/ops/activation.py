"""OpenGL activation operation boundary."""

from .. import implementation as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

