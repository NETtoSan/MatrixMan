"""OpenGL runtime ownership façade.

The implementation is currently shared with the legacy OpenGL module; this
module provides the stable runtime boundary for incremental extraction.
"""

from . import implementation as _implementation


def __getattr__(name):
    return getattr(_implementation, name)

