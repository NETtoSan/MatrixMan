"""MatrixMan operations routed through the active compute backend."""

from .backend import Backend, get_backend

__all__ = ["Backend", "get_backend"]

