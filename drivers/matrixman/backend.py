"""Small interface for MatrixMan compute backends."""

from __future__ import annotations


class Backend:
    """Operations required by the MatrixMan-facing layer."""

    def device_info(self) -> dict[str, str]:
        """Return concise runtime information for user-facing reporting."""
        return {"backend": self.__class__.__name__}

    def matmul(self, a, b):
        raise NotImplementedError

    def synchronize(self):
        raise NotImplementedError


_active_backend: Backend | None = None


def get_backend() -> Backend:
    """Return MatrixMan's selected backend."""
    global _active_backend
    if _active_backend is None:
        from .selector import select_backend

        select_backend()
    return _active_backend


def set_backend(backend: Backend) -> Backend:
    """Set the selected backend and return it."""
    global _active_backend
    _active_backend = backend
    return backend
