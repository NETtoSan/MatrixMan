"""Backend-neutral MatrixMan configuration.

This module owns user/environment configuration only.  Backend profilers keep
their own implementations and are synchronized when they are already loaded.
"""

from __future__ import annotations

import os
import sys


_VALID_BACKENDS = {"auto", "cuda", "opengl"}
_preferred_backend: str | None = None
_profiling_override: bool | None = None
_trace_override: bool | None = None


def _env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() not in {"", "0", "false", "no", "off"}


def set_preferred_backend(name: str) -> str:
    value = str(name).strip().lower()
    if value not in _VALID_BACKENDS:
        raise ValueError("MatrixMan backend must be 'auto', 'cuda', or 'opengl'")
    global _preferred_backend
    # ``auto`` means no explicit Python override. This lets the selector apply
    # MATRIXMAN_BACKEND before falling back to its normal capability order.
    _preferred_backend = None if value == "auto" else value
    return value


def preferred_backend() -> str | None:
    """Return the explicit Python preference, if one has been set."""
    return _preferred_backend


def profiling_enabled(*, legacy_cuda: bool = False) -> bool:
    """Resolve profiling as Python override, canonical env, then legacy CUDA env."""
    if _profiling_override is not None:
        return _profiling_override
    if "MATRIXMAN_PROFILE" in os.environ:
        return _env_flag(os.environ.get("MATRIXMAN_PROFILE"))
    if legacy_cuda:
        return _env_flag(os.environ.get("MATRIXMAN_CUDA_PROFILE"))
    return False


def set_profiling(enabled: bool) -> bool:
    """Set profiling explicitly and update only the active backend profiler."""
    global _profiling_override
    _profiling_override = bool(enabled)
    from .backend import active_backend

    backend = active_backend()
    if backend is None:
        return _profiling_override
    module_name = f"drivers.matrixman.backends.{backend.name}.profiling"
    module = sys.modules.get(module_name)
    setter = getattr(module, "set_enabled", None) if module is not None else None
    if setter is not None:
        setter(_profiling_override)
    return _profiling_override


def trace_enabled() -> bool:
    """Resolve high-level tracing as Python override, env, then disabled."""
    if _trace_override is not None:
        return _trace_override
    if "MATRIXMAN_TRACE" in os.environ:
        return _env_flag(os.environ.get("MATRIXMAN_TRACE"))
    return False


def set_trace(enabled: bool) -> bool:
    """Set high-level operation tracing without touching backend selection."""
    global _trace_override
    _trace_override = bool(enabled)
    return _trace_override


def trace_log(message: str) -> None:
    """Print a high-level trace message when tracing is explicitly enabled."""
    if trace_enabled():
        print(message)


def clear_profiling_override() -> None:
    """Reset Python profiling state so environment configuration applies again."""
    global _profiling_override
    _profiling_override = None


def clear_trace_override() -> None:
    """Reset Python tracing state so the environment configuration applies."""
    global _trace_override
    _trace_override = None


class Configuration:
    """Small mutable configuration view exposed as ``matrixman.config``."""

    @property
    def preferred_backend(self) -> str | None:
        return preferred_backend()

    @preferred_backend.setter
    def preferred_backend(self, value: str) -> None:
        set_preferred_backend(value)

    @property
    def profiling(self) -> bool:
        return profiling_enabled(legacy_cuda=True)

    @profiling.setter
    def profiling(self, value: bool) -> None:
        set_profiling(value)

    @property
    def trace(self) -> bool:
        return trace_enabled()

    @trace.setter
    def trace(self, value: bool) -> None:
        set_trace(value)


config = Configuration()
