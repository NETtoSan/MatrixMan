"""Central MatrixMan runtime configuration.

Environment values are loaded at import time. Python assignments take
precedence until ``reloadFromEnvironment()`` is called. Configuration changes
do not create or destroy GPU contexts.
"""

from __future__ import annotations

import os
import sys
from typing import Any


_FALSE = {"", "0", "false", "no", "off"}
_TRUE = {"1", "true", "yes", "on"}
_DEFAULTS = {
    "backend": "auto", "tileLimit": 256, "resolvedTileLimit": 256,
    "tileSync": "per_tile", "convSpatialReuse": False,
    "skipPreConsolidationSync": False, "diagnosticTiles": False,
    "diagnosticRectTiles": False, "diagTileWidth": None,
    "diagTileHeight": None, "diagTileOrder": "normal",
    "diagConvWorkload": "heavy", "profile": False, "cudaProfile": False,
    "profileDetail": False, "gpuTiming": False, "trace": False,
    "debug": False, "gpuPostprocess": False, "auditCpuLeaks": False,
    "cudaDebug": False, "cudaDisableAsyncQueue": False,
    "cudaDisableAllocPool": False, "cudaDisableSpecializedConv": False,
    "cudaConv3x3Variant": "plane", "cudaLegacyModuleLoad": False,
    "tileAutotuneRefresh": False,
}
_ENV_FIELDS = {
    "MATRIXMAN_BACKEND": "backend", "MATRIXMAN_TILE_LIMIT": "tileLimit",
    "MATRIXMAN_TILE_SYNC": "tileSync", "MATRIXMAN_CONV_SPATIAL_REUSE": "convSpatialReuse",
    "MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC": "skipPreConsolidationSync",
    "MATRIXMAN_DIAGNOSTIC_TILES": "diagnosticTiles", "MATRIXMAN_DIAGNOSTIC_RECT_TILES": "diagnosticRectTiles",
    "MATRIXMAN_DIAG_TILE_WIDTH": "diagTileWidth", "MATRIXMAN_DIAG_TILE_HEIGHT": "diagTileHeight",
    "MATRIXMAN_DIAG_TILE_ORDER": "diagTileOrder", "MATRIXMAN_DIAG_CONV_WORKLOAD": "diagConvWorkload",
    "MATRIXMAN_PROFILE": "profile", "MATRIXMAN_CUDA_PROFILE": "cudaProfile",
    "MATRIXMAN_PROFILE_DETAIL": "profileDetail", "MATRIXMAN_GPU_TIMING": "gpuTiming",
    "MATRIXMAN_TRACE": "trace", "MATRIXMAN_DEBUG": "debug",
    "MATRIXMAN_GPU_POSTPROCESS": "gpuPostprocess", "MATRIXMAN_AUDIT_CPU_LEAKS": "auditCpuLeaks",
    "MATRIXMAN_CUDA_DEBUG": "cudaDebug", "MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE": "cudaDisableAsyncQueue",
    "MATRIXMAN_CUDA_DISABLE_ALLOC_POOL": "cudaDisableAllocPool", "MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV": "cudaDisableSpecializedConv",
    "MATRIXMAN_CUDA_CONV3X3_VARIANT": "cudaConv3x3Variant", "MATRIXMAN_CUDA_LEGACY_MODULE_LOAD": "cudaLegacyModuleLoad",
    "MATRIXMAN_TILE_AUTOTUNE_REFRESH": "tileAutotuneRefresh",
}
_BOOL_FIELDS = {
    "convSpatialReuse", "skipPreConsolidationSync", "diagnosticTiles", "diagnosticRectTiles",
    "profile", "cudaProfile", "profileDetail", "gpuTiming", "trace", "debug", "gpuPostprocess",
    "auditCpuLeaks", "cudaDebug", "cudaDisableAsyncQueue", "cudaDisableAllocPool",
    "cudaDisableSpecializedConv", "cudaLegacyModuleLoad", "tileAutotuneRefresh",
}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def _parse_value(field: str, value: Any) -> Any:
    if field in _BOOL_FIELDS:
        return _parse_bool(value)
    if field == "tileLimit":
        if isinstance(value, str) and value.strip().lower() == "auto":
            return "auto"
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("tileLimit must be a positive integer or 'auto'") from exc
        if value <= 0:
            raise ValueError("tileLimit must be a positive integer or 'auto'")
        return value
    if field in {"diagTileWidth", "diagTileHeight"}:
        if value is None or str(value).strip() == "":
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"{field} must be positive")
        return value
    if field == "tileSync":
        value = str(value).strip().lower() or "per_tile"
        if value not in {"per_tile", "end", "flush", "none"}:
            raise ValueError("tileSync must be one of: per_tile, end, flush, none")
        return value
    if field == "backend":
        return str(value).strip().lower() or "auto"
    if field in {"diagTileOrder", "diagConvWorkload", "cudaConv3x3Variant"}:
        return str(value).strip().lower()
    return value


class Configuration:
    """Singleton-style public configuration exposed as ``matrixman.config``."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._overrides: set[str] = set()
        self.reloadFromEnvironment()

    def reloadFromEnvironment(self) -> None:
        """Reload all settings from the process environment."""
        self._overrides.clear()
        self._values = dict(_DEFAULTS)
        for env_name, field in _ENV_FIELDS.items():
            if env_name in os.environ:
                self._values[field] = _parse_value(field, os.environ[env_name])
        requested = self._values["tileLimit"]
        self._values["resolvedTileLimit"] = 256 if requested == "auto" else requested
        if "_sync_profile" in globals():
            _sync_profile(bool(self._values["profile"]))

    def _reload_field(self, field: str) -> None:
        environment_name = next(name for name, value in _ENV_FIELDS.items() if value == field)
        self._values[field] = (
            _parse_value(field, os.environ[environment_name])
            if environment_name in os.environ else _DEFAULTS[field]
        )
        self._overrides.discard(field)

    def reset(self) -> None:
        """Reset to built-in defaults without modifying ``os.environ``."""
        self._overrides.clear()
        self._values = dict(_DEFAULTS)
        if "_sync_profile" in globals():
            _sync_profile(False)

    def asDict(self) -> dict[str, Any]:
        return dict(self._values)

    def resolveTileLimit(self, value: int) -> int:
        value = _parse_value("tileLimit", value)
        if value == "auto":
            raise ValueError("resolved tile limit must be an integer")
        self._values["resolvedTileLimit"] = value
        return value

    def _set(self, field: str, value: Any) -> None:
        value = _parse_value(field, value)
        self._values[field] = value
        self._overrides.add(field)
        if field == "tileLimit":
            self._values["resolvedTileLimit"] = 256 if value == "auto" else value
        if field == "profile":
            _sync_profile(value)

    def __repr__(self) -> str:
        names = ("backend", "tileLimit", "resolvedTileLimit", "tileSync", "convSpatialReuse", "profile", "gpuTiming")
        return "MatrixManConfig(" + ", ".join(f"{n}={self._values[n]!r}" for n in names) + ")"

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


def _property(field: str):
    return property(lambda self: self._values[field], lambda self, value: self._set(field, value))


for _field in _DEFAULTS:
    if _field != "resolvedTileLimit":
        setattr(Configuration, _field, _property(_field))
Configuration.resolvedTileLimit = property(lambda self: self._values["resolvedTileLimit"])

config = Configuration()


def _sync_profile(value: bool) -> None:
    from .backend import active_backend
    backend = active_backend()
    if backend is None:
        return
    module = sys.modules.get(f"drivers.matrixman.backends.{backend.name}.profiling")
    setter = getattr(module, "set_enabled", None) if module is not None else None
    if setter is not None:
        setter(bool(value))


def set_preferred_backend(name: str) -> str:
    value = _parse_value("backend", name)
    if value not in {"auto", "cuda", "opengl"}:
        raise ValueError("MatrixMan backend must be 'auto', 'cuda', or 'opengl'")
    config._set("backend", value)
    return value


def preferred_backend() -> str | None:
    return None if config.backend == "auto" else config.backend


def profiling_enabled(*, legacy_cuda: bool = False) -> bool:
    return bool(config.profile or (legacy_cuda and config.cudaProfile))


def set_profiling(enabled: bool) -> bool:
    config._set("profile", enabled)
    return bool(config.profile)


def trace_enabled() -> bool:
    return bool(config.trace)


def set_trace(enabled: bool = True) -> bool:
    config._set("trace", enabled)
    return bool(config.trace)


def trace_log(message: str) -> None:
    if trace_enabled():
        print(message)


def clear_profiling_override() -> None:
    config._reload_field("profile")
    _sync_profile(bool(config.profile))


def clear_trace_override() -> None:
    config._reload_field("trace")
