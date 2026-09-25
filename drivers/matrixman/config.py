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
_PROFILE_MODES = ("summary", "detail", "trace")
_BACKEND_VALUES = ("auto", "opengl", "cuda")
_TILE_SYNC_VALUES = ("per_tile", "end", "flush", "none")
_SYNC_POLICY_VALUES = ("safe", "auto", "relaxed")
_POOL_VALUES = ("safe", "deferred")
_DEFAULTS = {
    "backend": "auto", "tileLimit": 256, "resolvedTileLimit": 256,
    "useDGPU": False,
    "tileSync": "per_tile", "syncPolicy": "safe", "convSpatialReuse": False,
    "activationPool": "safe",
    "scratchPool": "safe",
    "preparedExecution": True, "convDiag": False,
    "skipPreConsolidationSync": True, "diagnosticTiles": False,
    "diagnosticRectTiles": False, "diagTileWidth": None,
    "diagTileHeight": None, "diagTileOrder": "normal",
    "diagConvWorkload": "heavy", "profile": False, "cudaProfile": False,
    "profileDetail": False, "profileDispatch": False, "gpuTiming": False, "trace": False,
    "debug": False, "gpuPostprocess": False, "auditCpuLeaks": False,
    "unsafeScratchReuse": False, "debugFreshScratch": False, "scratchEpochPool": False,
    "debugClearReusedScratch": False,
    "cudaDebug": False, "cudaDisableAsyncQueue": False,
    "cudaDisableAllocPool": False, "cudaDisableSpecializedConv": False,
    "cudaConv3x3Variant": "plane", "cudaLegacyModuleLoad": False,
    "tileAutotuneRefresh": False, "disableAutoPrepare": False,
}
_ENV_FIELDS = {
    "MATRIXMAN_BACKEND": "backend", "MATRIXMAN_TILE_LIMIT": "tileLimit",
    "MATRIXMAN_USE_DGPU": "useDGPU",
    "MATRIXMAN_TILE_SYNC": "tileSync", "MATRIXMAN_SYNC_POLICY": "syncPolicy",
    "MATRIXMAN_ACTIVATION_POOL": "activationPool",
    "MATRIXMAN_SCRATCH_POOL": "scratchPool",
    "MATRIXMAN_CONV_SPATIAL_REUSE": "convSpatialReuse",
    "MATRIXMAN_PREPARED_EXECUTION": "preparedExecution",
    "MATRIXMAN_CONV_DIAG": "convDiag",
    "MATRIXMAN_SKIP_PRE_CONSOLIDATION_SYNC": "skipPreConsolidationSync",
    "MATRIXMAN_DIAGNOSTIC_TILES": "diagnosticTiles", "MATRIXMAN_DIAGNOSTIC_RECT_TILES": "diagnosticRectTiles",
    "MATRIXMAN_DIAG_TILE_WIDTH": "diagTileWidth", "MATRIXMAN_DIAG_TILE_HEIGHT": "diagTileHeight",
    "MATRIXMAN_DIAG_TILE_ORDER": "diagTileOrder", "MATRIXMAN_DIAG_CONV_WORKLOAD": "diagConvWorkload",
    "MATRIXMAN_PROFILE": "profile", "MATRIXMAN_CUDA_PROFILE": "cudaProfile",
    "MATRIXMAN_PROFILE_DETAIL": "profileDetail", "MATRIXMAN_PROFILE_DISPATCH": "profileDispatch",
    "MATRIXMAN_GPU_TIMING": "gpuTiming",
    "MATRIXMAN_TRACE": "trace", "MATRIXMAN_DEBUG": "debug",
    "MATRIXMAN_DEBUG_UNSAFE_SCRATCH_REUSE": "unsafeScratchReuse",
    "MATRIXMAN_DEBUG_FRESH_SCRATCH": "debugFreshScratch",
    "MATRIXMAN_SCRATCH_EPOCH_POOL": "scratchEpochPool",
    "MATRIXMAN_DEBUG_CLEAR_REUSED_SCRATCH": "debugClearReusedScratch",
    "MATRIXMAN_GPU_POSTPROCESS": "gpuPostprocess", "MATRIXMAN_AUDIT_CPU_LEAKS": "auditCpuLeaks",
    "MATRIXMAN_CUDA_DEBUG": "cudaDebug", "MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE": "cudaDisableAsyncQueue",
    "MATRIXMAN_CUDA_DISABLE_ALLOC_POOL": "cudaDisableAllocPool", "MATRIXMAN_CUDA_DISABLE_SPECIALIZED_CONV": "cudaDisableSpecializedConv",
    "MATRIXMAN_CUDA_CONV3X3_VARIANT": "cudaConv3x3Variant", "MATRIXMAN_CUDA_LEGACY_MODULE_LOAD": "cudaLegacyModuleLoad",
    "MATRIXMAN_TILE_AUTOTUNE_REFRESH": "tileAutotuneRefresh", "MATRIXMAN_DISABLE_AUTO_PREPARE": "disableAutoPrepare",
}
_BOOL_FIELDS = {
    "useDGPU", "convSpatialReuse", "preparedExecution", "convDiag", "skipPreConsolidationSync", "diagnosticTiles", "diagnosticRectTiles",
    "profile", "cudaProfile", "profileDetail", "profileDispatch", "gpuTiming", "trace", "debug", "gpuPostprocess",
    "auditCpuLeaks", "unsafeScratchReuse", "debugFreshScratch", "scratchEpochPool", "debugClearReusedScratch", "cudaDebug", "cudaDisableAsyncQueue", "cudaDisableAllocPool",
    "cudaDisableSpecializedConv", "cudaLegacyModuleLoad", "tileAutotuneRefresh", "disableAutoPrepare",
}

_SNAKE_ALIASES = {
    "useDGPU": "use_dgpu",
    "tileLimit": "tile_limit",
    "tileSync": "tile_sync",
    "syncPolicy": "sync_policy",
    "activationPool": "activation_pool",
    "scratchPool": "scratch_pool",
    "convSpatialReuse": "conv_spatial_reuse",
    "preparedExecution": "prepared_execution",
    "convDiag": "conv_diag",
    "skipPreConsolidationSync": "skip_pre_consolidation_sync",
    "diagnosticTiles": "diagnostic_tiles",
    "diagnosticRectTiles": "diagnostic_rect_tiles",
    "diagTileWidth": "diag_tile_width",
    "diagTileHeight": "diag_tile_height",
    "diagTileOrder": "diag_tile_order",
    "diagConvWorkload": "diag_conv_workload",
    "cudaProfile": "cuda_profile",
    "profileDetail": "profile_detail",
    "profileDispatch": "profile_dispatch",
    "gpuTiming": "gpu_timing",
    "gpuPostprocess": "gpu_postprocess",
    "auditCpuLeaks": "audit_cpu_leaks",
    "unsafeScratchReuse": "unsafe_scratch_reuse",
    "debugFreshScratch": "debug_fresh_scratch",
    "scratchEpochPool": "scratch_epoch_pool",
    "debugClearReusedScratch": "debug_clear_reused_scratch",
    "cudaDebug": "cuda_debug",
    "cudaDisableAsyncQueue": "cuda_disable_async_queue",
    "cudaDisableAllocPool": "cuda_disable_alloc_pool",
    "cudaDisableSpecializedConv": "cuda_disable_specialized_conv",
    "cudaConv3x3Variant": "cuda_conv3x3_variant",
    "cudaLegacyModuleLoad": "cuda_legacy_module_load",
    "tileAutotuneRefresh": "tile_autotune_refresh",
    "disableAutoPrepare": "disable_auto_prepare",
}
_RUNTIME_PROTECTED_FIELDS = {
    "backend", "useDGPU", "tileLimit", "tileSync", "syncPolicy",
    "activationPool", "scratchPool",
}
_FIELD_DOCS = {
    "backend": "Backend selection. Values: auto, cuda, opengl.",
    "useDGPU": "Prefer a discrete GPU when the OpenGL adapter can honor it. bool.",
    "tileLimit": "Maximum physical OpenGL tile dimension. Integer or auto.",
    "tileSync": "Tiled-convolution synchronization. Values: per_tile, end, flush, none.",
    "syncPolicy": "OpenGL synchronization policy. Values: safe, auto, relaxed.",
    "activationPool": "Activation texture reuse. Values: safe or deferred.",
    "scratchPool": "Scratch texture reuse. Values: safe or deferred.",
    "profile": "Profiler report mode. Values: off, summary, detail, trace.",
    "profileDispatch": "Collect PyTorch-facing dispatch timing. bool.",
    "trace": "Enables verbose live MatrixMan execution diagnostics. bool.",
    "convDiag": "Enables the persistent OpenGL convolution diagnostics window. bool.",
    "gpuTiming": "Enables deferred OpenGL timer queries. bool.",
}
_CANONICAL_BY_PUBLIC = {
    **{field: field for field in _DEFAULTS if field != "resolvedTileLimit"},
    **{alias: field for field, alias in _SNAKE_ALIASES.items()},
    "gpu_preference": "gpu_preference",
    "scratchPolicy": "scratchPolicy",
    "scratch_policy": "scratchPolicy",
}
_DISPLAY_FIELDS = (
    "backend", "profile", "trace", "syncPolicy", "activationPool", "scratchPool",
    "tileLimit", "tileSync", "scratchPolicy", "useDGPU", "convDiag",
    "preparedExecution", "convSpatialReuse", "skipPreConsolidationSync",
    "diagnosticTiles", "diagnosticRectTiles", "diagTileWidth", "diagTileHeight",
    "diagTileOrder", "diagConvWorkload", "profileDetail", "profileDispatch", "gpuTiming", "debug",
    "gpuPostprocess", "auditCpuLeaks", "unsafeScratchReuse", "debugFreshScratch",
    "scratchEpochPool", "debugClearReusedScratch", "cudaProfile", "cudaDebug",
    "cudaDisableAsyncQueue", "cudaDisableAllocPool", "cudaDisableSpecializedConv",
    "cudaConv3x3Variant", "cudaLegacyModuleLoad", "tileAutotuneRefresh",
    "disableAutoPrepare",
)

_SETTING_DETAILS = {
    "backend": ("Backend / device", "stable", "Selects the MatrixMan execution backend."),
    "gpu_preference": ("Backend / device", "backend-specific", "Requests an integrated or discrete OpenGL adapter."),
    "tile_limit": ("OpenGL tiling", "stable", "Sets the maximum physical OpenGL tile dimension."),
    "tile_sync": ("OpenGL tiling", "stable", "Controls synchronization around tiled convolution work."),
    "sync_policy": ("Synchronization", "stable", "Controls the OpenGL synchronization policy."),
    "skip_pre_consolidation_sync": ("Synchronization", "experimental", "Skips a redundant pre-consolidation synchronization point."),
    "activation_pool": ("Memory / texture reuse", "experimental", "Controls safe or completion-deferred activation texture reuse."),
    "scratch_pool": ("Memory / texture reuse", "experimental", "Controls safe or completion-deferred scratch texture reuse."),
    "scratch_policy": ("Memory / texture reuse", "diagnostic", "Reports the effective legacy scratch diagnostic policy."),
    "conv_spatial_reuse": ("Convolution", "experimental", "Enables the OpenGL spatial-reuse convolution path."),
    "prepared_execution": ("Convolution", "experimental", "Enables prepared OpenGL execution and inference folding."),
    "disable_auto_prepare": ("Convolution", "stable", "Disables lazy model preparation."),
    "profile": ("Profiling", "stable", "Controls profiler collection and report verbosity."),
    "trace": ("Profiling", "diagnostic", "Enables verbose live execution diagnostics."),
    "profile_detail": ("Profiling", "diagnostic", "Enables detailed profiler sections."),
    "profile_dispatch": ("Profiling", "diagnostic", "Collects PyTorch-facing dispatch timing."),
    "gpu_timing": ("Profiling", "diagnostic", "Enables deferred OpenGL timer queries."),
    "conv_diag": ("Diagnostics", "diagnostic", "Enables the persistent convolution diagnostic view."),
    "diagnostic_tiles": ("Diagnostics", "diagnostic", "Captures tiled-convolution diagnostic snapshots."),
    "diagnostic_rect_tiles": ("Diagnostics", "diagnostic", "Enables rectangular diagnostic tile experiments."),
    "diag_tile_width": ("Diagnostics", "diagnostic", "Sets the diagnostic rectangular tile width."),
    "diag_tile_height": ("Diagnostics", "diagnostic", "Sets the diagnostic rectangular tile height."),
    "diag_tile_order": ("Diagnostics", "diagnostic", "Selects diagnostic tile traversal order."),
    "diag_conv_workload": ("Diagnostics", "diagnostic", "Selects the compatibility diagnostic workload."),
    "debug": ("Advanced / debug", "diagnostic", "Enables low-level OpenGL diagnostic output."),
    "gpu_postprocess": ("Advanced / debug", "experimental", "Enables benchmark GPU postprocessing."),
    "audit_cpu_leaks": ("Advanced / debug", "diagnostic", "Audits unexpected CPU materialization."),
    "tile_autotune_refresh": ("Advanced / debug", "diagnostic", "Forces tile autotuning to refresh its cache entry."),
    "unsafe_scratch_reuse": ("Advanced / debug", "experimental", "Disables safe scratch reuse fencing."),
    "debug_fresh_scratch": ("Advanced / debug", "diagnostic", "Allocates fresh scratch textures for diagnosis."),
    "scratch_epoch_pool": ("Advanced / debug", "diagnostic", "Uses the diagnostic epoch scratch pool."),
    "debug_clear_reused_scratch": ("Advanced / debug", "diagnostic", "Clears reused scratch textures for diagnosis."),
    "cuda_profile": ("CUDA", "backend-specific", "Enables the legacy CUDA profiler."),
    "cuda_debug": ("CUDA", "backend-specific", "Enables CUDA/PTX debug output."),
    "cuda_disable_async_queue": ("CUDA", "backend-specific", "Disables CUDA asynchronous queueing."),
    "cuda_disable_alloc_pool": ("CUDA", "backend-specific", "Disables the CUDA allocation pool."),
    "cuda_disable_specialized_conv": ("CUDA", "backend-specific", "Disables specialized CUDA convolutions."),
    "cuda_conv3x3_variant": ("CUDA", "backend-specific", "Selects the CUDA 3x3 convolution variant."),
    "cuda_legacy_module_load": ("CUDA", "backend-specific", "Uses legacy CUDA module loading."),
}
_CATEGORY_ORDER = (
    "Backend / device", "OpenGL tiling", "Synchronization",
    "Memory / texture reuse", "Convolution", "Profiling", "Diagnostics",
    "CUDA", "Advanced / debug",
)
_SPECIAL_OPTIONS = {
    "backend": _BACKEND_VALUES,
    "profile": ("off", "summary", "detail", "trace"),
    "tile_limit": {"type": "int|string", "values": ("auto",), "range": "positive integer"},
    "tile_sync": _TILE_SYNC_VALUES,
    "sync_policy": _SYNC_POLICY_VALUES,
    "activation_pool": _POOL_VALUES,
    "scratch_pool": _POOL_VALUES,
    "scratch_policy": ("safe", "fresh", "epoch", "unsafe_reuse"),
    "gpu_preference": ("integrated", "discrete"),
    "diag_tile_width": {"type": "int|None", "range": "positive integer or None"},
    "diag_tile_height": {"type": "int|None", "range": "positive integer or None"},
    "diag_tile_order": ("normal", "reverse", "column", "reverse_column"),
    "diag_conv_workload": ("heavy", "medium", "light", "one_by_one"),
    "cuda_conv3x3_variant": ("plane", "plane_legacy"),
}


def _canonical_field(name: str) -> str:
    try:
        return _CANONICAL_BY_PUBLIC[name]
    except KeyError as exc:
        raise AttributeError(f"unknown MatrixMan configuration setting: {name}") from exc


def _public_name_for_field(field: str) -> str:
    if field == "useDGPU":
        return "gpu_preference"
    if field == "scratchPolicy":
        return "scratch_policy"
    return _SNAKE_ALIASES.get(field, field)


def _metadata_options(name: str, field: str) -> Any:
    if name in _SPECIAL_OPTIONS:
        return _SPECIAL_OPTIONS[name]
    if field in _BOOL_FIELDS:
        return (False, True)
    if field in {"diagTileWidth", "diagTileHeight"}:
        return {"type": "int|None", "range": "positive integer or None"}
    if field in {"diagTileOrder", "diagConvWorkload", "cudaConv3x3Variant"}:
        return {"type": "str", "values": "parser-defined"}
    if field == "scratchPolicy":
        return _SPECIAL_OPTIONS["scratch_policy"]
    if field == "backend":
        return _SPECIAL_OPTIONS["backend"]
    return {"type": type(_DEFAULTS[field]).__name__}


def _metadata_type(name: str, field: str, options: Any) -> str:
    if isinstance(options, dict) and "type" in options:
        return options["type"]
    if name == "profile":
        return "str|bool"
    if field in _BOOL_FIELDS:
        return "bool"
    if name == "gpu_preference":
        return "str"
    if field in {"diagTileWidth", "diagTileHeight"}:
        return "int|None"
    if field == "tileLimit":
        return "int|str"
    return type(_DEFAULTS.get(field, "")).__name__


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
    if field == "profile":
        if isinstance(value, bool):
            return "summary" if value else False
        text = str(value).strip().lower()
        if text in _FALSE:
            return False
        if text in _TRUE:
            return "summary"
        if text in _PROFILE_MODES:
            return text
        raise ValueError(
            "profile must be one of: 0, 1, summary, detail, trace"
        )
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
        if value not in _TILE_SYNC_VALUES:
            raise ValueError("tileSync must be one of: per_tile, end, flush, none")
        return value
    if field == "syncPolicy":
        value = str(value).strip().lower() or "safe"
        if value not in _SYNC_POLICY_VALUES:
            raise ValueError("syncPolicy must be one of: safe, auto, relaxed")
        return value
    if field == "activationPool":
        value = str(value).strip().lower() or "safe"
        if value not in _POOL_VALUES:
            raise ValueError("activationPool must be one of: safe, deferred")
        return value
    if field == "scratchPool":
        value = str(value).strip().lower() or "safe"
        if value not in _POOL_VALUES:
            raise ValueError("scratchPool must be one of: safe, deferred")
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
        self._sources: dict[str, str] = {}
        self.reloadFromEnvironment()

    def reloadFromEnvironment(self) -> None:
        """Reload all settings from the process environment."""
        self._overrides.clear()
        values = dict(_DEFAULTS)
        sources = {field: "default" for field in _DEFAULTS if field != "resolvedTileLimit"}
        for env_name, field in _ENV_FIELDS.items():
            if env_name in os.environ:
                values[field] = _parse_value(field, os.environ[env_name])
                sources[field] = "environment"
        _validate_values(values)
        for field in _RUNTIME_PROTECTED_FIELDS:
            if field in self._values and self._values.get(field) != values.get(field):
                self._ensure_mutable(field)
        requested = values["tileLimit"]
        values["resolvedTileLimit"] = 256 if requested == "auto" else requested
        self._values = values
        self._sources = sources
        if "_sync_profile" in globals():
            _sync_profile(bool(self._values["profile"]))
            _sync_frontend_profile()

    def _reload_field(self, field: str) -> None:
        self._ensure_mutable(field)
        environment_name = next(name for name, value in _ENV_FIELDS.items() if value == field)
        value = (
            _parse_value(field, os.environ[environment_name])
            if environment_name in os.environ else _DEFAULTS[field]
        )
        old_value = self._values.get(field)
        old_source = self._sources.get(field, "default")
        self._values[field] = value
        self._sources[field] = "environment" if environment_name in os.environ else "default"
        try:
            _validate_values(self._values)
        except Exception:
            self._values[field] = old_value
            self._sources[field] = old_source
            raise
        self._overrides.discard(field)

    def reset(self) -> None:
        """Reset to built-in defaults without modifying ``os.environ``."""
        for field in _RUNTIME_PROTECTED_FIELDS:
            if field in self._values and self._values.get(field) != _DEFAULTS.get(field):
                self._ensure_mutable(field)
        self._overrides.clear()
        self._values = dict(_DEFAULTS)
        self._sources = {field: "default" for field in _DEFAULTS if field != "resolvedTileLimit"}
        if "_sync_profile" in globals():
            _sync_profile(False)
            _sync_frontend_profile()

    def asDict(self) -> dict[str, Any]:
        return dict(self._values)

    def resolveTileLimit(self, value: int) -> int:
        value = _parse_value("tileLimit", value)
        if value == "auto":
            raise ValueError("resolved tile limit must be an integer")
        self._values["resolvedTileLimit"] = value
        return value

    def _set(self, field: str, value: Any) -> None:
        self._ensure_mutable(field)
        value = _parse_value(field, value)
        old_value = self._values.get(field)
        old_limit = self._values.get("resolvedTileLimit")
        old_source = self._sources.get(field, "default")
        self._values[field] = value
        if field == "tileLimit":
            self._values["resolvedTileLimit"] = 256 if value == "auto" else value
        try:
            _validate_values(self._values)
        except Exception:
            self._values[field] = old_value
            self._values["resolvedTileLimit"] = old_limit
            self._sources[field] = old_source
            raise
        self._overrides.add(field)
        self._sources[field] = "python override"
        if field == "profile":
            _sync_profile(value)
        if field in {"profile", "profileDispatch"}:
            _sync_frontend_profile()

    def __repr__(self) -> str:
        names = ("backend", "useDGPU", "tileLimit", "resolvedTileLimit", "tileSync", "syncPolicy", "activationPool", "scratchPool", "scratchPolicy", "convSpatialReuse", "preparedExecution", "convDiag", "disableAutoPrepare", "profile", "gpuTiming")
        values = {
            name: self.scratchPolicy if name == "scratchPolicy" else self._values[name]
            for name in names
        }
        return "MatrixManConfig(" + ", ".join(f"{n}={values[n]!r}" for n in names) + ")"

    def __str__(self) -> str:
        lines = ["MatrixMan configuration", "-----------------------"]
        for name in self.keys():
            info = self.metadata(name)
            values = info["options"]
            if isinstance(values, dict):
                value_text = values.get("type", "")
                if values.get("values"):
                    value_text += "|" + "|".join(map(str, values["values"]))
                if values.get("range"):
                    value_text += f" ({values['range']})"
            else:
                value_text = "|".join(str(item) for item in values)
            lines.append(
                f"{name:<28} current={info['current']!s:<12} "
                f"default={info['default']!s:<12} values={value_text}"
            )
        return "\n".join(lines)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()).union(self.keys()))

    def keys(self) -> tuple[str, ...]:
        """Return public configurable attribute names in display order."""
        return tuple(_public_name_for_field(field) for field in _DISPLAY_FIELDS)

    def items(self) -> tuple[tuple[str, Any], ...]:
        """Return public configurable names and their current values."""
        return tuple((name, self.metadata(name)["current"]) for name in self.keys())

    def _current_value(self, name: str) -> Any:
        field = _canonical_field(name)
        if name == "profile":
            return profile_mode()
        if name == "gpu_preference":
            return self.gpu_preference
        if name == "scratch_policy":
            return self.scratchPolicy
        return self._values[field]

    def metadata(self, name: str) -> dict[str, Any]:
        """Return self-describing metadata for one public setting."""
        try:
            field = _canonical_field(name)
        except AttributeError as exc:
            raise ValueError(f"unknown MatrixMan configuration setting: {name}") from exc
        public_name = _public_name_for_field(field)
        if name in {"gpu_preference", "scratch_policy"}:
            public_name = name
        if public_name not in _SETTING_DETAILS:
            raise ValueError(f"unknown MatrixMan configuration setting: {name}")
        category, status, description = _SETTING_DETAILS[public_name]
        options = _metadata_options(public_name, field)
        if public_name == "profile":
            default = "off"
        elif public_name == "gpu_preference":
            default = "integrated"
        elif public_name == "scratch_policy":
            default = "safe"
        else:
            default = _DEFAULTS[field]
        env_name = None if public_name in {"gpu_preference", "scratch_policy"} else next(
            (env for env, mapped in _ENV_FIELDS.items() if mapped == field), None
        )
        lifecycle = (
            "set before backend initialization"
            if field in _RUNTIME_PROTECTED_FIELDS else "may be changed before use"
        )
        return {
            "name": public_name,
            "environment": env_name,
            "type": _metadata_type(public_name, field, options),
            "options": options,
            "default": default,
            "current": self._current_value(public_name),
            "description": description,
            "category": category,
            "lifecycle": lifecycle,
            "status": status,
            "writable": public_name != "scratch_policy",
        }

    def options(self, name: str) -> Any:
        """Return accepted values or constraints for one public setting."""
        return self.metadata(name)["options"]

    def describe(self, name: str | None = None) -> str:
        """Print and return readable metadata for one or all settings."""
        names = (name,) if name is not None else self.keys()
        records = [self.metadata(item) for item in names]
        if name is None:
            records.sort(key=lambda item: (_CATEGORY_ORDER.index(item["category"]), item["name"]))
        lines: list[str] = []
        current_category = None
        for info in records:
            if name is None and info["category"] != current_category:
                if lines:
                    lines.append("")
                lines.extend((info["category"], "=" * len(info["category"])))
                current_category = info["category"]
                lines.append("")
            lines.extend((info["name"], "-" * len(info["name"])))
            lines.append(f"type: {info['type']}")
            options = info["options"]
            if isinstance(options, dict):
                if "values" in options:
                    values = options["values"]
                    lines.append(f"values: {' | '.join(map(str, values)) if isinstance(values, (tuple, list)) else values}")
                if "range" in options:
                    lines.append(f"range: {options['range']}")
            else:
                lines.append(f"values: {' | '.join(map(str, options))}")
            lines.append(f"default: {info['default']}")
            lines.append(f"current: {info['current']}")
            lines.append(f"category: {info['category']}")
            lines.append(f"status: {info['status']}")
            lines.append(f"environment: {info['environment'] or 'none'}")
            lines.append(f"lifecycle: {info['lifecycle']}")
            lines.append(f"description: {info['description']}")
            if not info["writable"]:
                lines.append("writable: no (derived value)")
            lines.append("")
        text = "\n".join(lines).rstrip()
        print(text)
        return text

    def help(self, name: str | None = None) -> str:
        """Alias for :meth:`describe`."""
        return self.describe(name)

    def _ensure_mutable(self, field: str) -> None:
        if field not in _RUNTIME_PROTECTED_FIELDS:
            return
        try:
            from .backend import active_backend
            initialized = active_backend() is not None
        except Exception:
            initialized = False
        if initialized:
            name = _SNAKE_ALIASES.get(field, field)
            raise RuntimeError(
                f"MatrixMan configuration '{name}' must be set before backend "
                "initialization; call matrixman.shutdown() and configure the next "
                "runtime explicitly"
            )

    def source(self, name: str) -> str:
        """Return ``default``, ``environment``, or ``python override``."""
        field = _canonical_field(name)
        if field == "gpu_preference":
            field = "useDGPU"
        if field == "scratchPolicy":
            fields = ("unsafeScratchReuse", "debugFreshScratch", "scratchEpochPool", "debugClearReusedScratch")
            sources = [self._sources.get(item, "default") for item in fields]
            if "python override" in sources:
                return "python override"
            if "environment" in sources:
                return "environment"
            return "default"
        return self._sources.get(field, "default")

    def show(self, *, as_text: bool = False) -> str | None:
        """Print a compact configuration listing with value provenance."""
        lines = ["MatrixMan configuration", "-----------------------"]
        for field in _DISPLAY_FIELDS:
            name = "gpu_preference" if field == "useDGPU" else (
                "scratch_policy" if field == "scratchPolicy" else _SNAKE_ALIASES.get(field, field)
            )
            if field == "profile":
                value = profile_mode()
            elif field == "useDGPU":
                value = self.gpu_preference
            elif field == "scratchPolicy":
                value = self.scratchPolicy
            else:
                value = self._values[field]
            if isinstance(value, bool):
                value = str(value).lower()
            if value is None:
                value = "unset"
            lines.append(f"{name}: {value} [{self.source(field)}]")
        text = "\n".join(lines)
        if as_text:
            return text
        print(text)
        return None

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
    def scratchPolicy(self) -> str:
        return scratch_policy(self._values)

    @property
    def gpu_preference(self) -> str:
        """GPU preference presentation: ``integrated`` or ``discrete``."""
        return "discrete" if self.useDGPU else "integrated"

    @gpu_preference.setter
    def gpu_preference(self, value: Any) -> None:
        text = str(value).strip().lower()
        if isinstance(value, bool):
            selected = value
        elif text in {"discrete", "dgpu", "1", "true", "yes", "on"}:
            selected = True
        elif text in {"integrated", "igpu", "0", "false", "no", "off"}:
            selected = False
        else:
            raise ValueError("gpu_preference must be 'integrated' or 'discrete'")
        self._set("useDGPU", selected)


def _property(field: str, public_name: str | None = None):
    public_name = public_name or field
    doc = _FIELD_DOCS.get(field, f"MatrixMan configuration setting for {public_name}.")
    return property(
        lambda self: self._values[field],
        lambda self, value: self._set(field, value),
        doc=doc,
    )


for _field in _DEFAULTS:
    if _field != "resolvedTileLimit":
        setattr(Configuration, _field, _property(_field))
        _alias = _SNAKE_ALIASES.get(_field)
        if _alias is not None:
            setattr(Configuration, _alias, _property(_field, _alias))
Configuration.resolvedTileLimit = property(lambda self: self._values["resolvedTileLimit"])
Configuration.resolved_tile_limit = property(
    lambda self: self._values["resolvedTileLimit"],
    doc="Resolved integer tile limit after auto selection.",
)
Configuration.scratch_policy = property(
    lambda self: self.scratchPolicy,
    doc="Effective scratch lifetime policy derived from the scratch debug flags.",
)

def _validate_values(values: dict[str, Any]) -> None:
    fresh = bool(values.get("debugFreshScratch"))
    epoch = bool(values.get("scratchEpochPool"))
    unsafe = bool(values.get("unsafeScratchReuse"))
    clear = bool(values.get("debugClearReusedScratch"))
    if fresh and epoch:
        raise ValueError("debugFreshScratch and scratchEpochPool are incompatible")
    if unsafe and (fresh or epoch):
        raise ValueError("unsafeScratchReuse cannot be combined with fresh or epoch scratch policy")
    if clear and not epoch:
        raise ValueError("debugClearReusedScratch requires scratchEpochPool")


def scratch_policy(values: dict[str, Any] | None = None) -> str:
    values = values or config._values
    if values.get("debugFreshScratch"):
        return "fresh"
    if values.get("scratchEpochPool"):
        return "epoch"
    if values.get("unsafeScratchReuse"):
        return "unsafe_reuse"
    return "safe"


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


def _sync_frontend_profile() -> None:
    module = sys.modules.get("drivers.matrixman.frontend_profiling")
    if module is not None:
        module.enabled = bool(config.profile or config.profileDispatch)


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


def profile_mode() -> str:
    """Return the canonical profile mode, with legacy booleans normalized."""
    value = config.profile
    if value is True:
        return "summary"
    if value in {False, None, "0", "off"}:
        return "off"
    return str(value)


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
