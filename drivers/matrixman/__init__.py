"""MatrixMan's backend-neutral public API."""

import sys
import types

from . import config as _config_module

from .backend import (
    gpu_postprocess_detection,
    prefer,
    debug_enabled,
    init,
    install_tensor_method,
    profile_enabled,
    profile_report,
    profile_reset,
    randn,
    reset_unsupported_report,
    set_trace,
    shutdown,
    tensor,
    to_device,
    to_gm45,
    unsupported_report,
)
from .tensor import Gm45Tensor, MatrixManTensor, is_gm45_tensor, is_matrixman_tensor
from .selector import select_backend

config = _config_module.config
set_profiling = _config_module.set_profiling
set_tracing = _config_module.set_trace


class _MatrixManModule(types.ModuleType):
    """Make public configuration assignments update central configuration."""

    def __getattribute__(self, name):
        if name == "profiling":
            return _config_module.profiling_enabled(legacy_cuda=True)
        if name == "trace":
            return _config_module.trace_enabled()
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "profiling":
            _config_module.set_profiling(value)
            return
        if name == "trace":
            _config_module.set_trace(value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _MatrixManModule

__all__ = [
    "MatrixManTensor",
    "Gm45Tensor",
    "gpu_postprocess_detection",
    "prefer",
    "config",
    "profiling",
    "set_profiling",
    "set_tracing",
    "trace",
    "debug_enabled",
    "init",
    "install_tensor_method",
    "is_matrixman_tensor",
    "is_gm45_tensor",
    "profile_enabled",
    "profile_report",
    "profile_reset",
    "randn",
    "reset_unsupported_report",
    "set_trace",
    "shutdown",
    "tensor",
    "to_device",
    "to_gm45",
    "unsupported_report",
]
