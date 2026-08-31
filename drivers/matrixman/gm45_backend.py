"""Deprecated compatibility imports for the former GM45 frontend module."""

from .backend import (
    debug_enabled,
    gpu_postprocess_detection,
    init,
    install_tensor_method,
    is_gm45_tensor,
    is_matrixman_tensor,
    prefer,
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
from .tensor import Gm45Tensor, MatrixManTensor

__all__ = [
    "Gm45Tensor", "MatrixManTensor", "debug_enabled",
    "gpu_postprocess_detection", "init", "install_tensor_method",
    "is_gm45_tensor", "is_matrixman_tensor", "prefer", "profile_enabled",
    "profile_report", "profile_reset", "randn", "reset_unsupported_report",
    "set_trace", "shutdown", "tensor", "to_device", "to_gm45",
    "unsupported_report",
]
