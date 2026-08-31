"""MatrixMan's backend-neutral public API."""

from .gm45_backend import (
    gpu_postprocess_detection,
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

select_backend()

__all__ = [
    "MatrixManTensor",
    "Gm45Tensor",
    "gpu_postprocess_detection",
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
