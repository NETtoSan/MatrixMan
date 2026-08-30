"""MatrixMan's backend-neutral public API."""

from .gm45_backend import (
    Gm45Tensor,
    debug_enabled,
    init,
    install_tensor_method,
    is_gm45_tensor,
    profile_enabled,
    profile_report,
    profile_reset,
    randn,
    reset_unsupported_report,
    set_trace,
    shutdown,
    tensor,
    to_gm45,
    unsupported_report,
)
from .selector import select_backend

select_backend()

__all__ = [
    "Gm45Tensor",
    "debug_enabled",
    "init",
    "install_tensor_method",
    "is_gm45_tensor",
    "profile_enabled",
    "profile_report",
    "profile_reset",
    "randn",
    "reset_unsupported_report",
    "set_trace",
    "shutdown",
    "tensor",
    "to_gm45",
    "unsupported_report",
]
