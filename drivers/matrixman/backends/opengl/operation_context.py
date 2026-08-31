"""Small service boundary shared by OpenGL operator implementations.

This module assembles existing runtime, resource, tensor, metadata, kernel,
render, and profiling services.  It intentionally contains no operator math.
"""

from __future__ import annotations

import numpy as np
import torch

from . import gpumatrix as gm
from . import kernels, metadata, profiling, render, resources, runtime
from ...tensor import MatrixManTensor
from .tensor import owner_from_texture


def gl_runtime():
    return runtime.runtime_required()


def output_texture(shape):
    """Allocate an empty packed logical output through the shared services."""
    shape = tuple(int(value) for value in shape)
    validate_shape(shape)
    texture, layout = resources.allocate_packed_texture(shape)
    owner = owner_from_texture(texture, layout)
    from . import diagnostics
    diagnostics.trace(
        f"gm45.texture_alloc -> packed output texture #{texture} shape={list(shape)} "
        f"atlas={layout.texture_width}x{layout.texture_height}"
    )
    return owner


def tensor_from_owner(owner, shape, storage_offset=0, logical_strides=None):
    return MatrixManTensor._from_owner(owner, shape, storage_offset, logical_strides)


def validate_shape(shape) -> None:
    metadata.validate_supported_shape(shape)


def is_scalar_operand(value) -> bool:
    if isinstance(value, MatrixManTensor):
        return False
    if isinstance(value, torch.Tensor):
        return value.device.type == "cpu" and value.numel() == 1 and value.dtype in {
            torch.float32, torch.float64, torch.int64, torch.int32
        }
    return isinstance(value, (int, float, np.integer, np.floating))


def scalar_value(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.numel() != 1:
            raise RuntimeError("gm45 scalar add only supports CPU scalar tensor operands")
        return float(value.item())
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return float(value)


def contiguous(tensor) -> bool:
    return metadata.is_contiguous_logical(tensor)


def require_contiguous(tensor, op_name: str) -> None:
    metadata.require_contiguous_logical(tensor, op_name)


def program(kind: str, size: int):
    return kernels.program(kind, size)


def attach_output(owner, width=None, height=None) -> None:
    render.attach_output(gl_runtime(), owner, width, height)


def draw_fullscreen_quad() -> None:
    render.draw_fullscreen_quad()


def framebuffer_complete(error_message: str) -> None:
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(error_message)


def profile_state():
    return profiling


def upload_array(array):
    return resources.upload_array_to_texture(array)


def upload_raw_parameter(array, parameter_kind="parameter"):
    return resources.upload_raw_packed_array(array, parameter_kind)
