"""Small service boundary shared by OpenGL operator implementations.

This module assembles existing runtime, resource, tensor, metadata, kernel,
render, and profiling services.  It intentionally contains no operator math.
"""

from __future__ import annotations

from . import gpumatrix as gm
from . import kernels, metadata, profiling, render, resources, runtime
from .tensor import Gm45Tensor


def gl_runtime():
    return runtime.runtime_required()


def output_texture(shape):
    """Allocate the existing logical packed output through its compatibility boundary."""
    # The allocation helper still includes tensor-specific tracing and remains
    # in implementation.py until allocation is split further.
    from . import implementation
    return implementation._new_empty_packed_texture(shape)


def tensor_from_owner(owner, shape, storage_offset=0, logical_strides=None):
    return Gm45Tensor._from_owner(owner, shape, storage_offset, logical_strides)


def validate_shape(shape) -> None:
    from . import implementation
    implementation._validate_supported_shape(shape)


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
