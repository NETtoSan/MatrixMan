"""OpenGL-backed MatrixMan tensor ownership boundary."""

from __future__ import annotations

import ctypes
import time
import weakref

import numpy as np
import torch

from . import profiling
from . import gpumatrix as gm
from . import runtime
from .storage import contiguous_strides, max_storage_index, numel


def _impl():
    # Compatibility hook retained for callers that still inspect the old bridge.
    from . import dispatch
    return dispatch


live_textures = weakref.WeakSet()


class _TextureOwner:
    """Own one GL texture and delete it while the active runtime is alive."""

    def __init__(self, texture, layout):
        self.texture = texture
        self.layout = layout
        live_textures.add(self)

    def __del__(self):
        if runtime.is_active() and self.texture:
            texture = ctypes.c_uint(self.texture)
            gm.glDeleteTextures(1, ctypes.byref(texture))
            self.texture = 0


def owner_from_texture(texture, layout):
    """Construct the tensor-owned wrapper for an already allocated texture."""
    return _TextureOwner(texture, layout)


PRIVATEUSE_DEVICE = torch.device("privateuseone:0")


def _validate_cpu_input(tensor: torch.Tensor) -> np.ndarray:
    from . import metadata

    if tensor.device.type != "cpu":
        raise RuntimeError("gm45 transfer only supports source tensors on CPU")
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 only supports torch.float32")
    shape = tuple(int(v) for v in tensor.shape)
    metadata.validate_supported_shape(shape)
    if not tensor.is_contiguous():
        raise RuntimeError("gm45 only supports contiguous tensors")
    return tensor.detach().numpy().astype(np.float32, copy=False)


def texture_from_cpu(tensor: torch.Tensor) -> _TextureOwner:
    from . import diagnostics, resources

    array = _validate_cpu_input(tensor)
    owner = resources.upload_array_to_texture(array)
    if owner.layout.kind == "packed_rgba":
        layout_text = (
            f"packed RGBA atlas {owner.layout.texture_width}x{owner.layout.texture_height}; "
            "linear element i -> texel i//4, component i%4"
        )
    else:
        layout_text = "legacy 2D matrix texture; tensor[y,x] -> texel (x,y).r"
    diagnostics.trace(
        "gm45.upload:\n"
        f"  torch shape {list(tensor.shape)} float32\n"
        f"  -> texture #{owner.texture} {layout_text}"
    )
    return owner


def tensor(data: torch.Tensor | np.ndarray | list[list[float]], *, device=None) -> "Gm45Tensor":
    """Create a gm45 tensor by uploading CPU data into an OpenGL texture."""
    from . import runtime

    runtime.init()
    if device is not None and str(device) not in {"gm45", "gm45:0", "privateuseone", "privateuseone:0"}:
        raise RuntimeError("gm45.tensor only creates tensors on the gm45 device")
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    if data.dtype != torch.float32:
        data = data.to(dtype=torch.float32)
    if not data.is_contiguous():
        data = data.contiguous()
    owner = texture_from_cpu(data)
    return Gm45Tensor._from_owner(owner, tuple(int(v) for v in data.shape))


def randn(*shape: int, seed: int | None = None) -> "Gm45Tensor":
    """Create a gm45 tensor from CPU-generated random float32 data."""
    if len(shape) == 1:
        shape = (shape[0], shape[0])
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    return tensor(torch.randn(shape, dtype=torch.float32, generator=generator))


def to_gm45(data: torch.Tensor) -> "Gm45Tensor":
    return tensor(data)


def is_gm45_tensor(value) -> bool:
    return isinstance(value, Gm45Tensor)


def install_tensor_method() -> None:
    """Convenience method: cpu_tensor.gm45() uploads to a gm45 texture."""
    setattr(torch.Tensor, "gm45", lambda self: to_gm45(self))


class Gm45Tensor(torch.Tensor):
    """The public OpenGL-backed tensor wrapper."""

    @staticmethod
    def __new__(cls, owner, shape, storage_offset=0, logical_strides=None):
        strides = logical_strides or contiguous_strides(shape)
        return torch.Tensor._make_wrapper_subclass(
            cls, shape, strides=strides, dtype=torch.float32,
            layout=torch.strided, device=PRIVATEUSE_DEVICE, requires_grad=False,
        )

    def __init__(self, owner, shape, storage_offset=0, logical_strides=None):
        self._owner = owner
        self._shape = tuple(shape)
        self._storage_offset = int(storage_offset)
        self._logical_strides = tuple(logical_strides or contiguous_strides(self._shape))

    @staticmethod
    def _from_owner(owner, shape, storage_offset=0, logical_strides=None):
        strides = tuple(logical_strides or contiguous_strides(shape))
        if len(strides) != len(shape):
            raise RuntimeError("gm45 tensor logical strides must match shape rank")
        max_index = max_storage_index(shape, strides)
        if storage_offset < 0 or (numel(shape) > 0 and storage_offset + max_index >= owner.layout.numel):
            raise RuntimeError("gm45 tensor view storage offset is outside texture storage")
        return Gm45Tensor(owner, shape, storage_offset, strides)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return _impl().handle_torch_dispatch(cls, func, types, args, kwargs)

    def __repr__(self):
        return (
            f"Gm45Tensor(shape={tuple(self.shape)}, dtype=float32, device={self.device}, "
            f"texture={self._owner.texture}, storage={self._owner.layout.kind}, "
            f"offset={self._storage_offset}, strides={self._logical_strides})"
        )


def readback_tensor(owner: _TextureOwner, shape: tuple[int, ...], storage_offset: int = 0,
                    logical_strides: tuple[int, ...] | None = None) -> torch.Tensor:
    """Synchronize and reconstruct a CPU tensor from packed GL storage."""
    from . import diagnostics, resources, runtime

    rt = runtime.runtime_required()
    readback_started = time.perf_counter()
    if profiling.enabled:
        profiling.counters["readback_calls"] += 1
    diagnostics.kernel_log(f"Readback {diagnostics.shape_text(shape)} -> CPU")
    logical_strides = logical_strides or contiguous_strides(shape)
    diagnostics.trace(
        "gm45.readback:\n"
        f"  texture #{owner.texture} {owner.layout.kind} "
        f"{owner.layout.texture_width}x{owner.layout.texture_height} offset={storage_offset} "
        f"strides={list(logical_strides)}\n"
        f"  -> torch shape {list(shape)}"
    )
    sync_started = time.perf_counter()
    gm.glFinish()
    if profiling.enabled:
        profiling.counters["readback_sync_seconds"] += time.perf_counter() - sync_started
    element_count = numel(shape)
    if element_count == 0:
        if profiling.enabled:
            profiling.counters["readback_total_seconds"] += time.perf_counter() - readback_started
        return torch.empty(shape, dtype=torch.float32, device="cpu")
    if owner.layout.kind == "empty":
        raise RuntimeError("gm45 empty placeholder has no readable texture data")

    if owner.layout.kind == "matrix2d_red" and storage_offset != 0:
        raise RuntimeError("gm45 matrix2d_red readback does not support nonzero storage offsets")
    transfer_started = time.perf_counter()
    raw = resources.read_texture_pixels(owner, rt.fbo)
    if profiling.enabled:
        profiling.counters["readback_transfer_seconds"] += time.perf_counter() - transfer_started
    conversion_started = time.perf_counter()
    if owner.layout.kind == "matrix2d_red":
        result = torch.from_numpy(raw.copy()).reshape(shape)
    else:
        max_index = max_storage_index(shape, logical_strides)
        if storage_offset < 0 or storage_offset + max_index >= owner.layout.numel:
            raise RuntimeError("gm45 readback storage offset is outside texture storage")
        flat_storage = raw.reshape(-1)
        if logical_strides == contiguous_strides(shape):
            flat = flat_storage[storage_offset : storage_offset + element_count].copy()
            result = torch.from_numpy(flat.reshape(shape))
        else:
            result_array = np.empty(shape, dtype=np.float32)
            for index in np.ndindex(shape):
                source_index = storage_offset + sum(i * stride for i, stride in zip(index, logical_strides))
                result_array[index] = flat_storage[source_index]
            result = torch.from_numpy(result_array)
    if profiling.enabled:
        profiling.counters["readback_conversion_seconds"] += time.perf_counter() - conversion_started
        profiling.counters["readback_bytes"] += owner.layout.texture_width * owner.layout.texture_height * 4 * 4
        profiling.counters["readback_total_seconds"] += time.perf_counter() - readback_started
    return result
