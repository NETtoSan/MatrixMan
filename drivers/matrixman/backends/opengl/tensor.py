"""OpenGL-backed MatrixMan tensor ownership boundary."""

from __future__ import annotations

import ctypes

import torch

from . import gpumatrix as gm
from .storage import contiguous_strides, max_storage_index, numel


def _impl():
    # Tensor classes remain in implementation.py until dispatch is extracted.
    from . import implementation
    return implementation


class _TextureOwner:
    """Own one GL texture and delete it while the active runtime is alive."""

    def __init__(self, texture, layout):
        self.texture = texture
        self.layout = layout
        _impl()._live_textures.add(self)

    def __del__(self):
        impl = _impl()
        if impl._runtime is not None and self.texture:
            texture = ctypes.c_uint(self.texture)
            gm.glDeleteTextures(1, ctypes.byref(texture))
            self.texture = 0


PRIVATEUSE_DEVICE = torch.device("privateuseone:0")


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
        # Dispatch remains in implementation.py until its dedicated extraction.
        return _impl()._DispatchBridge.__torch_dispatch__.__func__(cls, func, types, args, kwargs)

    def __repr__(self):
        return (
            f"Gm45Tensor(shape={tuple(self.shape)}, dtype=float32, device={self.device}, "
            f"texture={self._owner.texture}, storage={self._owner.layout.kind}, "
            f"offset={self._storage_offset}, strides={self._logical_strides})"
        )
