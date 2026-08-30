"""Context-local OpenGL texture and parameter resource operations."""

from __future__ import annotations

import ctypes
import time
import weakref

import numpy as np
import torch

from . import gpumatrix as gm
from . import runtime
from .storage import matrix_red_rgba, pack_linear_rgba


def _impl():
    # Resolve the still-unextracted tensor owner and profiling state at call
    # time, avoiding an import cycle while the extraction remains incremental.
    from . import implementation
    return implementation


def create_rgba32f_texture(width: int, height: int, data: np.ndarray | None = None) -> int:
    impl = _impl()
    if impl._profile_enabled:
        impl._profile_counters["texture_allocations"] += 1
        if data is not None:
            impl._profile_counters["texture_uploads"] += 1
            impl._profile_counters["texture_upload_bytes"] += data.nbytes
    texture = ctypes.c_uint()
    gm.glGenTextures(1, ctypes.byref(texture))
    gm.glBindTexture(gm.GL_TEXTURE_2D, texture.value)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_MIN_FILTER, gm.GL_NEAREST)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_MAG_FILTER, gm.GL_NEAREST)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_WRAP_S, gm.GL_CLAMP_TO_EDGE)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_WRAP_T, gm.GL_CLAMP_TO_EDGE)
    ptr = data.ctypes.data_as(ctypes.c_void_p) if data is not None else None
    upload_started = time.perf_counter() if impl._profile_enabled and data is not None else 0.0
    gm.glTexImage2D(gm.GL_TEXTURE_2D, 0, gm.GL_RGBA32F, width, height, 0, gm.GL_RGBA, gm.GL_FLOAT, ptr)
    if impl._profile_enabled and data is not None:
        impl._profile_counters["texture_upload_seconds"] += time.perf_counter() - upload_started
    return texture.value


def acquire_scratch_texture(width: int, height: int) -> int:
    """Acquire an empty RGBA32F texture from the runtime's bounded pool."""
    impl = _impl()
    rt = runtime.runtime_required()
    key = (int(width), int(height))
    pooled = rt.scratch_texture_pool.get(key)
    if pooled:
        texture = pooled.pop()
        if impl._profile_enabled:
            impl._profile_counters["scratch_texture_reuses"] += 1
        return texture
    if impl._profile_enabled:
        impl._profile_counters["scratch_texture_allocations"] += 1
    return create_rgba32f_texture(width, height)


def release_scratch_texture(owner) -> None:
    """Return a no-longer-live scratch owner to the bounded runtime pool."""
    impl = _impl()
    texture = owner.texture
    owner.texture = 0
    if not texture or impl._runtime is None:
        return
    rt = runtime.runtime_required()
    key = (owner.layout.texture_width, owner.layout.texture_height)
    pooled = rt.scratch_texture_pool.setdefault(key, [])
    pooled_count = sum(len(textures) for textures in rt.scratch_texture_pool.values())
    if pooled_count >= runtime._MAX_SCRATCH_TEXTURES:
        texture_id = ctypes.c_uint(texture)
        gm.glDeleteTextures(1, ctypes.byref(texture_id))
        if impl._profile_enabled:
            impl._profile_counters["scratch_texture_evictions"] += 1
        return
    pooled.append(texture)
    if impl._profile_enabled:
        impl._profile_counters["scratch_texture_releases"] += 1


def upload_array_to_texture(array: np.ndarray):
    """Pack a CPU tensor array and create its RGBA32F texture owner."""
    impl = _impl()
    shape = tuple(int(v) for v in array.shape)
    impl._validate_supported_shape(shape)
    if len(shape) == 2 and shape[0] == shape[1]:
        data, layout = matrix_red_rgba(array)
    else:
        data, layout = pack_linear_rgba(array)
    texture = create_rgba32f_texture(layout.texture_width, layout.texture_height, data)
    return impl._TextureOwner(texture, layout)


def upload_raw_packed_array(array: np.ndarray, parameter_kind: str = "parameter"):
    """Pack and upload a raw parameter array without changing its layout."""
    impl = _impl()
    if impl._profile_enabled:
        key = (parameter_kind, int(array.__array_interface__["data"][0]))
        impl._profile_counters["parameter_uploads"] += 1
        impl._profile_counters["parameter_upload_bytes"] += array.nbytes
        impl._profile_parameters[parameter_kind]["count"] += 1
        impl._profile_parameters[parameter_kind]["bytes"] += array.nbytes
        if key in impl._profile_parameter_keys:
            impl._profile_counters["repeated_parameter_uploads"] += 1
            impl._profile_parameters[parameter_kind]["repeated"] += 1
        impl._profile_parameter_keys.add(key)
    data, layout = pack_linear_rgba(array)
    texture = create_rgba32f_texture(layout.texture_width, layout.texture_height, data)
    return impl._TextureOwner(texture, layout)


def parameter_cache_key(tensor: torch.Tensor, parameter_kind: str) -> tuple:
    """Build the existing identity/version-sensitive parameter cache key."""
    storage = tensor.untyped_storage()
    storage_identity = int(storage._cdata)
    return (
        parameter_kind,
        id(tensor),
        storage_identity,
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(size) for size in tensor.shape),
        str(tensor.dtype),
        int(tensor._version),
    )


def cached_parameter_texture(tensor: torch.Tensor, parameter_kind: str):
    """Return the persistent texture for an eligible Conv2D parameter."""
    impl = _impl()
    rt = runtime.runtime_required()
    key = parameter_cache_key(tensor, parameter_kind)
    base_key = key[:-1]
    current_key = rt.parameter_cache_current.get(base_key)
    if current_key is not None and current_key != key and impl._profile_enabled:
        impl._profile_counters["parameter_cache_invalidations"] += 1
    entry = rt.parameter_cache.get(key)
    if entry is not None and entry.source_ref() is tensor and entry.owner.texture:
        if impl._profile_enabled:
            impl._profile_counters["parameter_cache_hits"] += 1
        return entry.owner
    if impl._profile_enabled:
        impl._profile_counters["parameter_cache_misses"] += 1
    array = tensor.detach().numpy().astype(np.float32, copy=False)
    owner = upload_raw_packed_array(array, parameter_kind)
    if len(rt.parameter_cache) >= runtime._MAX_PARAMETER_CACHE_ENTRIES:
        if impl._profile_enabled:
            impl._profile_counters["parameter_cache_bypasses"] += 1
        return owner
    rt.parameter_cache[key] = impl._ParameterCacheEntry(owner, weakref.ref(tensor))
    rt.parameter_cache_current[base_key] = key
    return owner
