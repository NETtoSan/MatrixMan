"""Context-local OpenGL texture and parameter resource operations."""

from __future__ import annotations

import ctypes
import time
import weakref
from dataclasses import dataclass

import numpy as np
import torch

from . import gpumatrix as gm
from . import gpu_stress
from . import metadata, profiling
from . import runtime
from .tensor import _TextureOwner, owner_from_texture
from .storage import StorageLayout, matrix_red_rgba, numel, pack_linear_rgba, packed_atlas_size


@dataclass
class ParameterCacheEntry:
    owner: _TextureOwner
    source_ref: weakref.ReferenceType[torch.Tensor]


def create_rgba32f_texture(width: int, height: int, data: np.ndarray | None = None) -> int:
    if profiling.enabled:
        profiling.counters["texture_allocations"] += 1
        if data is not None:
            profiling.counters["texture_uploads"] += 1
            profiling.counters["texture_upload_bytes"] += data.nbytes
    texture = ctypes.c_uint()
    gm.glGenTextures(1, ctypes.byref(texture))
    gm.glBindTexture(gm.GL_TEXTURE_2D, texture.value)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_MIN_FILTER, gm.GL_NEAREST)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_MAG_FILTER, gm.GL_NEAREST)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_WRAP_S, gm.GL_CLAMP_TO_EDGE)
    gm.glTexParameteri(gm.GL_TEXTURE_2D, gm.GL_TEXTURE_WRAP_T, gm.GL_CLAMP_TO_EDGE)
    ptr = data.ctypes.data_as(ctypes.c_void_p) if data is not None else None
    upload_started = time.perf_counter() if profiling.enabled and data is not None else 0.0
    gm.glTexImage2D(gm.GL_TEXTURE_2D, 0, gm.GL_RGBA32F, width, height, 0, gm.GL_RGBA, gm.GL_FLOAT, ptr)
    if profiling.enabled and data is not None:
        profiling.counters["texture_upload_seconds"] += time.perf_counter() - upload_started
    return texture.value


def allocate_packed_texture(shape: tuple[int, ...]) -> tuple[int, StorageLayout]:
    """Allocate an empty packed RGBA texture and return its storage layout."""
    element_count = numel(shape)
    width, height = packed_atlas_size(element_count)
    texture = create_rgba32f_texture(width, height)
    return texture, StorageLayout("packed_rgba", width, height, element_count)


def allocate_matrix_texture(n: int) -> int:
    """Allocate the legacy square RGBA32F matrix texture."""
    return gpu_stress.create_texture(n)


def acquire_scratch_texture(width: int, height: int) -> int:
    """Acquire an empty RGBA32F texture from the runtime's bounded pool."""
    rt = runtime.runtime_required()
    key = (int(width), int(height))
    pooled = rt.scratch_texture_pool.get(key)
    if pooled:
        texture = pooled.pop()
        if profiling.enabled:
            profiling.counters["scratch_texture_reuses"] += 1
        return texture
    if profiling.enabled:
        profiling.counters["scratch_texture_allocations"] += 1
    return create_rgba32f_texture(width, height)


def release_scratch_texture(owner) -> None:
    """Return a no-longer-live scratch owner to the bounded runtime pool."""
    texture = owner.texture
    owner.texture = 0
    if not texture or not runtime.is_active():
        return
    rt = runtime.runtime_required()
    key = (owner.layout.texture_width, owner.layout.texture_height)
    pooled = rt.scratch_texture_pool.setdefault(key, [])
    pooled_count = sum(len(textures) for textures in rt.scratch_texture_pool.values())
    if pooled_count >= runtime._MAX_SCRATCH_TEXTURES:
        texture_id = ctypes.c_uint(texture)
        gm.glDeleteTextures(1, ctypes.byref(texture_id))
        if profiling.enabled:
            profiling.counters["scratch_texture_evictions"] += 1
        return
    pooled.append(texture)
    if profiling.enabled:
        profiling.counters["scratch_texture_releases"] += 1


def read_texture_pixels(owner, fbo):
    """Read the raw contents of an owned GL texture through the shared FBO."""
    if owner.layout.kind == "matrix2d_red":
        return gpu_stress.read_texture(owner.texture, fbo, owner.layout.texture_width)

    gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, fbo.value)
    gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, owner.texture, 0)
    pixels = np.zeros((owner.layout.texture_height, owner.layout.texture_width, 4), dtype=np.float32)
    gm.glReadPixels(
        0,
        0,
        owner.layout.texture_width,
        owner.layout.texture_height,
        gm.GL_RGBA,
        gm.GL_FLOAT,
        pixels.ctypes.data_as(ctypes.c_void_p),
    )
    return pixels


def upload_array_to_texture(array: np.ndarray):
    """Pack a CPU tensor array and create its RGBA32F texture owner."""
    shape = tuple(int(v) for v in array.shape)
    metadata.validate_supported_shape(shape)
    if len(shape) == 2 and shape[0] == shape[1]:
        data, layout = matrix_red_rgba(array)
    else:
        data, layout = pack_linear_rgba(array)
    texture = create_rgba32f_texture(layout.texture_width, layout.texture_height, data)
    return owner_from_texture(texture, layout)


def upload_raw_packed_array(array: np.ndarray, parameter_kind: str = "parameter"):
    """Pack and upload a raw parameter array without changing its layout."""
    if profiling.enabled:
        key = (parameter_kind, int(array.__array_interface__["data"][0]))
        profiling.counters["parameter_uploads"] += 1
        profiling.counters["parameter_upload_bytes"] += array.nbytes
        profiling.parameters[parameter_kind]["count"] += 1
        profiling.parameters[parameter_kind]["bytes"] += array.nbytes
        if key in profiling.parameter_keys:
            profiling.counters["repeated_parameter_uploads"] += 1
            profiling.parameters[parameter_kind]["repeated"] += 1
        profiling.parameter_keys.add(key)
    data, layout = pack_linear_rgba(array)
    texture = create_rgba32f_texture(layout.texture_width, layout.texture_height, data)
    return owner_from_texture(texture, layout)


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
    rt = runtime.runtime_required()
    key = parameter_cache_key(tensor, parameter_kind)
    base_key = key[:-1]
    current_key = rt.parameter_cache_current.get(base_key)
    if current_key is not None and current_key != key and profiling.enabled:
        profiling.counters["parameter_cache_invalidations"] += 1
    entry = rt.parameter_cache.get(key)
    if entry is not None and entry.source_ref() is tensor and entry.owner.texture:
        if profiling.enabled:
            profiling.counters["parameter_cache_hits"] += 1
        return entry.owner
    if profiling.enabled:
        profiling.counters["parameter_cache_misses"] += 1
    array = tensor.detach().numpy().astype(np.float32, copy=False)
    owner = upload_raw_packed_array(array, parameter_kind)
    if len(rt.parameter_cache) >= runtime._MAX_PARAMETER_CACHE_ENTRIES:
        if profiling.enabled:
            profiling.counters["parameter_cache_bypasses"] += 1
        return owner
    rt.parameter_cache[key] = ParameterCacheEntry(owner, weakref.ref(tensor))
    rt.parameter_cache_current[base_key] = key
    return owner
