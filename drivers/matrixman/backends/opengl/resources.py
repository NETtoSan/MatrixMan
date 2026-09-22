"""Context-local OpenGL texture and parameter resource operations."""

from __future__ import annotations

import ctypes
import inspect
import time
import weakref
from dataclasses import dataclass

import numpy as np
import torch

from . import gpumatrix as gm
from . import gpu_stress
from . import metadata, profiling
from . import runtime
from ...config import config
from .tensor import _TextureOwner, owner_from_texture
from .storage import StorageLayout, matrix_red_rgba, numel, pack_linear_rgba, packed_atlas_size


@dataclass
class ParameterCacheEntry:
    owner: _TextureOwner
    source_ref: weakref.ReferenceType[torch.Tensor]


def _record_parameter_resource_count(rt) -> None:
    if not profiling.enabled:
        return
    current = sum(1 for entry in rt.parameter_cache.values() if entry.owner.texture)
    profiling.counters["persistent_parameter_resources"] = current
    profiling.counters["persistent_parameter_resources_peak"] = max(
        int(profiling.counters["persistent_parameter_resources_peak"]), current
    )


def _parameter_source_gone(_reference, key: tuple) -> None:
    """Drop a cached parameter texture when its source tensor is destroyed."""
    if not runtime.is_active():
        return
    rt = runtime.runtime_required()
    entry = rt.parameter_cache.pop(key, None)
    if entry is None:
        return
    for base_key, current_key in list(rt.parameter_cache_current.items()):
        if current_key == key:
            rt.parameter_cache_current.pop(base_key, None)
    if entry.owner.texture:
        texture = ctypes.c_uint(entry.owner.texture)
        gm.glDeleteTextures(1, ctypes.byref(texture))
        entry.owner.texture = 0
    _record_parameter_resource_count(rt)


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


def acquire_scratch_texture(
    width: int,
    height: int,
    operation: str | None = None,
    tile: int | None = None,
) -> int:
    """Acquire an empty RGBA32F texture from the runtime's bounded pool."""
    rt = runtime.runtime_required()
    key = (int(width), int(height))
    policy = config.scratchPolicy
    pooled = None if policy == "fresh" else rt.scratch_texture_pool.get(key)
    if profiling.enabled:
        profiling.counters["scratch_pool_acquires"] += 1
    if pooled:
        if policy == "safe":
            _finish_before_pool_reuse("scratch_reuse")
        texture = pooled.pop()
        if profiling.enabled:
            profiling.counters["scratch_texture_reuses"] += 1
            profiling.counters["scratch_pool_reuses"] += 1
            if policy == "epoch":
                profiling.counters["scratch_cross_epoch_reuses"] += 1
        _scratch_checkout(rt, texture, key, source="reuse", operation=operation, tile=tile)
        _record_scratch_pool_stats(rt)
        return texture
    if profiling.enabled:
        profiling.counters["scratch_texture_allocations"] += 1
        profiling.counters["scratch_pool_allocations"] += 1
        if policy == "fresh":
            profiling.counters["scratch_fresh_allocations"] += 1
    texture = create_rgba32f_texture(width, height)
    _scratch_checkout(rt, texture, key, source="fresh", operation=operation, tile=tile)
    _record_scratch_pool_stats(rt)
    return texture


def release_scratch_texture(owner) -> None:
    """Return a no-longer-live scratch owner to the bounded runtime pool."""
    texture = owner.texture
    _scratch_release_metadata(owner, texture)
    owner.texture = 0
    if not texture or not runtime.is_active():
        return
    rt = runtime.runtime_required()
    key = (owner.layout.texture_width, owner.layout.texture_height)
    policy = config.scratchPolicy
    if policy in {"fresh", "epoch"}:
        record = rt.scratch_texture_records.get(int(texture)) if _scratch_debug_enabled() else None
        if record is not None and record["state"] != "retired":
            raise AssertionError(f"scratch texture {texture} expected retired, got {record['state']}")
        retired_bytes = int(owner.layout.texture_width) * int(owner.layout.texture_height) * 4 * 4
        rt.retired_scratch_textures.append((int(texture), retired_bytes, key))
        rt.retired_scratch_bytes += retired_bytes
        if profiling.enabled:
            profiling.counters["scratch_texture_releases"] += 1
            if policy == "fresh":
                profiling.counters["scratch_fresh_releases"] += 1
            if policy == "epoch":
                profiling.counters["scratch_epoch_releases"] += 1
            profiling.counters["scratch_retired_textures"] += 1
            profiling.counters["scratch_retired_bytes"] += retired_bytes
            profiling.counters["scratch_retired_current"] = len(rt.retired_scratch_textures)
            profiling.counters["scratch_retired_bytes_current"] = rt.retired_scratch_bytes
        if record is not None:
            print(f"SCRATCH retire tex={texture} gen={record['generation']} retired -> held")
        return
    pooled = rt.scratch_texture_pool.setdefault(key, [])
    pooled_count = sum(len(textures) for textures in rt.scratch_texture_pool.values())
    if pooled_count >= runtime._MAX_SCRATCH_TEXTURES:
        _finish_before_pool_reuse("scratch_eviction")
        if _scratch_debug_enabled():
            record = rt.scratch_texture_records.get(int(texture))
            if record is not None:
                if record["state"] != "pooled":
                    raise AssertionError(f"scratch texture {texture} evicted from {record['state']}")
                record["state"] = "deleted"
                print(
                    f"SCRATCH delete tex={texture} gen={record['generation']} "
                    f"key={record['pool_key'][0]}x{record['pool_key'][1]}"
                )
        texture_id = ctypes.c_uint(texture)
        gm.glDeleteTextures(1, ctypes.byref(texture_id))
        if profiling.enabled:
            profiling.counters["scratch_texture_evictions"] += 1
            profiling.counters["scratch_pool_evictions"] += 1
        return
    pooled.append(texture)
    if profiling.enabled:
        profiling.counters["scratch_texture_releases"] += 1
        profiling.counters["scratch_pool_releases"] += 1
    _record_scratch_pool_stats(rt)


def _scratch_debug_enabled() -> bool:
    return config.scratchPolicy != "safe"


def _record_scratch_pool_stats(rt) -> None:
    if not profiling.enabled:
        return
    count = 0
    bytes_total = 0
    for key, textures in rt.scratch_texture_pool.items():
        count += len(textures)
        bytes_total += len(textures) * int(key[0]) * int(key[1]) * 4 * 4
    profiling.counters["scratch_pool_current"] = count
    profiling.counters["scratch_pool_bytes"] = bytes_total


def _scratch_site() -> str:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        if caller is None:
            return "unknown"
        return f"{caller.f_code.co_filename.rsplit('/', 1)[-1]}:{caller.f_lineno}"
    finally:
        del frame


def _scratch_record(rt, texture: int, key: tuple[int, int]) -> dict:
    record = rt.scratch_texture_records.get(int(texture))
    if record is None:
        record = {
            "texture": int(texture), "generation": 0, "width": int(key[0]),
            "height": int(key[1]), "pool_key": tuple(key), "state": "deleted",
            "owner": None, "operation": None, "tile": None,
            "last_release_site": None, "next_acquire_site": None,
        }
        rt.scratch_texture_records[int(texture)] = record
    return record


def _scratch_checkout(
    rt,
    texture: int,
    key: tuple[int, int],
    *,
    source: str,
    operation: str | None,
    tile: int | None,
) -> None:
    if not _scratch_debug_enabled():
        return
    record = _scratch_record(rt, texture, key)
    if record["state"] == "checked_out":
        raise AssertionError(f"scratch texture {texture} acquired while checked_out")
    if record["state"] == "retired":
        if profiling.enabled:
            profiling.counters["scratch_same_epoch_reuse_attempts"] += 1
        raise AssertionError(f"retired scratch texture {texture} acquired before safe-point promotion")
    if record["state"] == "deleted":
        # A fresh GL name has no prior record; a recorded deleted name must
        # never re-enter the pool or be checked out again.
        if record["generation"]:
            raise AssertionError(f"deleted scratch texture {texture} acquired")
    record["generation"] += 1
    record["width"], record["height"] = int(key[0]), int(key[1])
    record["pool_key"] = tuple(key)
    record["state"] = "checked_out"
    record["owner"] = None
    record["operation"] = operation
    record["tile"] = tile
    record["source"] = source
    record["next_acquire_site"] = _scratch_site()
    print(
        f"SCRATCH acquire tex={texture} gen={record['generation']} "
        f"source={source} key={key[0]}x{key[1]} op={operation or '-'} "
        f"tile={tile if tile is not None else '-'} site={record['next_acquire_site']}"
    )


def _scratch_assign_owner(owner, operation: str | None, tile: int | None) -> None:
    if not _scratch_debug_enabled() or not owner.texture:
        return
    rt = runtime.runtime_required()
    record = rt.scratch_texture_records.get(int(owner.texture))
    if record is None or record["state"] != "checked_out":
        raise AssertionError(f"scratch texture {owner.texture} missing checked_out record")
    record["owner"] = id(owner)
    record["operation"] = operation
    record["tile"] = tile


def _scratch_release_metadata(owner, texture: int) -> None:
    if not _scratch_debug_enabled() or not texture:
        return
    rt = runtime.runtime_required()
    record = rt.scratch_texture_records.get(int(texture))
    if record is None:
        raise AssertionError(f"scratch texture {texture} released without a record")
    if record["state"] != "checked_out":
        raise AssertionError(f"scratch texture {texture} released from {record['state']}")
    next_state = "retired" if config.scratchPolicy in {"fresh", "epoch"} else "pooled"
    print(
        f"SCRATCH state tex={texture} checked_out -> {next_state} "
        f"gen={record['generation']}"
    )
    record["state"] = next_state
    record["owner"] = None
    record["last_release_site"] = _scratch_site()
    print(
        f"SCRATCH release tex={texture} gen={record['generation']} "
        f"op={record['operation'] or 'unknown'} tile={record['tile'] if record['tile'] is not None else '-'} "
        f"site={record['last_release_site']}"
    )


def reclaim_retired_scratch_textures(rt=None) -> None:
    """Delete fresh-scratch diagnostic textures after a caller's global wait."""
    policy = config.scratchPolicy
    if policy not in {"fresh", "epoch"}:
        return
    rt = rt or runtime.runtime_required()
    if not rt.retired_scratch_textures:
        return
    reclaimed = len(rt.retired_scratch_textures)
    reclaimed_bytes = rt.retired_scratch_bytes
    for texture, _size, key in rt.retired_scratch_textures:
        record = rt.scratch_texture_records.get(int(texture)) if _scratch_debug_enabled() else None
        if record is not None:
            if record["state"] != "retired":
                raise AssertionError(f"scratch texture {texture} reclaimed from {record['state']}")
        if policy == "epoch":
            pooled_count = sum(len(textures) for textures in rt.scratch_texture_pool.values())
            if pooled_count < runtime._MAX_SCRATCH_TEXTURES:
                rt.scratch_texture_pool.setdefault(key, []).append(texture)
                if record is not None:
                    record["state"] = "pooled"
                if profiling.enabled:
                    profiling.counters["scratch_retired_promotions"] += 1
                if record is not None:
                    print(f"SCRATCH reclaim tex={texture} retired -> pooled")
                continue
            if profiling.enabled:
                profiling.counters["scratch_retired_evictions"] += 1
        texture_id = ctypes.c_uint(texture)
        gm.glDeleteTextures(1, ctypes.byref(texture_id))
        if record is not None:
            record["state"] = "deleted"
            print(f"SCRATCH reclaim tex={texture} retired -> deleted")
    rt.retired_scratch_textures.clear()
    rt.retired_scratch_bytes = 0
    _record_scratch_pool_stats(rt)
    if profiling.enabled:
        profiling.counters["scratch_retired_reclaims"] += reclaimed
        profiling.counters["scratch_retired_reclaim_bytes"] += reclaimed_bytes
        profiling.counters["scratch_retired_current"] = 0
        profiling.counters["scratch_retired_bytes_current"] = 0


def scratch_assign_owner(owner, operation: str | None, tile: int | None) -> None:
    """DEBUG-only metadata hook for tiled scratch ownership."""
    _scratch_assign_owner(owner, operation, tile)


def scratch_was_reused(texture: int) -> bool:
    """Return whether this checkout came from the reusable scratch pool."""
    if not _scratch_debug_enabled():
        return False
    rt = runtime.runtime_required()
    record = rt.scratch_texture_records.get(int(texture))
    return bool(record and record.get("source") == "reuse")


def _activation_pool_key(
    rt,
    width: int,
    height: int,
    *,
    target=gm.GL_TEXTURE_2D,
    internal_format=gm.GL_RGBA32F,
    dtype="float32",
    layout_kind="packed_rgba",
) -> tuple:
    """Identify the exact physical storage accepted by the activation pool."""
    return (
        "opengl",
        id(rt),
        target,
        internal_format,
        dtype,
        layout_kind,
        int(width),
        int(height),
    )


def _activation_texture_bytes(key: tuple) -> int:
    return int(key[-2]) * int(key[-1]) * 4 * 4


def _finish_before_pool_reuse(resource_kind: str) -> None:
    """Complete prior GL users before a pooled texture becomes reusable.

    A texture owner can become unreachable while its draw is still queued on
    the compatibility driver.  Reusing that texture name as a later FBO or
    sampler resource is therefore a real lifetime boundary, especially on
    old Intel drivers.  Keep the wait at the pool boundary rather than after
    every operator or Conv.  This is synchronization only; it does not add a
    readback or a CPU arithmetic path.
    """
    with profiling.stage("pool_reuse_synchronization"):
        profiling.sync_finish(resource_kind)
    if profiling.enabled:
        counter_name = {
            "activation_reuse": "activation_pool_reuse_sync_calls",
            "activation_eviction": "activation_pool_eviction_sync_calls",
            "scratch_reuse": "scratch_pool_reuse_sync_calls",
            "scratch_eviction": "scratch_pool_eviction_sync_calls",
        }.get(resource_kind)
        if counter_name is not None:
            profiling.counters[counter_name] += 1


def _record_activation_pool_stats(rt) -> None:
    current_count = len(rt.activation_texture_pool_order)
    current_bytes = int(rt.activation_texture_pool_bytes)
    rt.activation_texture_pool_peak_count = max(
        int(rt.activation_texture_pool_peak_count), current_count
    )
    rt.activation_texture_pool_peak_bytes = max(
        int(rt.activation_texture_pool_peak_bytes), current_bytes
    )
    if profiling.enabled:
        profiling.counters["activation_pool_current_textures"] = current_count
        profiling.counters["activation_pool_current_bytes"] = current_bytes
        profiling.counters["activation_pool_peak_textures"] = rt.activation_texture_pool_peak_count
        profiling.counters["activation_pool_peak_bytes"] = rt.activation_texture_pool_peak_bytes


def acquire_activation_texture(shape: tuple[int, ...]):
    """Acquire a poolable packed RGBA32F owner for a fresh activation."""
    rt = runtime.runtime_required()
    shape = tuple(int(value) for value in shape)
    element_count = numel(shape)
    width, height = packed_atlas_size(element_count)
    with profiling.stage("activation_pool_key_construction"):
        key = _activation_pool_key(rt, width, height)
    with profiling.stage("activation_pool_acquire_lookup"):
        pooled = rt.activation_texture_pool.get(key)
    if profiling.enabled:
        profiling.counters["activation_pool_acquires"] += 1
    if pooled:
        with profiling.stage("activation_pool_resource_retrieval"):
            _finish_before_pool_reuse("activation_reuse")
            texture = pooled.pop()
            rt.activation_texture_pool_order.remove((key, texture))
            rt.activation_texture_pool_bytes -= _activation_texture_bytes(key)
        if profiling.enabled:
            profiling.counters["scratch_texture_reuses"] += 1
            profiling.counters["activation_pool_reuses"] += 1
    else:
        texture = create_rgba32f_texture(width, height)
        if profiling.enabled:
            profiling.counters["scratch_texture_allocations"] += 1
            profiling.counters["activation_pool_allocations"] += 1
    _record_activation_pool_stats(rt)
    return owner_from_texture(
        texture,
        StorageLayout("packed_rgba", width, height, element_count),
        pool_key=key,
    )


def release_activation_texture(owner) -> None:
    """Return a dead activation owner to the bounded context-local pool."""
    with profiling.stage("activation_pool_release_bookkeeping"):
        texture = owner.texture
        owner.texture = 0
        if not texture or not runtime.is_active():
            return
        rt = runtime.runtime_required()
        key = getattr(owner, "_pool_key", None)
        if not key or key[1] != id(rt):
            # Never issue a delete through a different OpenGL context.
            return
        texture_bytes = _activation_texture_bytes(key)
        if (
            len(rt.activation_texture_pool_order) >= runtime._MAX_ACTIVATION_TEXTURES
            or rt.activation_texture_pool_bytes + texture_bytes > runtime._MAX_ACTIVATION_POOL_BYTES
        ):
            _finish_before_pool_reuse("activation_eviction")
            texture_id = ctypes.c_uint(texture)
            gm.glDeleteTextures(1, ctypes.byref(texture_id))
            if profiling.enabled:
                profiling.counters["scratch_texture_evictions"] += 1
                profiling.counters["activation_pool_evictions"] += 1
            _record_activation_pool_stats(rt)
            return
        rt.activation_texture_pool.setdefault(key, []).append(texture)
        rt.activation_texture_pool_order.append((key, texture))
        rt.activation_texture_pool_bytes += texture_bytes
        if profiling.enabled:
            profiling.counters["scratch_texture_releases"] += 1
            profiling.counters["activation_pool_releases"] += 1
        _record_activation_pool_stats(rt)


def clear_activation_pool(rt=None) -> None:
    """Delete all pooled activation textures while their context is active."""
    rt = rt or runtime.runtime_required()
    for textures in rt.activation_texture_pool.values():
        for texture in textures:
            texture_id = ctypes.c_uint(texture)
            gm.glDeleteTextures(1, ctypes.byref(texture_id))
    rt.activation_texture_pool.clear()
    rt.activation_texture_pool_order.clear()
    rt.activation_texture_pool_bytes = 0
    _record_activation_pool_stats(rt)


def activation_pool_stats() -> dict[str, int | float]:
    """Return current/peak activation-pool usage and reuse effectiveness."""
    if not runtime.is_active():
        raise RuntimeError("OpenGL runtime is not active")
    rt = runtime.runtime_required()
    _record_activation_pool_stats(rt)
    allocations = int(profiling.counters["activation_pool_allocations"])
    reuses = int(profiling.counters["activation_pool_reuses"])
    attempts = allocations + reuses
    return {
        "current_pooled_textures": len(rt.activation_texture_pool_order),
        "peak_pooled_textures": int(rt.activation_texture_pool_peak_count),
        "current_pooled_bytes": int(rt.activation_texture_pool_bytes),
        "peak_pooled_bytes": int(rt.activation_texture_pool_peak_bytes),
        "allocation_avoidance_rate": reuses / attempts if attempts else 0.0,
    }


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


def upload_raw_packed_array(
    array: np.ndarray,
    parameter_kind: str = "parameter",
    *,
    operation: str | None = None,
    parameter_role: str | None = None,
    cacheable: bool = False,
):
    """Pack and upload a raw parameter array without changing its layout."""
    profiling.record_parameter_upload(
        array,
        parameter_kind,
        operation=operation,
        parameter_role=parameter_role,
        cacheable=cacheable,
    )
    data, layout = pack_linear_rgba(array)
    texture = create_rgba32f_texture(layout.texture_width, layout.texture_height, data)
    return owner_from_texture(texture, layout)


def parameter_cache_key(tensor: torch.Tensor, parameter_kind: str) -> tuple:
    """Build the existing identity/version-sensitive parameter cache key."""
    storage = tensor.untyped_storage()
    storage_identity = int(storage._cdata)
    return (
        parameter_kind,
        "opengl",
        str(tensor.device),
        id(tensor),
        storage_identity,
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(int(size) for size in tensor.shape),
        str(tensor.dtype),
        int(tensor._version),
    )


def cached_parameter_texture(
    tensor: torch.Tensor,
    parameter_kind: str,
    *,
    operation: str | None = None,
    parameter_role: str | None = None,
):
    """Return the persistent texture for an eligible Conv2D parameter."""
    rt = runtime.runtime_required()
    role = parameter_role or parameter_kind
    with profiling.stage("parameter_cache_key_construction"):
        key = parameter_cache_key(tensor, parameter_kind)
    with profiling.stage("parameter_cache_version_mutation_check"):
        base_key = key[:-1]
        current_key = rt.parameter_cache_current.get(base_key)
    if current_key is not None and current_key != key:
        if profiling.enabled:
            profiling.counters["parameter_cache_invalidations"] += 1
        profiling.record_parameter_cache_event(operation, role, tensor.shape, "invalidation")
        stale = rt.parameter_cache.pop(current_key, None)
        if stale is not None and stale.owner.texture:
            texture = ctypes.c_uint(stale.owner.texture)
            gm.glDeleteTextures(1, ctypes.byref(texture))
            stale.owner.texture = 0
        _record_parameter_resource_count(rt)
    with profiling.stage("parameter_cache_dict_lookup"):
        entry = rt.parameter_cache.get(key)
    if entry is not None and entry.source_ref() is tensor and entry.owner.texture:
        if profiling.enabled:
            profiling.counters["parameter_cache_hits"] += 1
        profiling.record_parameter_cache_event(operation, role, tensor.shape, "hit")
        with profiling.stage("parameter_cache_resource_retrieval"):
            return entry.owner
    if profiling.enabled:
        profiling.counters["parameter_cache_misses"] += 1
    profiling.record_parameter_cache_event(operation, role, tensor.shape, "miss")
    array = tensor.detach().numpy().astype(np.float32, copy=False)
    can_cache = len(rt.parameter_cache) < runtime._MAX_PARAMETER_CACHE_ENTRIES
    owner = upload_raw_packed_array(
        array,
        parameter_kind,
        operation=operation,
        parameter_role=role,
        cacheable=can_cache,
    )
    if not can_cache:
        if profiling.enabled:
            profiling.counters["parameter_cache_bypasses"] += 1
        return owner
    source_ref = weakref.ref(tensor, lambda reference: _parameter_source_gone(reference, key))
    rt.parameter_cache[key] = ParameterCacheEntry(owner, source_ref)
    rt.parameter_cache_current[base_key] = key
    _record_parameter_resource_count(rt)
    return owner


def persistent_parameter_resources() -> int:
    """Return the number of live cached parameter textures in this context."""
    if not runtime.is_active():
        raise RuntimeError("OpenGL runtime is not active")
    rt = runtime.runtime_required()
    current = sum(1 for entry in rt.parameter_cache.values() if entry.owner.texture)
    _record_parameter_resource_count(rt)
    return current
