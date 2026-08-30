"""OpenGL context-owned runtime state and lifecycle."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from . import gpumatrix as gm


@dataclass
class _GlRuntime:
    window: int
    context: int
    fbo: ctypes.c_uint
    add_programs: dict[int, int]
    matmul_programs: dict[int, int]
    conv_programs: dict[tuple, int]
    conv_tile_programs: dict[tuple[bytes, bytes], int]
    tile_copy_programs: dict[tuple[bytes, bytes], int]
    batchnorm_programs: dict[tuple, int]
    silu_programs: dict[tuple, int]
    packed_add_programs: dict[tuple, int]
    packed_sub_programs: dict[tuple, int]
    packed_strided_add_programs: dict[tuple, int]
    packed_scalar_div_programs: dict[tuple, int]
    packed_broadcast_mul_programs: dict[tuple, int]
    packed_sigmoid_programs: dict[tuple, int]
    scalar_add_programs: dict[tuple, int]
    stack_programs: dict[tuple, int]
    fill_programs: dict[tuple, int]
    cat_programs: dict[tuple, int]
    cat_dim0_2d_programs: dict[tuple, int]
    cat_lastdim_programs: dict[tuple, int]
    cat_dim1_3d_programs: dict[tuple, int]
    maxpool_programs: dict[tuple, int]
    upsample_programs: dict[tuple, int]
    arange_programs: dict[tuple, int]
    softmax_programs: dict[tuple, int]
    add_uniforms: dict[int, tuple[int, int]]
    matmul_uniforms: dict[int, tuple[int, int]]
    conv_uniforms: dict[tuple, tuple[int, int, int]]
    conv_tile_uniforms: dict[tuple[bytes, bytes], tuple[int, int, int]]
    tile_copy_uniforms: dict[tuple[bytes, bytes], int]
    batchnorm_uniforms: dict[tuple, tuple[int, int, int, int, int]]
    silu_uniforms: dict[tuple, int]
    packed_add_uniforms: dict[tuple, tuple[int, int]]
    packed_sub_uniforms: dict[tuple, tuple[int, int]]
    packed_strided_add_uniforms: dict[tuple, tuple[int, int]]
    packed_scalar_div_uniforms: dict[tuple, int]
    packed_broadcast_mul_uniforms: dict[tuple, tuple[int, int]]
    packed_sigmoid_uniforms: dict[tuple, int]
    scalar_add_uniforms: dict[tuple, int]
    stack_uniforms: dict[tuple, tuple[int, ...]]
    fill_uniforms: dict[tuple, tuple]
    cat_uniforms: dict[tuple, tuple[int, ...]]
    cat_dim0_2d_uniforms: dict[tuple, tuple[int, ...]]
    cat_lastdim_uniforms: dict[tuple, tuple[int, ...]]
    cat_dim1_3d_uniforms: dict[tuple, tuple[int, int]]
    maxpool_uniforms: dict[tuple, int]
    upsample_uniforms: dict[tuple, int]
    arange_uniforms: dict[tuple, tuple]
    softmax_uniforms: dict[tuple, int]
    scratch_texture_pool: dict[tuple[int, int], list[int]]
    parameter_cache: dict[tuple, object]
    parameter_cache_current: dict[tuple, tuple]


_runtime: _GlRuntime | None = None
_MAX_SCRATCH_TEXTURES = 32
_MAX_PARAMETER_CACHE_ENTRIES = 256


def init() -> None:
    """Initialize the hidden SDL/OpenGL context and runtime-owned caches."""
    global _runtime
    if _runtime is not None:
        return

    from . import diagnostics, factories

    factories.register_privateuse_name()
    diagnostics.trace("gm45.init -> SDL hidden OpenGL 2.1 context")
    gm.sdl_check(gm.sdl.SDL_Init(gm.SDL_INIT_VIDEO) == 0, "SDL_Init failed")
    gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
    gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_MINOR_VERSION, 1)
    gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_PROFILE_MASK, gm.SDL_GL_CONTEXT_PROFILE_COMPATIBILITY)
    window = gm.sdl.SDL_CreateWindow(
        b"gm45_backend", 0, 0, 64, 64, gm.SDL_WINDOW_OPENGL | gm.SDL_WINDOW_HIDDEN
    )
    gm.sdl_check(bool(window), "SDL_CreateWindow failed")
    context = gm.sdl.SDL_GL_CreateContext(window)
    gm.sdl_check(bool(context), "SDL_GL_CreateContext failed")

    fbo = ctypes.c_uint()
    gm.glGenFramebuffers(1, ctypes.byref(fbo))
    _runtime = _GlRuntime(
        window=window, context=context, fbo=fbo,
        add_programs={}, matmul_programs={}, conv_programs={}, conv_tile_programs={},
        tile_copy_programs={}, batchnorm_programs={}, silu_programs={},
        packed_add_programs={}, packed_sub_programs={}, packed_strided_add_programs={},
        packed_scalar_div_programs={}, packed_broadcast_mul_programs={},
        packed_sigmoid_programs={}, scalar_add_programs={}, stack_programs={},
        fill_programs={}, cat_programs={}, cat_dim0_2d_programs={},
        cat_lastdim_programs={}, cat_dim1_3d_programs={}, maxpool_programs={},
        upsample_programs={}, arange_programs={}, softmax_programs={},
        add_uniforms={}, matmul_uniforms={}, conv_uniforms={}, conv_tile_uniforms={},
        tile_copy_uniforms={}, batchnorm_uniforms={}, silu_uniforms={},
        packed_add_uniforms={}, packed_sub_uniforms={}, packed_strided_add_uniforms={},
        packed_scalar_div_uniforms={}, packed_broadcast_mul_uniforms={},
        packed_sigmoid_uniforms={}, scalar_add_uniforms={}, stack_uniforms={},
        fill_uniforms={}, cat_uniforms={}, cat_dim0_2d_uniforms={},
        cat_lastdim_uniforms={}, cat_dim1_3d_uniforms={}, maxpool_uniforms={},
        upsample_uniforms={}, arange_uniforms={}, softmax_uniforms={},
        scratch_texture_pool={}, parameter_cache={}, parameter_cache_current={},
    )


def shutdown() -> None:
    """Release all GL objects owned by the current runtime."""
    global _runtime
    if _runtime is None:
        return

    from .tensor import live_textures

    for owner in list(live_textures):
        if owner.texture:
            tex = ctypes.c_uint(owner.texture)
            gm.glDeleteTextures(1, ctypes.byref(tex))
            owner.texture = 0
    for textures in _runtime.scratch_texture_pool.values():
        for texture in textures:
            texture_id = ctypes.c_uint(texture)
            gm.glDeleteTextures(1, ctypes.byref(texture_id))
    _runtime.scratch_texture_pool.clear()
    _runtime.parameter_cache.clear()
    _runtime.parameter_cache_current.clear()
    for program in (
        list(_runtime.add_programs.values()) + list(_runtime.matmul_programs.values())
        + list(_runtime.conv_programs.values()) + list(_runtime.conv_tile_programs.values())
        + list(_runtime.tile_copy_programs.values()) + list(_runtime.batchnorm_programs.values())
        + list(_runtime.silu_programs.values()) + list(_runtime.packed_add_programs.values())
        + list(_runtime.packed_sub_programs.values()) + list(_runtime.packed_strided_add_programs.values())
        + list(_runtime.packed_scalar_div_programs.values())
        + list(_runtime.packed_broadcast_mul_programs.values())
        + list(_runtime.packed_sigmoid_programs.values()) + list(_runtime.scalar_add_programs.values())
        + list(_runtime.stack_programs.values()) + list(_runtime.fill_programs.values())
        + list(_runtime.cat_programs.values()) + list(_runtime.cat_dim0_2d_programs.values())
        + list(_runtime.cat_lastdim_programs.values()) + list(_runtime.cat_dim1_3d_programs.values())
        + list(_runtime.maxpool_programs.values()) + list(_runtime.upsample_programs.values())
        + list(_runtime.arange_programs.values()) + list(_runtime.softmax_programs.values())
    ):
        gm.glDeleteProgram(program)
    gm.glDeleteFramebuffers(1, ctypes.byref(_runtime.fbo))
    gm.sdl.SDL_GL_DeleteContext(_runtime.context)
    gm.sdl.SDL_DestroyWindow(_runtime.window)
    gm.sdl.SDL_Quit()
    _runtime = None


def runtime_required() -> _GlRuntime:
    init()
    assert _runtime is not None
    return _runtime


def is_active() -> bool:
    return _runtime is not None
