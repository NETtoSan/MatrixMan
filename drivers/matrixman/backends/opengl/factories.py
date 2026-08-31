"""Framework-facing PrivateUse1 factories for the OpenGL backend."""

from __future__ import annotations

import math
import traceback

import torch

from . import diagnostics, gpumatrix as gm, kernels, operation_context, render, runtime
from .storage import StorageLayout, numel
from ...tensor import MatrixManTensor


_aten_privateuse1_lib = None


def register_privateuse_name() -> None:
    """Deprecated compatibility wrapper for shared MatrixMan registration."""
    from ...privateuse import register_privateuse1_backend

    register_privateuse1_backend()


def _new_zero_element_placeholder(shape: tuple[int, ...]):
    if len(shape) == 0:
        raise RuntimeError("gm45 zero-element placeholder does not support scalar shape")
    if len(shape) not in {1, 2, 3, 4}:
        raise RuntimeError("gm45 zero-element placeholder supports only 1D, 2D, 3D, and 4D tensors")
    if numel(shape) != 0:
        raise RuntimeError("gm45 zero-element placeholder requires numel=0")
    if len(shape) == 4 and shape[0] not in {0, 1}:
        raise RuntimeError("gm45 zero-element 4D placeholder supports only batch size 0 or 1")
    layout = StorageLayout("empty", 0, 0, 0)
    diagnostics.trace(f"gm45.empty -> zero-element float32 framework placeholder shape={list(shape)}; no GL texture allocated")
    from .tensor import _TextureOwner
    return _TextureOwner(0, layout)


def _validate_factory_options(op_name: str, dtype, layout, device, pin_memory) -> None:
    if dtype not in {None, torch.float32}:
        raise RuntimeError(f"gm45 {op_name} supports only float32, got {dtype}")
    if layout not in {None, torch.strided}:
        raise RuntimeError(f"gm45 {op_name} supports only strided layout, got {layout}")
    if pin_memory:
        raise RuntimeError(f"gm45 {op_name} does not support pin_memory")
    if device is not None and str(device) not in {"matrixman", "matrixman:0", "privateuseone", "privateuseone:0"}:
        raise RuntimeError(f"gm45 {op_name} got unsupported device {device}")


def empty_gm45(size, *, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    del memory_format
    diagnostics.trace(
        "gm45.empty request:\n"
        f"  size={list(size)} dtype={dtype} layout={layout} device={device} pin_memory={pin_memory}"
    )
    if diagnostics.debug_enabled():
        stack = "".join(traceback.format_stack(limit=12))
        diagnostics.trace("  Python stack:\n" + stack.rstrip())
    shape = tuple(int(v) for v in size)
    if dtype == torch.uint8 and numel(shape) == 0:
        diagnostics.trace("gm45.empty -> zero-sized uint8 framework bookkeeping tensor on CPU; no tensor arithmetic")
        return torch.empty(shape, dtype=torch.uint8, device="cpu")
    if dtype not in {None, torch.float32}:
        raise RuntimeError(f"gm45 empty supports only float32, got {dtype}")
    if layout not in {None, torch.strided}:
        raise RuntimeError(f"gm45 empty supports only strided layout, got {layout}")
    if pin_memory:
        raise RuntimeError("gm45 empty does not support pin_memory")
    if device is not None and str(device) not in {"matrixman", "matrixman:0", "privateuseone", "privateuseone:0"}:
        raise RuntimeError(f"gm45 empty got unsupported device {device}")
    if numel(shape) == 0:
        owner = _new_zero_element_placeholder(shape)
        return MatrixManTensor._from_owner(owner, shape)
    owner = operation_context.output_texture(shape)
    diagnostics.trace(f"gm45.empty -> allocated texture #{owner.texture} shape={list(shape)}")
    return MatrixManTensor._from_owner(owner, shape)


def arange_program(params: tuple) -> int:
    rt = runtime.runtime_required()
    if params not in rt.arange_programs:
        diagnostics.trace(f"gm45.compile -> arange GLSL fragment shader params={params}")
        program = gm.make_program(arange_shader_source(params))
        rt.arange_programs[params] = program
        rt.arange_uniforms[params] = ()
    return rt.arange_programs[params]


def arange_shader_source(params: tuple) -> bytes:
    length, start, step, out_tex_w = params
    source = """
#version 120

float arange_at(int linear_index)
{
    if (linear_index >= __LENGTH__) return 0.0;
    return __START__ + float(linear_index) * __STEP__;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * __OUT_TEX_W__ + tex_x) * 4;
    gl_FragColor = vec4(
        arange_at(base),
        arange_at(base + 1),
        arange_at(base + 2),
        arange_at(base + 3)
    );
}
"""
    replacements = {
        "__LENGTH__": int(length),
        "__START__": kernels.glsl_float(start),
        "__STEP__": kernels.glsl_float(step),
        "__OUT_TEX_W__": int(out_tex_w),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _arange_length(start: float, end: float, step: float) -> int:
    if step <= 0.0:
        raise RuntimeError("gm45 arange currently supports only positive step")
    if end <= start:
        return 0
    return max(0, int(math.ceil((end - start) / step)))


def render_arange(start, end, step, *, dtype=None, layout=None, device=None, pin_memory=None) -> MatrixManTensor:
    _validate_factory_options("arange", dtype, layout, device, bool(pin_memory))
    start_f, end_f, step_f = float(start), float(end), float(step)
    length = _arange_length(start_f, end_f, step_f)
    shape = (length,)
    if length == 0:
        return MatrixManTensor._from_owner(_new_zero_element_placeholder(shape), shape)
    out_owner = operation_context.output_texture(shape)
    params = (length, start_f, step_f, out_owner.layout.texture_width)
    program = arange_program(params)
    rt = runtime.runtime_required()
    diagnostics.trace(
        "gm45.kernel -> arange shader:\n"
        f"  start={start_f:.10g} end={end_f:.10g} step={step_f:.10g} length={length}\n"
        f"  -> output texture #{out_owner.texture} shape={list(shape)} offset=0"
    )
    render.attach_output(rt, out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 arange framebuffer incomplete: 0x{status:04x}")
    gm.glUseProgram(program)
    render.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted arange fullscreen quad, output texture #{out_owner.texture}")
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after arange: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, shape)


def arange_default(end, **kwargs):
    return render_arange(0, end, 1, **kwargs)


def arange_start(start, end, **kwargs):
    return render_arange(start, end, 1, **kwargs)


def arange_start_step(start, end, step=1, **kwargs):
    return render_arange(start, end, step, **kwargs)


def install_privateuse1_factory_kernels() -> None:
    global _aten_privateuse1_lib
    if _aten_privateuse1_lib is not None:
        return
    lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
    lib.impl("empty.memory_format", empty_gm45)
    lib.impl("arange", arange_default)
    lib.impl("arange.start", arange_start)
    lib.impl("arange.start_step", arange_start_step)
    _aten_privateuse1_lib = lib
