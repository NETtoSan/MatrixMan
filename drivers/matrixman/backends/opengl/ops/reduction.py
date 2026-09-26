"""Generic packed GPU reductions for OpenGL MatrixMan tensors."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context
from ....tensor import MatrixManTensor
from ..storage import numel


def normalize_reduction_dims(dim, rank: int, operation: str = "mean") -> tuple[int, ...]:
    """Normalize an ATen reduction dimension argument and validate it."""
    prefix = f"gm45 {operation}.dim"
    if isinstance(dim, int):
        raw = [dim]
    elif isinstance(dim, (list, tuple, torch.Size)):
        raw = list(dim)
    else:
        raise RuntimeError(f"{prefix} requires dim to be an int, list, or tuple")
    if not raw:
        raise RuntimeError(f"{prefix} requires at least one reduction dimension")
    normalized = []
    for value in raw:
        if not isinstance(value, int):
            raise RuntimeError(f"{prefix} dimensions must be integers, got {value!r}")
        current = int(value)
        if current < 0:
            current += rank
        if current < 0 or current >= rank:
            raise RuntimeError(f"{prefix} dimension {value} is out of range for rank {rank}")
        if current in normalized:
            raise RuntimeError(f"{prefix} contains duplicate dimension {value}")
        normalized.append(current)
    return tuple(sorted(normalized))


def _coordinate_expression(dim: int, shape: tuple[int, ...], reduced: set[int]) -> str:
    """Return an input coordinate from generated output/reduction indices."""
    if dim in reduced:
        reduced_dims = [axis for axis in range(len(shape)) if axis in reduced]
        position = reduced_dims.index(dim)
        suffix = numel(tuple(shape[axis] for axis in reduced_dims[position + 1:]))
        source = "reduction_index"
    else:
        output_dims = [axis for axis in range(len(shape)) if axis not in reduced]
        position = output_dims.index(dim)
        suffix = numel(tuple(shape[axis] for axis in output_dims[position + 1:]))
        source = "output_index"
    divisor = max(1, suffix)
    return f"(({source} / {divisor}) - ({source} / {max(1, divisor * shape[dim])}) * {shape[dim]})"


def _reduction_shader_source(params: tuple, operation: str) -> bytes:
    shape, dims, keepdim, input_offset, input_strides, input_tex_w, input_tex_h, out_tex_w = params
    shape = tuple(int(value) for value in shape)
    dims = tuple(int(value) for value in dims)
    reduced = set(dims)
    out_shape = tuple(1 if axis in reduced and keepdim else shape[axis] for axis in range(len(shape)) if keepdim or axis not in reduced)
    output_numel = numel(out_shape)
    reduction_numel = numel(tuple(shape[axis] for axis in dims))
    terms = []
    for axis, stride in enumerate(input_strides):
        terms.append(f"({ _coordinate_expression(axis, shape, reduced) }) * {int(stride)}")
    input_index = " + ".join(terms) or "0"
    if operation not in {"sum", "mean"}:
        raise RuntimeError(f"unsupported reduction operation: {operation}")
    function_name = "sum_at" if operation == "sum" else "mean_at"
    result_expression = "total" if operation == "sum" else f"total / float({reduction_numel})"
    source = f"""
#version 120
uniform sampler2D input_tex;

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

float read_packed(int linear_index)
{{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / {int(input_tex_w)}) * {int(input_tex_w)};
    int y = texel / {int(input_tex_w)};
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float({int(input_tex_w)}), float({int(input_tex_h)}));
    return pick_component(texture2D(input_tex, uv), component);
}}

float {function_name}(int output_index)
{{
    if (output_index >= {output_numel}) return 0.0;
    float total = 0.0;
    for (int reduction_index = 0; reduction_index < {reduction_numel}; ++reduction_index) {{
        int input_index = {int(input_offset)} + {input_index};
        total += read_packed(input_index);
    }}
    return {result_expression};
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * {int(out_tex_w)} + tex_x) * 4;
    gl_FragColor = vec4({function_name}(base), {function_name}(base + 1),
                        {function_name}(base + 2), {function_name}(base + 3));
}}
"""
    return source.encode("ascii")


def _mean_shader_source(params: tuple) -> bytes:
    """Compatibility wrapper preserving the existing mean shader entrypoint."""
    return _reduction_shader_source(params, "mean")


def _reduction_program(operation: str, params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    key = (operation, params)
    if key not in rt.mean_programs:
        diagnostics.trace(f"gm45.compile -> packed {operation} GLSL fragment shader params={params}")
        program = gm.make_program(_reduction_shader_source(params, operation))
        rt.mean_programs[key] = program
        rt.mean_uniforms[key] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.mean_programs[key], rt.mean_uniforms[key]


def _mean_program(params: tuple) -> tuple[int, int]:
    return _reduction_program("mean", params)


def render_mean_dim(args, kwargs) -> "MatrixManTensor":
    return _render_reduction_dim(args, kwargs, "mean")


def render_mean_default(args, kwargs) -> "MatrixManTensor":
    input_tensor = args[0]
    dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 mean requires a MatrixManTensor input")
    return _render_reduction(
        input_tensor, tuple(range(len(input_tensor.shape))), False, dtype, "mean"
    )


def render_sum_default(args, kwargs) -> "MatrixManTensor":
    input_tensor = args[0]
    dtype = kwargs.get("dtype", args[1] if len(args) > 1 else None)
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 sum requires a MatrixManTensor input")
    return _render_reduction(
        input_tensor, tuple(range(len(input_tensor.shape))), False, dtype, "sum"
    )


def render_sum_dim(args, kwargs) -> "MatrixManTensor":
    return _render_reduction_dim(args, kwargs, "sum")


def _render_reduction_dim(args, kwargs, operation: str) -> "MatrixManTensor":
    input_tensor = args[0]
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError(f"gm45 {operation}.dim requires a MatrixManTensor input")
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
    dtype = kwargs.get("dtype", args[3] if len(args) > 3 else None)
    dims = normalize_reduction_dims(dim, len(input_tensor.shape), operation)
    return _render_reduction(input_tensor, dims, keepdim, dtype, operation)


def _render_reduction(input_tensor, dims, keepdim, dtype, operation: str) -> "MatrixManTensor":
    prefix = f"gm45 {operation}"
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError(f"{prefix} requires a MatrixManTensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError(f"{prefix} supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError(f"{prefix} requires packed_rgba input storage")
    operation_context.require_contiguous(input_tensor, prefix)
    if dtype is not None and dtype != torch.float32:
        raise RuntimeError(f"{prefix} supports dtype=None or torch.float32 only")
    keepdim = bool(keepdim)
    shape = tuple(int(value) for value in input_tensor.shape)
    out_shape = tuple(
        1 if keepdim and axis in dims else shape[axis]
        for axis in range(len(shape))
        if keepdim or axis not in dims
    )
    # MatrixMan's allocator names physical packed storage by a non-empty
    # shape, while the wrapper itself can represent a scalar logical result.
    allocation_shape = out_shape or (1,)
    out_owner = operation_context.output_texture(allocation_shape)
    params = (
        shape, dims, keepdim, input_tensor._storage_offset,
        tuple(int(value) for value in input_tensor._logical_strides),
        input_tensor._owner.layout.texture_width,
        input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _reduction_program(operation, params)
    diagnostics.trace(
        f"gm45.kernel -> packed {operation}.dim shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(shape)} "
        f"offset={input_tensor._storage_offset} dims={list(dims)} keepdim={keepdim}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete(f"gm45 {operation}.dim framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after {operation}.dim: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, out_shape)
