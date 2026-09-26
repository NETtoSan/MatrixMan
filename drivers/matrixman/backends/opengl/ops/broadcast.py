"""GPU materialization of broadcasted MatrixMan tensors."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context
from ..storage import contiguous_strides, numel
from ....tensor import MatrixManTensor


def _transpose_shader_source(params: tuple) -> bytes:
    rows, cols, source_offset, source_stride0, source_stride1, source_tex_w, source_tex_h, out_tex_w = params
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
    int x = texel - (texel / {int(source_tex_w)}) * {int(source_tex_w)};
    int y = texel / {int(source_tex_w)};
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float({int(source_tex_w)}), float({int(source_tex_h)}));
    return pick_component(texture2D(input_tex, uv), component);
}}

float transpose_at(int output_index)
{{
    if (output_index >= {int(rows * cols)}) return 0.0;
    int output_row = output_index / {int(rows)};
    int output_col = output_index - output_row * {int(rows)};
    int source_index = {int(source_offset)} + output_col * {int(source_stride0)} +
                       output_row * {int(source_stride1)};
    return read_packed(source_index);
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * {int(out_tex_w)} + tex_x) * 4;
    gl_FragColor = vec4(transpose_at(base), transpose_at(base + 1),
                        transpose_at(base + 2), transpose_at(base + 3));
}}
"""
    return source.encode("ascii")


def _transpose_program(params: tuple) -> int:
    rt = operation_context.gl_runtime()
    key = ("transpose", params)
    if key not in rt.broadcast_programs:
        diagnostics.trace(f"gm45.compile -> transpose GLSL fragment shader params={params}")
        rt.broadcast_programs[key] = gm.make_program(_transpose_shader_source(params))
        rt.broadcast_uniforms[key] = ()
    return rt.broadcast_programs[key]


def _broadcast_shader_source(params: tuple) -> bytes:
    source_shape, source_strides, requested_shape, source_offset, source_tex_w, source_tex_h, out_tex_w = params
    source_shape = tuple(int(value) for value in source_shape)
    source_strides = tuple(int(value) for value in source_strides)
    requested_shape = tuple(int(value) for value in requested_shape)
    source_rank = len(source_shape)
    output_rank = len(requested_shape)
    rank_offset = output_rank - source_rank
    output_numel = numel(requested_shape) if requested_shape else 1

    terms = []
    for source_index, source_size in enumerate(source_shape):
        output_index = source_index + rank_offset
        suffix = numel(requested_shape[output_index + 1:])
        output_size = requested_shape[output_index]
        coordinate = (
            "0"
            if source_size == 1
            else f"((output_index / {max(1, suffix)}) - "
                 f"(output_index / {max(1, suffix * output_size)}) * {output_size})"
        )
        terms.append(f"({coordinate}) * {source_strides[source_index]}")
    source_index_expression = " + ".join(terms) or "0"

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
    int x = texel - (texel / {int(source_tex_w)}) * {int(source_tex_w)};
    int y = texel / {int(source_tex_w)};
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float({int(source_tex_w)}), float({int(source_tex_h)}));
    return pick_component(texture2D(input_tex, uv), component);
}}

float broadcast_at(int output_index)
{{
    if (output_index >= {int(output_numel)}) return 0.0;
    int source_index = {int(source_offset)} + {source_index_expression};
    return read_packed(source_index);
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * {int(out_tex_w)} + tex_x) * 4;
    gl_FragColor = vec4(broadcast_at(base), broadcast_at(base + 1),
                        broadcast_at(base + 2), broadcast_at(base + 3));
}}
"""
    return source.encode("ascii")


def _broadcast_program(params: tuple) -> int:
    rt = operation_context.gl_runtime()
    cache = getattr(rt, "broadcast_programs", None)
    if cache is None:
        cache = rt.broadcast_programs = {}
        rt.broadcast_uniforms = {}
    if params not in cache:
        diagnostics.trace(f"gm45.compile -> broadcast/expand GLSL fragment shader params={params}")
        cache[params] = gm.make_program(_broadcast_shader_source(params))
        rt.broadcast_uniforms[params] = ()
    return cache[params]


def render_expand(args, kwargs) -> "MatrixManTensor":
    input_tensor = args[0]
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 expand requires a MatrixManTensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 expand supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 expand requires packed_rgba input storage")
    if len(args) < 2:
        raise RuntimeError("gm45 expand requires a target size")
    requested = tuple(int(value) for value in args[1])
    implicit = bool((kwargs or {}).get("implicit", args[2] if len(args) > 2 else False))
    source_shape = tuple(int(value) for value in input_tensor.shape)
    source_strides = tuple(int(value) for value in input_tensor._logical_strides)
    if len(requested) < len(source_shape):
        raise RuntimeError("gm45 expand cannot remove dimensions")
    rank_offset = len(requested) - len(source_shape)
    output_shape = []
    for output_index, requested_size in enumerate(requested):
        if requested_size < -1:
            raise RuntimeError("gm45 expand sizes must be non-negative or -1")
        if output_index < rank_offset:
            if requested_size == -1:
                raise RuntimeError("gm45 expand does not allow -1 on prepended dimensions")
            size = requested_size
        else:
            source_size = source_shape[output_index - rank_offset]
            size = source_size if requested_size == -1 else requested_size
            if size != source_size and source_size != 1:
                raise RuntimeError(
                    f"gm45 expand cannot expand dimension {output_index - rank_offset} "
                    f"from {source_size} to {size}"
                )
        if size <= 0:
            raise RuntimeError("gm45 expand does not support empty dimensions")
        output_shape.append(size)
    output_shape = tuple(output_shape)
    if output_shape:
        if len(output_shape) not in {1, 2, 3, 4}:
            raise RuntimeError("gm45 supports only 1D, 2D, 3D, and 4D expanded tensors")
        if len(output_shape) == 4 and output_shape[0] != 1:
            raise RuntimeError("gm45 4D expanded tensors require batch size 1")

    allocation_shape = output_shape or (1,)
    out_owner = operation_context.output_texture(allocation_shape)
    params = (
        source_shape, source_strides, output_shape, input_tensor._storage_offset,
        input_tensor._owner.layout.texture_width, input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program = _broadcast_program(params)
    diagnostics.trace(
        "gm45.kernel -> packed broadcast expand shader:\n"
        f"  input shape={list(source_shape)} offset={input_tensor._storage_offset} strides={list(source_strides)}\n"
        f"  requested={list(requested)} normalized={list(output_shape)} implicit={implicit}\n"
        f"  -> output texture #{out_owner.texture} shape={list(output_shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 expand framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after expand: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, output_shape)


def render_transpose(args) -> "MatrixManTensor":
    input_tensor = args[0]
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 t requires a MatrixManTensor input")
    rank = len(input_tensor.shape)
    if rank <= 1:
        return input_tensor
    if rank != 2:
        raise RuntimeError("gm45 t supports only scalar, 1D, and 2D tensors")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 t supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 t requires packed_rgba input storage")

    rows, cols = (int(value) for value in input_tensor.shape)
    output_shape = (cols, rows)
    out_owner = operation_context.output_texture(output_shape)
    params = (
        rows, cols, input_tensor._storage_offset,
        int(input_tensor._logical_strides[0]), int(input_tensor._logical_strides[1]),
        input_tensor._owner.layout.texture_width, input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program = _transpose_program(params)
    diagnostics.trace(
        "gm45.kernel -> packed transpose shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={[rows, cols]} "
        f"offset={input_tensor._storage_offset} strides={list(input_tensor._logical_strides)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(output_shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 t framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after t: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, output_shape)
