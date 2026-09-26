"""OpenGL fixed-bin channel softmax operation."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context
from ....tensor import MatrixManTensor
from ..storage import numel


def _logsoftmax_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.logsoftmax_programs:
        diagnostics.trace(f"gm45.compile -> generic log-softmax GLSL fragment shader params={params}")
        program = gm.make_program(_logsoftmax_shader_source(params))
        rt.logsoftmax_programs[params] = program
        rt.logsoftmax_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.logsoftmax_programs[params], rt.logsoftmax_uniforms[params]


def _logsoftmax_shader_source(params: tuple) -> bytes:
    shape, dim, input_offset, input_strides, input_tex_w, input_tex_h, out_tex_w = params
    shape = tuple(int(value) for value in shape)
    dim = int(dim)
    reduction_size = shape[dim]
    non_reduced_terms = []
    current_terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((out_index / {max(1, suffix)}) - "
            f"(out_index / {max(1, suffix * size)}) * {size})"
        )
        if axis == dim:
            non_reduced_terms.append("reduction_index")
            current_terms.append(coordinate)
        else:
            non_reduced_terms.append(coordinate)
            current_terms.append(coordinate)
    reduction_index = " + ".join(
        f"({term}) * {int(input_strides[axis])}" for axis, term in enumerate(non_reduced_terms)
    ) or "0"
    current_index = " + ".join(
        f"({term}) * {int(input_strides[axis])}" for axis, term in enumerate(current_terms)
    ) or "0"
    source = """
#version 120
uniform sampler2D input_tex;

float pick_component(vec4 value, int component)
{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}

float read_packed(int linear_index)
{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / INPUT_TEX_W) * INPUT_TEX_W;
    int y = texel / INPUT_TEX_W;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(INPUT_TEX_W), float(INPUT_TEX_H));
    return pick_component(texture2D(input_tex, uv), component);
}

float source_for_reduction(int out_index, int reduction_index)
{
    int input_index = INPUT_OFFSET + __REDUCTION_INDEX__;
    return read_packed(input_index);
}

float source_for_output(int out_index)
{
    int input_index = INPUT_OFFSET + __CURRENT_INDEX__;
    return read_packed(input_index);
}

float logsoftmax_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    float maximum = -3.402823e+38;
    for (int reduction_index = 0; reduction_index < REDUCTION_SIZE; ++reduction_index)
        maximum = max(maximum, source_for_reduction(out_index, reduction_index));
    float total = 0.0;
    for (int reduction_index = 0; reduction_index < REDUCTION_SIZE; ++reduction_index)
        total += exp(source_for_reduction(out_index, reduction_index) - maximum);
    return source_for_output(out_index) - maximum - log(total);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        logsoftmax_at(base),
        logsoftmax_at(base + 1),
        logsoftmax_at(base + 2),
        logsoftmax_at(base + 3)
    );
}
"""
    replacements = {
        "INPUT_OFFSET": input_offset,
        "INPUT_TEX_W": input_tex_w,
        "INPUT_TEX_H": input_tex_h,
        "OUT_TEX_W": out_tex_w,
        "OUT_NUMEL": numel(shape),
        "REDUCTION_SIZE": reduction_size,
        "__REDUCTION_INDEX__": reduction_index,
        "__CURRENT_INDEX__": current_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_logsoftmax(args) -> "MatrixManTensor":
    if len(args) < 2:
        raise RuntimeError("gm45 log_softmax requires input and dim")
    input_tensor = args[0]
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 log_softmax requires a MatrixManTensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 log_softmax supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 log_softmax requires packed_rgba input storage")
    if bool(args[2]) if len(args) > 2 else False:
        raise RuntimeError("gm45 log_softmax supports half_to_float=False only")
    shape = tuple(int(value) for value in input_tensor.shape)
    if not shape or numel(shape) <= 0:
        raise RuntimeError("gm45 log_softmax does not support empty or scalar tensors")
    dim = int(args[1])
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        raise RuntimeError(f"gm45 log_softmax dimension out of range: {args[1]}")
    operation_context.require_contiguous(input_tensor, "log_softmax")
    out_owner = operation_context.output_texture(shape)
    params = (
        shape, dim, input_tensor._storage_offset, input_tensor._logical_strides,
        input_tensor._owner.layout.texture_width, input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _logsoftmax_program(params)
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 log_softmax framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after log_softmax: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, shape)


def _logsoftmax_backward_program(params):
    rt = operation_context.gl_runtime()
    if params not in rt.logsoftmax_backward_programs:
        diagnostics.trace(f"gm45.compile -> log-softmax backward GLSL fragment shader params={params}")
        program = gm.make_program(_logsoftmax_backward_shader_source(params))
        rt.logsoftmax_backward_programs[params] = program
        rt.logsoftmax_backward_uniforms[params] = (
            gm.glGetUniformLocation(program, b"grad_tex"),
            gm.glGetUniformLocation(program, b"output_tex"),
        )
    grad_loc, output_loc = rt.logsoftmax_backward_uniforms[params]
    return rt.logsoftmax_backward_programs[params], grad_loc, output_loc


def _logsoftmax_backward_shader_source(params):
    shape, dim, grad_offset, grad_strides, grad_w, grad_h, output_offset, output_strides, output_w, output_h, out_w = params
    shape = tuple(int(value) for value in shape)
    dim = int(dim)
    reduction_size = shape[dim]
    sum_terms = []
    output_terms = []
    grad_value_terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((out_index / {max(1, suffix)}) - "
            f"(out_index / {max(1, suffix * size)}) * {size})"
        )
        sum_coordinate = "reduction_index" if axis == dim else coordinate
        sum_terms.append(f"({sum_coordinate}) * {int(grad_strides[axis])}")
        output_terms.append(f"({coordinate}) * {int(output_strides[axis])}")
        grad_value_terms.append(f"({coordinate}) * {int(grad_strides[axis])}")
    sum_index = " + ".join(sum_terms) or "0"
    output_index = " + ".join(output_terms) or "0"
    grad_value_index = " + ".join(grad_value_terms) or "0"
    source = """
#version 120
uniform sampler2D grad_tex;
uniform sampler2D output_tex;

float pick_component(vec4 value, int component)
{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}

float read_packed(sampler2D tex, int linear_index, int tex_width, int tex_height)
{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}

float summed_grad(int out_index)
{
    float total = 0.0;
    for (int reduction_index = 0; reduction_index < REDUCTION_SIZE; ++reduction_index)
        total += read_packed(grad_tex, GRAD_OFFSET + __SUM_INDEX__, GRAD_TEX_W, GRAD_TEX_H);
    return total;
}

float backward_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    float grad = read_packed(grad_tex, GRAD_OFFSET + __GRAD_VALUE_INDEX__, GRAD_TEX_W, GRAD_TEX_H);
    float logp = read_packed(output_tex, OUTPUT_OFFSET + __OUTPUT_INDEX__, OUTPUT_TEX_W, OUTPUT_TEX_H);
    return grad - exp(logp) * summed_grad(out_index);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        backward_at(base), backward_at(base + 1),
        backward_at(base + 2), backward_at(base + 3)
    );
}
"""
    replacements = {
        "GRAD_OFFSET": grad_offset, "GRAD_TEX_W": grad_w, "GRAD_TEX_H": grad_h,
        "OUTPUT_OFFSET": output_offset, "OUTPUT_TEX_W": output_w, "OUTPUT_TEX_H": output_h,
        "OUT_TEX_W": out_w, "OUT_NUMEL": numel(shape), "REDUCTION_SIZE": reduction_size,
        "__SUM_INDEX__": sum_index, "__GRAD_VALUE_INDEX__": grad_value_index,
        "__OUTPUT_INDEX__": output_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_logsoftmax_backward(args):
    if len(args) < 4:
        raise RuntimeError("gm45 log_softmax backward requires grad_output, output, dim, and input_dtype")
    grad_output, output, dim, input_dtype = args[:4]
    if not isinstance(grad_output, MatrixManTensor) or not isinstance(output, MatrixManTensor):
        raise RuntimeError("gm45 log_softmax backward requires MatrixManTensor operands")
    if grad_output.dtype != torch.float32 or output.dtype != torch.float32:
        raise RuntimeError("gm45 log_softmax backward supports only float32")
    if grad_output.shape != output.shape or output._owner.layout.kind != "packed_rgba" or grad_output._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 log_softmax backward requires equal packed_rgba tensors")
    if input_dtype != torch.float32:
        raise RuntimeError("gm45 log_softmax backward supports input_dtype=torch.float32 only")
    shape = tuple(int(value) for value in output.shape)
    if not shape or numel(shape) <= 0:
        raise RuntimeError("gm45 log_softmax backward does not support empty or scalar tensors")
    dim = int(dim)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        raise RuntimeError(f"gm45 log_softmax backward dimension out of range: {args[2]}")
    operation_context.require_contiguous(grad_output, "log_softmax backward grad_output")
    operation_context.require_contiguous(output, "log_softmax backward output")
    owner = operation_context.output_texture(shape)
    params = (
        shape, dim, grad_output._storage_offset, grad_output._logical_strides,
        grad_output._owner.layout.texture_width, grad_output._owner.layout.texture_height,
        output._storage_offset, output._logical_strides,
        output._owner.layout.texture_width, output._owner.layout.texture_height,
        owner.layout.texture_width,
    )
    program, grad_loc, output_loc = _logsoftmax_backward_program(params)
    operation_context.attach_output(owner)
    operation_context.framebuffer_complete("gm45 log_softmax backward framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in ((gm.GL_TEXTURE0, grad_output, grad_loc), (gm.GL_TEXTURE1, output, output_loc)):
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    return MatrixManTensor._from_owner(owner, shape)

def _softmax_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.softmax_programs:
        diagnostics.trace(f"gm45.compile -> softmax GLSL fragment shader params={params}")
        program = gm.make_program(_softmax_shader_source(params))
        rt.softmax_programs[params] = program
        rt.softmax_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.softmax_programs[params], rt.softmax_uniforms[params]

def _softmax_shader_source(params: tuple) -> bytes:
    channels, coord_count, anchor_count, input_offset, input_strides, input_tex_w, input_tex_h, out_tex_w = params
    source = """
#version 120
uniform sampler2D input_tex;

float pick_component(vec4 value, int component)
{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}

float read_packed(int linear_index)
{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / INPUT_TEX_W) * INPUT_TEX_W;
    int y = texel / INPUT_TEX_W;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(INPUT_TEX_W), float(INPUT_TEX_H));
    return pick_component(texture2D(input_tex, uv), component);
}

float source_at(int bin, int coord, int anchor)
{
    int source_index = INPUT_OFFSET
        + bin * INPUT_STRIDE_C
        + coord * INPUT_STRIDE_COORD
        + anchor * INPUT_STRIDE_ANCHOR;
    return read_packed(source_index);
}

float softmax_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int anchor = out_index - (out_index / ANCHORS) * ANCHORS;
    int tmp0 = out_index / ANCHORS;
    int coord = tmp0 - (tmp0 / COORDS) * COORDS;
    int channel = tmp0 / COORDS;

    float maximum = -3.402823e+38;
    for (int bin = 0; bin < 16; ++bin) {
        maximum = max(maximum, source_at(bin, coord, anchor));
    }

    float total = 0.0;
    for (int bin = 0; bin < 16; ++bin) {
        total += exp(source_at(bin, coord, anchor) - maximum);
    }
    float numerator = exp(source_at(channel, coord, anchor) - maximum);
    return numerator / total;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        softmax_at(base),
        softmax_at(base + 1),
        softmax_at(base + 2),
        softmax_at(base + 3)
    );
}
"""
    replacements = {
        "INPUT_OFFSET": int(input_offset),
        "INPUT_STRIDE_C": int(input_strides[1]),
        "INPUT_STRIDE_COORD": int(input_strides[2]),
        "INPUT_STRIDE_ANCHOR": int(input_strides[3]),
        "INPUT_TEX_W": int(input_tex_w),
        "INPUT_TEX_H": int(input_tex_h),
        "OUT_TEX_W": int(out_tex_w),
        "OUT_NUMEL": int(channels * coord_count * anchor_count),
        "COORDS": int(coord_count),
        "ANCHORS": int(anchor_count),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_softmax(args) -> "MatrixManTensor":
    input_tensor = args[0]
    dim = int(args[1])
    half_to_float = bool(args[2]) if len(args) > 2 else False
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 softmax requires a MatrixManTensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 softmax supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 softmax currently supports only packed_rgba tensor storage")
    if len(input_tensor.shape) != 4:
        raise RuntimeError("gm45 softmax currently supports only rank-4 input")
    normalized_dim = dim + len(input_tensor.shape) if dim < 0 else dim
    if normalized_dim != 1:
        raise RuntimeError("gm45 softmax currently supports only dim=1")
    if int(input_tensor.shape[0]) != 1 or int(input_tensor.shape[1]) != 16:
        raise RuntimeError("gm45 softmax currently supports only shape [1,16,coord,anchor]")
    if half_to_float:
        raise RuntimeError("gm45 softmax currently supports half_to_float=False only")

    _, channels, coord_count, anchor_count = (int(v) for v in input_tensor.shape)
    out_shape = tuple(int(v) for v in input_tensor.shape)
    out_owner = operation_context.output_texture(out_shape)
    params = (
        channels,
        coord_count,
        anchor_count,
        input_tensor._storage_offset,
        input_tensor._logical_strides,
        input_tensor._owner.layout.texture_width,
        input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _softmax_program(params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> softmax shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(input_tensor.shape)} "
        f"offset={input_tensor._storage_offset} strides={list(input_tensor._logical_strides)}\n"
        f"  dim={dim} normalized_dim={normalized_dim} bins={channels} half_to_float={half_to_float}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 softmax framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted softmax fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after softmax: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, out_shape)
