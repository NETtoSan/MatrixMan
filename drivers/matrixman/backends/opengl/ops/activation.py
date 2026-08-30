"""OpenGL activation operations."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, kernels, operation_context, profiling
from ..storage import numel
from ..tensor import Gm45Tensor

def _silu_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.silu_programs:
        diagnostics.trace(f"gm45.compile -> SiLU GLSL fragment shader params={params}")
        program = gm.make_program(_silu_shader_source(params))
        rt.silu_programs[params] = program
        rt.silu_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.silu_programs[params], rt.silu_uniforms[params]

def _packed_sigmoid_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_sigmoid_programs:
        diagnostics.trace(f"gm45.compile -> packed sigmoid GLSL fragment shader params={params}")
        program = gm.make_program(_packed_sigmoid_shader_source(params))
        rt.packed_sigmoid_programs[params] = program
        rt.packed_sigmoid_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.packed_sigmoid_programs[params], rt.packed_sigmoid_uniforms[params]

def _packed_sigmoid_shader_source(params: tuple) -> bytes:
    shape, input_offset, input_strides, input_tex_w, input_tex_h, out_tex_w = params
    _, channels, anchors = shape
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

float sigmoid_at(int linear_index)
{
    if (linear_index >= NUMEL) return 0.0;
    int anchor = linear_index - (linear_index / ANCHORS) * ANCHORS;
    int tmp = linear_index / ANCHORS;
    int channel = tmp - (tmp / CHANNELS) * CHANNELS;
    int batch = tmp / CHANNELS;
    int source_index = INPUT_OFFSET + batch * INPUT_STRIDE0
        + channel * INPUT_STRIDE1 + anchor * INPUT_STRIDE2;
    float x = read_packed(source_index);
    return 1.0 / (1.0 + exp(-x));
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(sigmoid_at(base), sigmoid_at(base + 1), sigmoid_at(base + 2), sigmoid_at(base + 3));
}
"""
    replacements = {
        "NUMEL": numel(shape), "CHANNELS": int(channels), "ANCHORS": int(anchors),
        "INPUT_OFFSET": int(input_offset),
        "INPUT_STRIDE0": int(input_strides[0]), "INPUT_STRIDE1": int(input_strides[1]),
        "INPUT_STRIDE2": int(input_strides[2]),
        "INPUT_TEX_W": int(input_tex_w), "INPUT_TEX_H": int(input_tex_h),
        "OUT_TEX_W": int(out_tex_w),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_packed_sigmoid(tensor: "Gm45Tensor") -> "Gm45Tensor":
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 sigmoid supports only float32")
    if tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 sigmoid requires packed_rgba input storage")
    shape = tuple(int(v) for v in tensor.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] <= 0:
        raise RuntimeError("gm45 sigmoid supports only shape [1,C,A] with C > 0 and A > 0")
    out_shape = shape
    out_owner = operation_context.output_texture(out_shape)
    params = (
        out_shape, tensor._storage_offset, tensor._logical_strides,
        tensor._owner.layout.texture_width, tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _packed_sigmoid_program(params)
    rt = operation_context.gl_runtime()
    diagnostics.trace(
        "gm45.kernel -> packed sigmoid shader:\n"
        f"  input texture #{tensor._owner.texture} shape={list(tensor.shape)} offset={tensor._storage_offset} strides={list(tensor._logical_strides)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 sigmoid framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    with profiling.gpu_timer("Sigmoid"):
        operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after sigmoid: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, out_shape)

def _silu_shader_source(params: tuple) -> bytes:
    numel, input_offset, input_tex_w, input_tex_h, out_tex_w = params
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
    int x = texel - (texel / __INPUT_TEX_W__) * __INPUT_TEX_W__;
    int y = texel / __INPUT_TEX_W__;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(__INPUT_TEX_W__), float(__INPUT_TEX_H__));
    return pick_component(texture2D(input_tex, uv), component);
}

float silu_at(int linear_index)
{
    if (linear_index >= __NUMEL__) return 0.0;
    float x = read_packed(linear_index + __INPUT_OFFSET__);
    return x / (1.0 + exp(-x));
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * __OUT_TEX_W__ + tex_x) * 4;
    gl_FragColor = vec4(
        silu_at(base),
        silu_at(base + 1),
        silu_at(base + 2),
        silu_at(base + 3)
    );
}
"""
    replacements = {
        "__NUMEL__": numel,
        "__INPUT_OFFSET__": input_offset,
        "__INPUT_TEX_W__": input_tex_w,
        "__INPUT_TEX_H__": input_tex_h,
        "__OUT_TEX_W__": out_tex_w,
    }
    for name, value in replacements.items():
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_silu_inplace(args) -> "Gm45Tensor":
    input_tensor = args[0]
    if not isinstance(input_tensor, Gm45Tensor):
        raise RuntimeError("gm45 silu_ requires a Gm45Tensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 silu_ supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 silu_ currently supports packed_rgba tensor storage")
    operation_context.require_contiguous(input_tensor, "silu_")

    shape = tuple(int(v) for v in input_tensor.shape)
    out_owner = operation_context.output_texture(shape)
    params = (
        numel(shape),
        input_tensor._storage_offset,
        input_tensor._owner.layout.texture_width,
        input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _silu_program(params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> SiLU shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(shape)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(shape)}\n"
        "  note: aten.silu_ is implemented with a new texture to avoid OpenGL FBO feedback hazards"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 SiLU framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)

    with profiling.gpu_timer("SiLU"):
        operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted SiLU fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after silu_: 0x{err:04x}")
    input_tensor._owner = out_owner
    input_tensor._shape = shape
    input_tensor._storage_offset = 0
    return input_tensor
