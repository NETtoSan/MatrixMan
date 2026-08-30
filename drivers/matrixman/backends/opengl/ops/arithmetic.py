"""OpenGL packed arithmetic operations."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, kernels, operation_context
from ..storage import numel
from ..tensor import Gm45Tensor

def _packed_add_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_add_programs:
        diagnostics.trace(f"gm45.compile -> packed add GLSL fragment shader params={params}")
        program = gm.make_program(_packed_add_shader_source(params))
        rt.packed_add_programs[params] = program
        rt.packed_add_uniforms[params] = (
            gm.glGetUniformLocation(program, b"left_tex"),
            gm.glGetUniformLocation(program, b"right_tex"),
        )
    left_loc, right_loc = rt.packed_add_uniforms[params]
    return rt.packed_add_programs[params], left_loc, right_loc

def _packed_add_shader_source(params: tuple) -> bytes:
    (
        numel,
        left_offset,
        right_offset,
        left_tex_w,
        left_tex_h,
        right_tex_w,
        right_tex_h,
        out_tex_w,
        alpha,
    ) = params
    source = """
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;

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
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}

float add_at(int linear_index)
{
    if (linear_index >= __NUMEL__) return 0.0;
    float left = read_packed(left_tex, linear_index + __LEFT_OFFSET__, __LEFT_TEX_W__, __LEFT_TEX_H__);
    float right = read_packed(right_tex, linear_index + __RIGHT_OFFSET__, __RIGHT_TEX_W__, __RIGHT_TEX_H__);
    return left + __ALPHA__ * right;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * __OUT_TEX_W__ + tex_x) * 4;
    gl_FragColor = vec4(
        add_at(base),
        add_at(base + 1),
        add_at(base + 2),
        add_at(base + 3)
    );
}
"""
    replacements = {
        "__NUMEL__": numel,
        "__LEFT_OFFSET__": left_offset,
        "__RIGHT_OFFSET__": right_offset,
        "__LEFT_TEX_W__": left_tex_w,
        "__LEFT_TEX_H__": left_tex_h,
        "__RIGHT_TEX_W__": right_tex_w,
        "__RIGHT_TEX_H__": right_tex_h,
        "__OUT_TEX_W__": out_tex_w,
        "__ALPHA__": f"{float(alpha):.10g}",
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _packed_sub_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_sub_programs:
        diagnostics.trace(f"gm45.compile -> packed sub GLSL fragment shader params={params}")
        program = gm.make_program(_packed_sub_shader_source(params))
        rt.packed_sub_programs[params] = program
        rt.packed_sub_uniforms[params] = (
            gm.glGetUniformLocation(program, b"left_tex"),
            gm.glGetUniformLocation(program, b"right_tex"),
        )
    left_loc, right_loc = rt.packed_sub_uniforms[params]
    return rt.packed_sub_programs[params], left_loc, right_loc

def _packed_sub_shader_source(params: tuple) -> bytes:
    shape, left_offset, right_offset, left_strides, right_strides, left_tex_w, left_tex_h, right_tex_w, right_tex_h, out_tex_w = params
    _, channels, anchors = shape
    source = """
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;

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

float subtract_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int anchor = out_index - (out_index / ANCHORS) * ANCHORS;
    int tmp0 = out_index / ANCHORS;
    int channel = tmp0 - (tmp0 / CHANNELS) * CHANNELS;
    int batch = tmp0 / CHANNELS;
    int left_index = LEFT_OFFSET + batch * LEFT_STRIDE0
        + channel * LEFT_STRIDE1 + anchor * LEFT_STRIDE2;
    int right_index = RIGHT_OFFSET + batch * RIGHT_STRIDE0
        + channel * RIGHT_STRIDE1 + anchor * RIGHT_STRIDE2;
    return read_packed(left_tex, left_index, LEFT_TEX_W, LEFT_TEX_H)
        - read_packed(right_tex, right_index, RIGHT_TEX_W, RIGHT_TEX_H);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        subtract_at(base),
        subtract_at(base + 1),
        subtract_at(base + 2),
        subtract_at(base + 3)
    );
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape),
        "CHANNELS": channels,
        "ANCHORS": anchors,
        "LEFT_OFFSET": left_offset,
        "RIGHT_OFFSET": right_offset,
        "LEFT_STRIDE0": left_strides[0],
        "LEFT_STRIDE1": left_strides[1],
        "LEFT_STRIDE2": left_strides[2],
        "RIGHT_STRIDE0": right_strides[0],
        "RIGHT_STRIDE1": right_strides[1],
        "RIGHT_STRIDE2": right_strides[2],
        "LEFT_TEX_W": left_tex_w,
        "LEFT_TEX_H": left_tex_h,
        "RIGHT_TEX_W": right_tex_w,
        "RIGHT_TEX_H": right_tex_h,
        "OUT_TEX_W": out_tex_w,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_packed_sub(left: "Gm45Tensor", right: "Gm45Tensor") -> "Gm45Tensor":
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("gm45 packed sub only supports float32")
    if left.shape != right.shape:
        raise RuntimeError("gm45 packed sub requires equal shapes")
    if left._owner.layout.kind != "packed_rgba" or right._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 packed sub requires packed_rgba input storage")
    if len(left.shape) != 3 or tuple(int(v) for v in left.shape[:1]) != (1,):
        raise RuntimeError("gm45 packed sub currently supports only batch-1 rank-3 tensors")
    shape = tuple(int(v) for v in left.shape)
    out_owner = operation_context.output_texture(shape)
    params = (
        shape,
        left._storage_offset,
        right._storage_offset,
        left._logical_strides,
        right._logical_strides,
        left._owner.layout.texture_width,
        left._owner.layout.texture_height,
        right._owner.layout.texture_width,
        right._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, left_loc, right_loc = _packed_sub_program(params)
    rt = operation_context.gl_runtime()
    diagnostics.trace(
        "gm45.kernel -> packed sub shader:\n"
        f"  left texture #{left._owner.texture} shape={list(left.shape)} offset={left._storage_offset} strides={list(left._logical_strides)}\n"
        f"  right texture #{right._owner.texture} shape={list(right.shape)} offset={right._storage_offset} strides={list(right._logical_strides)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 packed sub framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, left._owner.texture)
    gm.glUniform1i(left_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1)
    gm.glBindTexture(gm.GL_TEXTURE_2D, right._owner.texture)
    gm.glUniform1i(right_loc, 1)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after packed sub: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, shape)
def _render_packed_add(left: "Gm45Tensor", right: "Gm45Tensor", alpha: float) -> "Gm45Tensor":
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("gm45 packed add only supports float32")
    operation_context.require_contiguous(left, "packed add")
    operation_context.require_contiguous(right, "packed add")
    shape = tuple(int(v) for v in left.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 packed add does not support empty tensors")
    out_owner = operation_context.output_texture(shape)
    params = (
        numel(shape),
        left._storage_offset,
        right._storage_offset,
        left._owner.layout.texture_width,
        left._owner.layout.texture_height,
        right._owner.layout.texture_width,
        right._owner.layout.texture_height,
        out_owner.layout.texture_width,
        float(alpha),
    )
    program, left_loc, right_loc = _packed_add_program(params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> packed add shader:\n"
        f"  left texture #{left._owner.texture} shape={list(shape)} offset={left._storage_offset}\n"
        f"  right texture #{right._owner.texture} shape={list(right.shape)} offset={right._storage_offset}\n"
        f"  alpha={float(alpha):.10g}\n"
        f"  -> output texture #{out_owner.texture} shape={list(shape)}"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 packed add framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, left._owner.texture)
    gm.glUniform1i(left_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1)
    gm.glBindTexture(gm.GL_TEXTURE_2D, right._owner.texture)
    gm.glUniform1i(right_loc, 1)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted packed add fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after packed add: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, shape)


def _packed_strided_add_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_strided_add_programs:
        diagnostics.trace(f"gm45.compile -> stride-aware packed add GLSL fragment shader params={params}")
        program = gm.make_program(_packed_strided_add_shader_source(params))
        rt.packed_strided_add_programs[params] = program
        rt.packed_strided_add_uniforms[params] = (
            gm.glGetUniformLocation(program, b"left_tex"),
            gm.glGetUniformLocation(program, b"right_tex"),
        )
    left_loc, right_loc = rt.packed_strided_add_uniforms[params]
    return rt.packed_strided_add_programs[params], left_loc, right_loc

def _packed_strided_add_shader_source(params: tuple) -> bytes:
    shape, left_offset, right_offset, left_strides, right_strides, left_tex_w, left_tex_h, right_tex_w, right_tex_h, out_tex_w, alpha = params
    _, channels, anchors = shape
    source = """
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;

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

float add_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int anchor = out_index - (out_index / ANCHORS) * ANCHORS;
    int tmp0 = out_index / ANCHORS;
    int channel = tmp0 - (tmp0 / CHANNELS) * CHANNELS;
    int batch = tmp0 / CHANNELS;
    int left_index = LEFT_OFFSET + batch * LEFT_STRIDE0
        + channel * LEFT_STRIDE1 + anchor * LEFT_STRIDE2;
    int right_index = RIGHT_OFFSET + batch * RIGHT_STRIDE0
        + channel * RIGHT_STRIDE1 + anchor * RIGHT_STRIDE2;
    return read_packed(left_tex, left_index, LEFT_TEX_W, LEFT_TEX_H)
        + ALPHA * read_packed(right_tex, right_index, RIGHT_TEX_W, RIGHT_TEX_H);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(add_at(base), add_at(base + 1), add_at(base + 2), add_at(base + 3));
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape), "CHANNELS": channels, "ANCHORS": anchors,
        "LEFT_OFFSET": left_offset, "RIGHT_OFFSET": right_offset,
        "LEFT_STRIDE0": left_strides[0], "LEFT_STRIDE1": left_strides[1], "LEFT_STRIDE2": left_strides[2],
        "RIGHT_STRIDE0": right_strides[0], "RIGHT_STRIDE1": right_strides[1], "RIGHT_STRIDE2": right_strides[2],
        "LEFT_TEX_W": left_tex_w, "LEFT_TEX_H": left_tex_h,
        "RIGHT_TEX_W": right_tex_w, "RIGHT_TEX_H": right_tex_h,
        "OUT_TEX_W": out_tex_w, "ALPHA": kernels.glsl_float(alpha),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_packed_strided_add(left: "Gm45Tensor", right: "Gm45Tensor", alpha: float) -> "Gm45Tensor":
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("gm45 stride-aware packed add only supports float32")
    if left.shape != right.shape or len(left.shape) != 3 or int(left.shape[0]) != 1:
        raise RuntimeError("gm45 stride-aware packed add supports only equal-shape batch-1 rank-3 tensors")
    if left._owner.layout.kind != "packed_rgba" or right._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 stride-aware packed add requires packed_rgba input storage")
    shape = tuple(int(v) for v in left.shape)
    out_owner = operation_context.output_texture(shape)
    params = (shape, left._storage_offset, right._storage_offset, left._logical_strides, right._logical_strides,
              left._owner.layout.texture_width, left._owner.layout.texture_height,
              right._owner.layout.texture_width, right._owner.layout.texture_height,
              out_owner.layout.texture_width, float(alpha))
    program, left_loc, right_loc = _packed_strided_add_program(params)
    rt = operation_context.gl_runtime()
    diagnostics.trace(
        "gm45.kernel -> stride-aware packed add shader:\n"
        f"  left texture #{left._owner.texture} shape={list(shape)} offset={left._storage_offset} strides={list(left._logical_strides)}\n"
        f"  right texture #{right._owner.texture} shape={list(shape)} offset={right._storage_offset} strides={list(right._logical_strides)}\n"
        f"  alpha={float(alpha):.10g} -> output texture #{out_owner.texture} offset=0"
    )
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 stride-aware packed add framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, left._owner.texture)
    gm.glUniform1i(left_loc, 0)
    gm.glActiveTexture(gm.GL_TEXTURE1)
    gm.glBindTexture(gm.GL_TEXTURE_2D, right._owner.texture)
    gm.glUniform1i(right_loc, 1)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after stride-aware packed add: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, shape)

def _packed_scalar_div_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_scalar_div_programs:
        diagnostics.trace(f"gm45.compile -> packed scalar div GLSL fragment shader params={params}")
        program = gm.make_program(_packed_scalar_div_shader_source(params))
        rt.packed_scalar_div_programs[params] = program
        rt.packed_scalar_div_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.packed_scalar_div_programs[params], rt.packed_scalar_div_uniforms[params]

def _packed_scalar_div_shader_source(params: tuple) -> bytes:
    numel, input_offset, input_tex_w, input_tex_h, out_tex_w, divisor = params
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

float divide_at(int linear_index)
{
    if (linear_index >= NUMEL) return 0.0;
    return read_packed(linear_index + INPUT_OFFSET) / DIVISOR;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(divide_at(base), divide_at(base + 1), divide_at(base + 2), divide_at(base + 3));
}
"""
    replacements = {
        "NUMEL": int(numel), "INPUT_OFFSET": int(input_offset),
        "INPUT_TEX_W": int(input_tex_w), "INPUT_TEX_H": int(input_tex_h),
        "OUT_TEX_W": int(out_tex_w), "DIVISOR": kernels.glsl_float(divisor),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_packed_scalar_div(tensor: "Gm45Tensor", divisor: float) -> "Gm45Tensor":
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 scalar div supports only float32")
    if tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 scalar div requires packed_rgba input storage")
    shape = tuple(int(v) for v in tensor.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 2 or shape[2] <= 0:
        raise RuntimeError("gm45 scalar div currently supports only shape [1,2,A] with A > 0")
    if float(divisor) != 2.0:
        raise RuntimeError("gm45 scalar div currently supports only divisor 2.0")
    out_owner = operation_context.output_texture(shape)
    params = (
        numel(shape), tensor._storage_offset,
        tensor._owner.layout.texture_width, tensor._owner.layout.texture_height,
        out_owner.layout.texture_width, float(divisor),
    )
    program, input_loc = _packed_scalar_div_program(params)
    rt = operation_context.gl_runtime()
    diagnostics.trace(
        "gm45.kernel -> packed scalar div shader:\n"
        f"  input texture #{tensor._owner.texture} shape={list(tensor.shape)} offset={tensor._storage_offset} strides={list(tensor._logical_strides)}\n"
        f"  divisor={float(divisor):.10g} -> output texture #{out_owner.texture} offset=0"
    )
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 scalar div framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after scalar div: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, shape)

def _packed_broadcast_mul_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_broadcast_mul_programs:
        diagnostics.trace(f"gm45.compile -> packed broadcast mul GLSL fragment shader params={params}")
        program = gm.make_program(_packed_broadcast_mul_shader_source(params))
        rt.packed_broadcast_mul_programs[params] = program
        rt.packed_broadcast_mul_uniforms[params] = (
            gm.glGetUniformLocation(program, b"left_tex"),
            gm.glGetUniformLocation(program, b"right_tex"),
        )
    left_loc, right_loc = rt.packed_broadcast_mul_uniforms[params]
    return rt.packed_broadcast_mul_programs[params], left_loc, right_loc

def _packed_broadcast_mul_shader_source(params: tuple) -> bytes:
    shape, left_offset, right_offset, left_strides, right_strides, left_tex_w, left_tex_h, right_tex_w, right_tex_h, out_tex_w = params
    _, channels, anchors = shape
    source = """
#version 120
uniform sampler2D left_tex;
uniform sampler2D right_tex;

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

float multiply_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int anchor = out_index - (out_index / ANCHORS) * ANCHORS;
    int tmp0 = out_index / ANCHORS;
    int channel = tmp0 - (tmp0 / CHANNELS) * CHANNELS;
    int batch = tmp0 / CHANNELS;
    int left_index = LEFT_OFFSET + batch * LEFT_STRIDE0
        + channel * LEFT_STRIDE1 + anchor * LEFT_STRIDE2;
    int right_index = RIGHT_OFFSET + batch * RIGHT_STRIDE0 + anchor * RIGHT_STRIDE1;
    return read_packed(left_tex, left_index, LEFT_TEX_W, LEFT_TEX_H)
        * read_packed(right_tex, right_index, RIGHT_TEX_W, RIGHT_TEX_H);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(multiply_at(base), multiply_at(base + 1), multiply_at(base + 2), multiply_at(base + 3));
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape), "CHANNELS": channels, "ANCHORS": anchors,
        "LEFT_OFFSET": left_offset, "RIGHT_OFFSET": right_offset,
        "LEFT_STRIDE0": left_strides[0], "LEFT_STRIDE1": left_strides[1], "LEFT_STRIDE2": left_strides[2],
        "RIGHT_STRIDE0": right_strides[0], "RIGHT_STRIDE1": right_strides[1],
        "LEFT_TEX_W": left_tex_w, "LEFT_TEX_H": left_tex_h,
        "RIGHT_TEX_W": right_tex_w, "RIGHT_TEX_H": right_tex_h,
        "OUT_TEX_W": out_tex_w,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_packed_broadcast_mul(left: "Gm45Tensor", right: "Gm45Tensor") -> "Gm45Tensor":
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("gm45 broadcast mul supports only float32")
    left_shape = tuple(int(v) for v in left.shape)
    right_shape = tuple(int(v) for v in right.shape)
    if len(left_shape) != 3 or left_shape[0] != 1 or left_shape[1] != 4 or left_shape[2] <= 0:
        raise RuntimeError("gm45 broadcast mul supports only lhs shape [1,4,A] with A > 0")
    if right_shape != (1, left_shape[2]):
        raise RuntimeError("gm45 broadcast mul supports only rhs shape [1,A] matching lhs")
    if left._owner.layout.kind != "packed_rgba" or right._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 broadcast mul requires packed_rgba input storage")
    out_shape = left_shape
    out_owner = operation_context.output_texture(out_shape)
    params = (
        out_shape, left._storage_offset, right._storage_offset,
        left._logical_strides, right._logical_strides,
        left._owner.layout.texture_width, left._owner.layout.texture_height,
        right._owner.layout.texture_width, right._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, left_loc, right_loc = _packed_broadcast_mul_program(params)
    rt = operation_context.gl_runtime()
    diagnostics.trace(
        "gm45.kernel -> packed broadcast mul shader:\n"
        f"  left texture #{left._owner.texture} shape={list(left.shape)} offset={left._storage_offset} strides={list(left._logical_strides)}\n"
        f"  right texture #{right._owner.texture} shape={list(right.shape)} offset={right._storage_offset} strides={list(right._logical_strides)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 broadcast mul framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in [(gm.GL_TEXTURE0, left, left_loc), (gm.GL_TEXTURE1, right, right_loc)]:
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after broadcast mul: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, out_shape)

def _scalar_add_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.scalar_add_programs:
        diagnostics.trace(f"gm45.compile -> scalar add GLSL fragment shader params={params}")
        program = gm.make_program(_scalar_add_shader_source(params))
        rt.scalar_add_programs[params] = program
        rt.scalar_add_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.scalar_add_programs[params], rt.scalar_add_uniforms[params]

def _scalar_add_shader_source(params: tuple) -> bytes:
    numel, input_offset, input_tex_w, input_tex_h, out_tex_w, scalar, alpha, tensor_first = params
    if tensor_first:
        expr = "x + __ALPHA__ * __SCALAR__"
    else:
        expr = "__SCALAR__ + __ALPHA__ * x"
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
    int x = texel - (texel / __INPUT_TEX_W__) * __INPUT_TEX_W__;
    int y = texel / __INPUT_TEX_W__;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(__INPUT_TEX_W__), float(__INPUT_TEX_H__));
    return pick_component(texture2D(input_tex, uv), component);
}}

float add_at(int linear_index)
{{
    if (linear_index >= __NUMEL__) return 0.0;
    float x = read_packed(linear_index + __INPUT_OFFSET__);
    return {expr};
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * __OUT_TEX_W__ + tex_x) * 4;
    gl_FragColor = vec4(
        add_at(base),
        add_at(base + 1),
        add_at(base + 2),
        add_at(base + 3)
    );
}}
"""
    replacements = {
        "__NUMEL__": int(numel),
        "__INPUT_OFFSET__": int(input_offset),
        "__INPUT_TEX_W__": int(input_tex_w),
        "__INPUT_TEX_H__": int(input_tex_h),
        "__OUT_TEX_W__": int(out_tex_w),
        "__SCALAR__": kernels.glsl_float(scalar),
        "__ALPHA__": kernels.glsl_float(alpha),
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_scalar_add(tensor: "Gm45Tensor", scalar: float, alpha: float, *, tensor_first: bool) -> "Gm45Tensor":
    if not isinstance(tensor, Gm45Tensor):
        raise RuntimeError("gm45 scalar add requires one Gm45Tensor input")
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 scalar add supports only float32")
    if tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 scalar add currently supports only packed_rgba tensor storage")
    operation_context.require_contiguous(tensor, "scalar add")
    shape = tuple(int(v) for v in tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 scalar add does not support empty tensors")

    out_owner = operation_context.output_texture(shape)
    params = (
        numel(shape),
        tensor._storage_offset,
        tensor._owner.layout.texture_width,
        tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
        float(scalar),
        float(alpha),
        bool(tensor_first),
    )
    program, input_loc = _scalar_add_program(params)
    rt = operation_context.gl_runtime()

    if tensor_first:
        formula = f"out = input + {float(alpha):.10g} * {float(scalar):.10g}"
    else:
        formula = f"out = {float(scalar):.10g} + {float(alpha):.10g} * input"
    diagnostics.trace(
        "gm45.kernel -> scalar add shader:\n"
        f"  input texture #{tensor._owner.texture} shape={list(shape)} offset={tensor._storage_offset}\n"
        f"  {formula}\n"
        f"  -> output texture #{out_owner.texture} shape={list(shape)} offset=0"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 scalar add framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted scalar add fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after scalar add: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, shape)
