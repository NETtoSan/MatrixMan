"""OpenGL packed arithmetic operations."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, kernels, operation_context
from ..storage import numel
from ....tensor import MatrixManTensor

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
    shape, left_offset, right_offset, left_strides, right_strides, left_tex_w, left_tex_h, right_tex_w, right_tex_h, out_tex_w, alpha = params
    shape = tuple(int(value) for value in shape)
    left_terms = []
    right_terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((out_index / {max(1, suffix)}) - "
            f"(out_index / {max(1, suffix * size)}) * {size})"
        )
        left_terms.append(f"({coordinate}) * {int(left_strides[axis])}")
        right_terms.append(f"({coordinate}) * {int(right_strides[axis])}")
    left_index = " + ".join(left_terms) or "0"
    right_index = " + ".join(right_terms) or "0"
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
    int left_index = LEFT_OFFSET + __LEFT_INDEX__;
    int right_index = RIGHT_OFFSET + __RIGHT_INDEX__;
    return read_packed(left_tex, left_index, LEFT_TEX_W, LEFT_TEX_H)
        - ALPHA * read_packed(right_tex, right_index, RIGHT_TEX_W, RIGHT_TEX_H);
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
        "LEFT_OFFSET": left_offset,
        "RIGHT_OFFSET": right_offset,
        "LEFT_TEX_W": left_tex_w,
        "LEFT_TEX_H": left_tex_h,
        "RIGHT_TEX_W": right_tex_w,
        "RIGHT_TEX_H": right_tex_h,
        "OUT_TEX_W": out_tex_w,
        "ALPHA": kernels.glsl_float(alpha),
        "__LEFT_INDEX__": left_index,
        "__RIGHT_INDEX__": right_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_packed_sub(left: "MatrixManTensor", right: "MatrixManTensor", alpha: float = 1.0) -> "MatrixManTensor":
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("gm45 packed sub only supports float32")
    if left.shape != right.shape:
        raise RuntimeError("gm45 packed sub requires equal shapes")
    if left._owner.layout.kind != "packed_rgba" or right._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 packed sub requires packed_rgba input storage")
    shape = tuple(int(v) for v in left.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 packed sub does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
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
        float(alpha),
    )
    program, left_loc, right_loc = _packed_sub_program(params)
    rt = operation_context.gl_runtime()
    diagnostics.trace(
        "gm45.kernel -> packed sub shader:\n"
        f"  left texture #{left._owner.texture} shape={list(left.shape)} offset={left._storage_offset} strides={list(left._logical_strides)}\n"
        f"  right texture #{right._owner.texture} shape={list(right.shape)} offset={right._storage_offset} strides={list(right._logical_strides)}\n"
        f"  alpha={float(alpha):.10g}\n"
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
    return MatrixManTensor._from_owner(out_owner, shape)
def _render_packed_add(left: "MatrixManTensor", right: "MatrixManTensor", alpha: float) -> "MatrixManTensor":
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
    return MatrixManTensor._from_owner(out_owner, shape)


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

def _render_packed_strided_add(left: "MatrixManTensor", right: "MatrixManTensor", alpha: float) -> "MatrixManTensor":
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
    return MatrixManTensor._from_owner(out_owner, shape)

def _packed_scalar_div_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_scalar_div_programs:
        diagnostics.trace(f"gm45.compile -> packed scalar div GLSL fragment shader params={params}")
        program = gm.make_program(_packed_scalar_div_shader_source(params))
        rt.packed_scalar_div_programs[params] = program
        rt.packed_scalar_div_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.packed_scalar_div_programs[params], rt.packed_scalar_div_uniforms[params]

def _packed_scalar_div_shader_source(params: tuple) -> bytes:
    shape, input_offset, input_strides, input_tex_w, input_tex_h, out_tex_w, divisor = params
    shape = tuple(int(value) for value in shape)
    terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((linear_index / {max(1, suffix)}) - "
            f"(linear_index / {max(1, suffix * size)}) * {size})"
        )
        terms.append(f"({coordinate}) * {int(input_strides[axis])}")
    input_index = " + ".join(terms) or "0"
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
    int input_index = INPUT_OFFSET + __INPUT_INDEX__;
    return read_packed(input_index) / DIVISOR;
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
        "NUMEL": numel(shape), "INPUT_OFFSET": int(input_offset),
        "INPUT_TEX_W": int(input_tex_w), "INPUT_TEX_H": int(input_tex_h),
        "OUT_TEX_W": int(out_tex_w), "DIVISOR": kernels.glsl_float(divisor),
        "__INPUT_INDEX__": input_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_packed_scalar_div(tensor: "MatrixManTensor", divisor: float) -> "MatrixManTensor":
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 scalar div supports only float32")
    if tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 scalar div requires packed_rgba input storage")
    shape = tuple(int(v) for v in tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 scalar div does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
    params = (
        shape, tensor._storage_offset, tensor._logical_strides,
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
    return MatrixManTensor._from_owner(out_owner, shape)

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

def _render_packed_broadcast_mul(left: "MatrixManTensor", right: "MatrixManTensor") -> "MatrixManTensor":
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
    return MatrixManTensor._from_owner(out_owner, out_shape)


def _packed_mul_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_mul_programs:
        diagnostics.trace(f"gm45.compile -> packed mul GLSL fragment shader params={params}")
        program = gm.make_program(_packed_mul_shader_source(params))
        rt.packed_mul_programs[params] = program
        rt.packed_mul_uniforms[params] = (
            gm.glGetUniformLocation(program, b"left_tex"),
            gm.glGetUniformLocation(program, b"right_tex"),
        )
    left_loc, right_loc = rt.packed_mul_uniforms[params]
    return rt.packed_mul_programs[params], left_loc, right_loc


def _packed_mul_shader_source(params: tuple) -> bytes:
    shape, left_offset, right_offset, left_strides, right_strides, left_tex_w, left_tex_h, right_tex_w, right_tex_h, out_tex_w = params
    shape = tuple(int(value) for value in shape)
    left_terms = []
    right_terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((out_index / {max(1, suffix)}) - "
            f"(out_index / {max(1, suffix * size)}) * {size})"
        )
        left_terms.append(f"({coordinate}) * {int(left_strides[axis])}")
        right_terms.append(f"({coordinate}) * {int(right_strides[axis])}")
    left_index = " + ".join(left_terms) or "0"
    right_index = " + ".join(right_terms) or "0"
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
    int left_index = LEFT_OFFSET + __LEFT_INDEX__;
    int right_index = RIGHT_OFFSET + __RIGHT_INDEX__;
    return read_packed(left_tex, left_index, LEFT_TEX_W, LEFT_TEX_H)
        * read_packed(right_tex, right_index, RIGHT_TEX_W, RIGHT_TEX_H);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        multiply_at(base),
        multiply_at(base + 1),
        multiply_at(base + 2),
        multiply_at(base + 3)
    );
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape),
        "LEFT_OFFSET": left_offset,
        "RIGHT_OFFSET": right_offset,
        "LEFT_TEX_W": left_tex_w,
        "LEFT_TEX_H": left_tex_h,
        "RIGHT_TEX_W": right_tex_w,
        "RIGHT_TEX_H": right_tex_h,
        "OUT_TEX_W": out_tex_w,
        "__LEFT_INDEX__": left_index,
        "__RIGHT_INDEX__": right_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_packed_mul(left: "MatrixManTensor", right: "MatrixManTensor") -> "MatrixManTensor":
    if left.dtype != torch.float32 or right.dtype != torch.float32:
        raise RuntimeError("gm45 packed mul only supports float32")
    if left.shape != right.shape:
        raise RuntimeError("gm45 packed mul requires equal shapes")
    if left._owner.layout.kind != "packed_rgba" or right._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 packed mul requires packed_rgba input storage")
    shape = tuple(int(value) for value in left.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 packed mul does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
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
    program, left_loc, right_loc = _packed_mul_program(params)
    diagnostics.trace(
        "gm45.kernel -> packed mul shader:\n"
        f"  left texture #{left._owner.texture} shape={list(left.shape)} offset={left._storage_offset} strides={list(left._logical_strides)}\n"
        f"  right texture #{right._owner.texture} shape={list(right.shape)} offset={right._storage_offset} strides={list(right._logical_strides)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(shape)} offset=0"
    )
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 packed mul framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in ((gm.GL_TEXTURE0, left, left_loc), (gm.GL_TEXTURE1, right, right_loc)):
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after packed mul: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, shape)


def _packed_scalar_mul_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_scalar_mul_programs:
        diagnostics.trace(f"gm45.compile -> packed scalar mul GLSL fragment shader params={params}")
        program = gm.make_program(_packed_scalar_mul_shader_source(params))
        rt.packed_scalar_mul_programs[params] = program
        rt.packed_scalar_mul_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.packed_scalar_mul_programs[params], rt.packed_scalar_mul_uniforms[params]


def _packed_scalar_mul_shader_source(params: tuple) -> bytes:
    shape, input_offset, input_strides, input_tex_w, input_tex_h, out_tex_w, scalar = params
    shape = tuple(int(value) for value in shape)
    terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((linear_index / {max(1, suffix)}) - "
            f"(linear_index / {max(1, suffix * size)}) * {size})"
        )
        terms.append(f"({coordinate}) * {int(input_strides[axis])}")
    input_index = " + ".join(terms) or "0"
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

float multiply_at(int linear_index)
{
    if (linear_index >= NUMEL) return 0.0;
    int input_index = INPUT_OFFSET + __INPUT_INDEX__;
    return read_packed(input_index) * SCALAR;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        multiply_at(base), multiply_at(base + 1),
        multiply_at(base + 2), multiply_at(base + 3)
    );
}
"""
    replacements = {
        "NUMEL": numel(shape),
        "INPUT_OFFSET": int(input_offset),
        "INPUT_TEX_W": int(input_tex_w),
        "INPUT_TEX_H": int(input_tex_h),
        "OUT_TEX_W": int(out_tex_w),
        "SCALAR": kernels.glsl_float(scalar),
        "__INPUT_INDEX__": input_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_packed_scalar_mul(tensor: "MatrixManTensor", scalar: float) -> "MatrixManTensor":
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 scalar mul supports only float32")
    if tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 scalar mul requires packed_rgba input storage")
    shape = tuple(int(value) for value in tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 scalar mul does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
    params = (
        shape, tensor._storage_offset, tensor._logical_strides,
        tensor._owner.layout.texture_width, tensor._owner.layout.texture_height,
        out_owner.layout.texture_width, float(scalar),
    )
    program, input_loc = _packed_scalar_mul_program(params)
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 scalar mul framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after scalar mul: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, shape)


def _replace_inplace_tensor(target: "MatrixManTensor", result: "MatrixManTensor") -> "MatrixManTensor":
    target._owner = result._owner
    target._shape = tuple(result.shape)
    target._storage_offset = 0
    target._logical_strides = result._logical_strides
    return target


def _packed_sqrt_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_sqrt_programs:
        diagnostics.trace(f"gm45.compile -> packed sqrt GLSL fragment shader params={params}")
        program = gm.make_program(_packed_sqrt_shader_source(params))
        rt.packed_sqrt_programs[params] = program
        rt.packed_sqrt_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.packed_sqrt_programs[params], rt.packed_sqrt_uniforms[params]


def _packed_sqrt_shader_source(params: tuple) -> bytes:
    shape, input_offset, input_strides, input_tex_w, input_tex_h, out_tex_w = params
    shape = tuple(int(value) for value in shape)
    terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((linear_index / {max(1, suffix)}) - "
            f"(linear_index / {max(1, suffix * size)}) * {size})"
        )
        terms.append(f"({coordinate}) * {int(input_strides[axis])}")
    input_index = " + ".join(terms) or "0"
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

float sqrt_at(int linear_index)
{
    if (linear_index >= NUMEL) return 0.0;
    int input_index = INPUT_OFFSET + __INPUT_INDEX__;
    return sqrt(read_packed(input_index));
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        sqrt_at(base), sqrt_at(base + 1),
        sqrt_at(base + 2), sqrt_at(base + 3)
    );
}
"""
    replacements = {
        "NUMEL": numel(shape),
        "INPUT_OFFSET": int(input_offset),
        "INPUT_TEX_W": int(input_tex_w),
        "INPUT_TEX_H": int(input_tex_h),
        "OUT_TEX_W": int(out_tex_w),
        "__INPUT_INDEX__": input_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_packed_sqrt(tensor: "MatrixManTensor") -> "MatrixManTensor":
    if not isinstance(tensor, MatrixManTensor):
        raise RuntimeError("gm45 sqrt requires a MatrixManTensor input")
    if tensor.dtype != torch.float32:
        raise RuntimeError("gm45 sqrt supports only float32")
    if tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 sqrt requires packed_rgba input storage")
    shape = tuple(int(value) for value in tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 sqrt does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
    params = (
        shape, tensor._storage_offset, tensor._logical_strides,
        tensor._owner.layout.texture_width, tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _packed_sqrt_program(params)
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 sqrt framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after sqrt: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, shape)


def _packed_lerp_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_lerp_programs:
        diagnostics.trace(f"gm45.compile -> packed lerp GLSL fragment shader params={params}")
        program = gm.make_program(_packed_lerp_shader_source(params))
        rt.packed_lerp_programs[params] = program
        rt.packed_lerp_uniforms[params] = (
            gm.glGetUniformLocation(program, b"self_tex"),
            gm.glGetUniformLocation(program, b"end_tex"),
        )
    self_loc, end_loc = rt.packed_lerp_uniforms[params]
    return rt.packed_lerp_programs[params], self_loc, end_loc


def _packed_lerp_shader_source(params: tuple) -> bytes:
    shape, self_offset, end_offset, self_strides, end_strides, self_tex_w, self_tex_h, end_tex_w, end_tex_h, out_tex_w, weight = params
    shape = tuple(int(value) for value in shape)
    self_terms = []
    end_terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((out_index / {max(1, suffix)}) - "
            f"(out_index / {max(1, suffix * size)}) * {size})"
        )
        self_terms.append(f"({coordinate}) * {int(self_strides[axis])}")
        end_terms.append(f"({coordinate}) * {int(end_strides[axis])}")
    self_index = " + ".join(self_terms) or "0"
    end_index = " + ".join(end_terms) or "0"
    source = """
#version 120
uniform sampler2D self_tex;
uniform sampler2D end_tex;

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

float lerp_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int self_index = SELF_OFFSET + __SELF_INDEX__;
    int end_index = END_OFFSET + __END_INDEX__;
    float self_value = read_packed(self_tex, self_index, SELF_TEX_W, SELF_TEX_H);
    float end_value = read_packed(end_tex, end_index, END_TEX_W, END_TEX_H);
    return self_value + WEIGHT * (end_value - self_value);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        lerp_at(base),
        lerp_at(base + 1),
        lerp_at(base + 2),
        lerp_at(base + 3)
    );
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape),
        "SELF_OFFSET": int(self_offset),
        "END_OFFSET": int(end_offset),
        "SELF_TEX_W": int(self_tex_w),
        "SELF_TEX_H": int(self_tex_h),
        "END_TEX_W": int(end_tex_w),
        "END_TEX_H": int(end_tex_h),
        "OUT_TEX_W": int(out_tex_w),
        "WEIGHT": kernels.glsl_float(weight),
        "__SELF_INDEX__": self_index,
        "__END_INDEX__": end_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_packed_lerp_inplace(self_tensor: "MatrixManTensor", end: "MatrixManTensor", weight: float) -> "MatrixManTensor":
    """Render scalar lerp into fresh storage, then update the lhs wrapper."""
    if not isinstance(self_tensor, MatrixManTensor) or not isinstance(end, MatrixManTensor):
        raise RuntimeError("gm45 lerp_ requires MatrixManTensor operands")
    if self_tensor.dtype != torch.float32 or end.dtype != torch.float32:
        raise RuntimeError("gm45 lerp_ only supports float32")
    if self_tensor.shape != end.shape:
        raise RuntimeError("gm45 lerp_ requires equal shapes")
    if self_tensor._owner.layout.kind != "packed_rgba" or end._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 lerp_ requires packed_rgba operands")
    shape = tuple(int(value) for value in self_tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 lerp_ does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
    params = (
        shape,
        self_tensor._storage_offset,
        end._storage_offset,
        self_tensor._logical_strides,
        end._logical_strides,
        self_tensor._owner.layout.texture_width,
        self_tensor._owner.layout.texture_height,
        end._owner.layout.texture_width,
        end._owner.layout.texture_height,
        out_owner.layout.texture_width,
        float(weight),
    )
    program, self_loc, end_loc = _packed_lerp_program(params)
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 lerp_ framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in (
        (gm.GL_TEXTURE0, self_tensor, self_loc),
        (gm.GL_TEXTURE1, end, end_loc),
    ):
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after lerp_: 0x{err:04x}")
    self_tensor._owner = out_owner
    self_tensor._storage_offset = 0
    self_tensor._logical_strides = tuple(
        1 if axis == len(shape) - 1 else numel(shape[axis + 1:])
        for axis in range(len(shape))
    )
    return self_tensor


def _packed_addcmul_program(params: tuple) -> tuple[int, int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_addcmul_programs:
        diagnostics.trace(f"gm45.compile -> packed addcmul GLSL fragment shader params={params}")
        program = gm.make_program(_packed_addcmul_shader_source(params))
        rt.packed_addcmul_programs[params] = program
        rt.packed_addcmul_uniforms[params] = (
            gm.glGetUniformLocation(program, b"self_tex"),
            gm.glGetUniformLocation(program, b"tensor1_tex"),
            gm.glGetUniformLocation(program, b"tensor2_tex"),
        )
    uniforms = rt.packed_addcmul_uniforms[params]
    return rt.packed_addcmul_programs[params], *uniforms


def _packed_addcmul_shader_source(params: tuple) -> bytes:
    shape, offsets, strides, tex_sizes, out_tex_w, value = params
    shape = tuple(int(v) for v in shape)
    indices = []
    for offset, tensor_strides in zip(offsets, strides):
        terms = []
        for axis, size in enumerate(shape):
            suffix = numel(shape[axis + 1:])
            coordinate = (
                f"((out_index / {max(1, suffix)}) - "
                f"(out_index / {max(1, suffix * size)}) * {size})"
            )
            terms.append(f"({coordinate}) * {int(tensor_strides[axis])}")
        indices.append((int(offset), " + ".join(terms) or "0"))
    source = """
#version 120
uniform sampler2D self_tex;
uniform sampler2D tensor1_tex;
uniform sampler2D tensor2_tex;

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

float addcmul_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    float self_value = read_packed(self_tex, SELF_OFFSET + __SELF_INDEX__, SELF_TEX_W, SELF_TEX_H);
    float tensor1_value = read_packed(tensor1_tex, TENSOR1_OFFSET + __TENSOR1_INDEX__, TENSOR1_TEX_W, TENSOR1_TEX_H);
    float tensor2_value = read_packed(tensor2_tex, TENSOR2_OFFSET + __TENSOR2_INDEX__, TENSOR2_TEX_W, TENSOR2_TEX_H);
    return self_value + VALUE * tensor1_value * tensor2_value;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        addcmul_at(base), addcmul_at(base + 1),
        addcmul_at(base + 2), addcmul_at(base + 3)
    );
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape),
        "SELF_OFFSET": indices[0][0], "__SELF_INDEX__": indices[0][1],
        "TENSOR1_OFFSET": indices[1][0], "__TENSOR1_INDEX__": indices[1][1],
        "TENSOR2_OFFSET": indices[2][0], "__TENSOR2_INDEX__": indices[2][1],
        "SELF_TEX_W": tex_sizes[0][0], "SELF_TEX_H": tex_sizes[0][1],
        "TENSOR1_TEX_W": tex_sizes[1][0], "TENSOR1_TEX_H": tex_sizes[1][1],
        "TENSOR2_TEX_W": tex_sizes[2][0], "TENSOR2_TEX_H": tex_sizes[2][1],
        "OUT_TEX_W": out_tex_w,
        "VALUE": kernels.glsl_float(value),
    }
    for name, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(replacement))
    return source.encode("ascii")


def _render_packed_addcmul_inplace(
    self_tensor: "MatrixManTensor",
    tensor1: "MatrixManTensor",
    tensor2: "MatrixManTensor",
    value: float,
) -> "MatrixManTensor":
    tensors = (self_tensor, tensor1, tensor2)
    if any(not isinstance(tensor, MatrixManTensor) for tensor in tensors):
        raise RuntimeError("gm45 addcmul_ requires three MatrixManTensor operands")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError("gm45 addcmul_ only supports float32")
    if not (self_tensor.shape == tensor1.shape == tensor2.shape):
        raise RuntimeError("gm45 addcmul_ currently requires equal shapes")
    if any(tensor._owner.layout.kind != "packed_rgba" for tensor in tensors):
        raise RuntimeError("gm45 addcmul_ requires packed_rgba operands")
    shape = tuple(int(v) for v in self_tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 addcmul_ does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
    params = (
        shape,
        tuple(tensor._storage_offset for tensor in tensors),
        tuple(tensor._logical_strides for tensor in tensors),
        tuple((tensor._owner.layout.texture_width, tensor._owner.layout.texture_height) for tensor in tensors),
        out_owner.layout.texture_width,
        float(value),
    )
    program, self_loc, tensor1_loc, tensor2_loc = _packed_addcmul_program(params)
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 addcmul_ framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in (
        (gm.GL_TEXTURE0, self_tensor, self_loc),
        (gm.GL_TEXTURE1, tensor1, tensor1_loc),
        (gm.GL_TEXTURE2, tensor2, tensor2_loc),
    ):
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after addcmul_: 0x{err:04x}")
    return _replace_inplace_tensor(self_tensor, MatrixManTensor._from_owner(out_owner, shape))


def _packed_addcdiv_program(params: tuple) -> tuple[int, int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.packed_addcdiv_programs:
        diagnostics.trace(f"gm45.compile -> packed addcdiv GLSL fragment shader params={params}")
        program = gm.make_program(_packed_addcdiv_shader_source(params))
        rt.packed_addcdiv_programs[params] = program
        rt.packed_addcdiv_uniforms[params] = (
            gm.glGetUniformLocation(program, b"self_tex"),
            gm.glGetUniformLocation(program, b"tensor1_tex"),
            gm.glGetUniformLocation(program, b"tensor2_tex"),
        )
    return rt.packed_addcdiv_programs[params], *rt.packed_addcdiv_uniforms[params]


def _packed_addcdiv_shader_source(params: tuple) -> bytes:
    shape, offsets, strides, tex_sizes, out_tex_w, value = params
    shape = tuple(int(v) for v in shape)
    indices = []
    for offset, tensor_strides in zip(offsets, strides):
        terms = []
        for axis, size in enumerate(shape):
            suffix = numel(shape[axis + 1:])
            coordinate = (
                f"((out_index / {max(1, suffix)}) - "
                f"(out_index / {max(1, suffix * size)}) * {size})"
            )
            terms.append(f"({coordinate}) * {int(tensor_strides[axis])}")
        indices.append((int(offset), " + ".join(terms) or "0"))
    source = """
#version 120
uniform sampler2D self_tex;
uniform sampler2D tensor1_tex;
uniform sampler2D tensor2_tex;

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

float addcdiv_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    float self_value = read_packed(self_tex, SELF_OFFSET + __SELF_INDEX__, SELF_TEX_W, SELF_TEX_H);
    float tensor1_value = read_packed(tensor1_tex, TENSOR1_OFFSET + __TENSOR1_INDEX__, TENSOR1_TEX_W, TENSOR1_TEX_H);
    float tensor2_value = read_packed(tensor2_tex, TENSOR2_OFFSET + __TENSOR2_INDEX__, TENSOR2_TEX_W, TENSOR2_TEX_H);
    return self_value + VALUE * tensor1_value / tensor2_value;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        addcdiv_at(base), addcdiv_at(base + 1),
        addcdiv_at(base + 2), addcdiv_at(base + 3)
    );
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape),
        "SELF_OFFSET": indices[0][0], "__SELF_INDEX__": indices[0][1],
        "TENSOR1_OFFSET": indices[1][0], "__TENSOR1_INDEX__": indices[1][1],
        "TENSOR2_OFFSET": indices[2][0], "__TENSOR2_INDEX__": indices[2][1],
        "SELF_TEX_W": tex_sizes[0][0], "SELF_TEX_H": tex_sizes[0][1],
        "TENSOR1_TEX_W": tex_sizes[1][0], "TENSOR1_TEX_H": tex_sizes[1][1],
        "TENSOR2_TEX_W": tex_sizes[2][0], "TENSOR2_TEX_H": tex_sizes[2][1],
        "OUT_TEX_W": out_tex_w,
        "VALUE": kernels.glsl_float(value),
    }
    for name, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(replacement))
    return source.encode("ascii")


def _render_packed_addcdiv_inplace(
    self_tensor: "MatrixManTensor",
    tensor1: "MatrixManTensor",
    tensor2: "MatrixManTensor",
    value: float,
) -> "MatrixManTensor":
    tensors = (self_tensor, tensor1, tensor2)
    if any(not isinstance(tensor, MatrixManTensor) for tensor in tensors):
        raise RuntimeError("gm45 addcdiv_ requires three MatrixManTensor operands")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise RuntimeError("gm45 addcdiv_ only supports float32")
    if not (self_tensor.shape == tensor1.shape == tensor2.shape):
        raise RuntimeError("gm45 addcdiv_ currently requires equal shapes")
    if any(tensor._owner.layout.kind != "packed_rgba" for tensor in tensors):
        raise RuntimeError("gm45 addcdiv_ requires packed_rgba operands")
    shape = tuple(int(v) for v in self_tensor.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 addcdiv_ does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
    params = (
        shape,
        tuple(tensor._storage_offset for tensor in tensors),
        tuple(tensor._logical_strides for tensor in tensors),
        tuple((tensor._owner.layout.texture_width, tensor._owner.layout.texture_height) for tensor in tensors),
        out_owner.layout.texture_width,
        float(value),
    )
    program, self_loc, tensor1_loc, tensor2_loc = _packed_addcdiv_program(params)
    operation_context.attach_output(out_owner)
    if gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER) != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("gm45 addcdiv_ framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in (
        (gm.GL_TEXTURE0, self_tensor, self_loc),
        (gm.GL_TEXTURE1, tensor1, tensor1_loc),
        (gm.GL_TEXTURE2, tensor2, tensor2_loc),
    ):
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after addcdiv_: 0x{err:04x}")
    return _replace_inplace_tensor(self_tensor, MatrixManTensor._from_owner(out_owner, shape))

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

def _render_scalar_add(tensor: "MatrixManTensor", scalar: float, alpha: float, *, tensor_first: bool) -> "MatrixManTensor":
    if not isinstance(tensor, MatrixManTensor):
        raise RuntimeError("gm45 scalar add requires one MatrixManTensor input")
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
    return MatrixManTensor._from_owner(out_owner, shape)
