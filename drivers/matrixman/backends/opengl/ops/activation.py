"""OpenGL activation operations."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, kernels, operation_context, profiling
from ..storage import numel
from ....tensor import MatrixManTensor

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

def _render_packed_sigmoid(tensor: "MatrixManTensor") -> "MatrixManTensor":
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
    return MatrixManTensor._from_owner(out_owner, out_shape)

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

def _render_silu_inplace_unguarded(args) -> "MatrixManTensor":
    input_tensor = args[0]
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError("gm45 silu_ requires a MatrixManTensor input")
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
    status = int(gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER))
    framebuffer_error = int(gm.glGetError())
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        try:
            current_context = gm.sdl.SDL_GL_GetCurrentContext()
        except Exception as exc:
            current_context = f"<query failed: {exc}>"
        try:
            from .. import runtime
            expected_context = getattr(runtime._runtime, "context", None)
            fbo = getattr(runtime._runtime, "fbo", None)
            fbo_id = getattr(fbo, "value", fbo)
        except Exception:
            expected_context = None
            fbo_id = None

        def _gl_text(value):
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else value

        raise RuntimeError(
            "gm45 SiLU framebuffer incomplete: "
            f"status=0x{status:04x} expected=0x{gm.GL_FRAMEBUFFER_COMPLETE:04x} "
            f"gl_error_after_status=0x{framebuffer_error:04x} "
            f"current_context={current_context!r} expected_context={expected_context!r} "
            f"fbo={fbo_id!r} output_texture={out_owner.texture} "
            f"texture_size={out_owner.layout.texture_width}x{out_owner.layout.texture_height} "
            f"renderer={_gl_text(gm.glGetString(0x1F01))!r} "
            f"GL_VERSION={_gl_text(gm.glGetString(0x1F02))!r} "
            f"GLSL_VERSION={_gl_text(gm.glGetString(0x8B8C))!r}"
        )

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


def _render_silu_inplace(args) -> "MatrixManTensor":
    """Render SiLU while owning MatrixMan's shared OpenGL context."""
    from .. import runtime

    if runtime.is_active():
        with runtime.gl_execution_context():
            return _render_silu_inplace_unguarded(args)
    return _render_silu_inplace_unguarded(args)


def _relu_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.relu_programs:
        diagnostics.trace(f"gm45.compile -> ReLU GLSL fragment shader params={params}")
        program = gm.make_program(_relu_shader_source(params))
        rt.relu_programs[params] = program
        rt.relu_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.relu_programs[params], rt.relu_uniforms[params]


def _relu_shader_source(params: tuple) -> bytes:
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
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(__INPUT_TEX_W__), float(__INPUT_TEX_H__));
    return pick_component(texture2D(input_tex, uv), component);
}

float relu_at(int linear_index)
{
    if (linear_index >= __NUMEL__) return 0.0;
    return max(read_packed(linear_index + __INPUT_OFFSET__), 0.0);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * __OUT_TEX_W__ + tex_x) * 4;
    gl_FragColor = vec4(
        relu_at(base),
        relu_at(base + 1),
        relu_at(base + 2),
        relu_at(base + 3)
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


def _render_relu(input_tensor: "MatrixManTensor", *, inplace: bool) -> "MatrixManTensor":
    operation = "relu_" if inplace else "relu"
    if not isinstance(input_tensor, MatrixManTensor):
        raise RuntimeError(f"gm45 {operation} requires a MatrixManTensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError(f"gm45 {operation} supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError(f"gm45 {operation} requires packed_rgba tensor storage")
    operation_context.require_contiguous(input_tensor, operation)

    shape = tuple(int(v) for v in input_tensor.shape)
    out_owner = operation_context.output_texture(shape)
    params = (
        numel(shape),
        input_tensor._storage_offset,
        input_tensor._owner.layout.texture_width,
        input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _relu_program(params)
    diagnostics.trace(
        f"gm45.kernel -> ReLU{' in-place' if inplace else ''} shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(shape)} "
        f"offset={input_tensor._storage_offset}\n"
        f"  -> output texture #{out_owner.texture} shape={list(shape)} offset=0"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 ReLU framebuffer incomplete: 0x{status:04x}")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    with profiling.gpu_timer("ReLU"):
        operation_context.draw_fullscreen_quad()
    diagnostics.trace(
        f"gm45.opengl -> submitted ReLU{' in-place' if inplace else ''} fullscreen quad, "
        f"output texture #{out_owner.texture}"
    )
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after {operation}: 0x{err:04x}")

    if inplace:
        input_tensor._owner = out_owner
        input_tensor._shape = shape
        input_tensor._storage_offset = 0
        return input_tensor
    return MatrixManTensor._from_owner(out_owner, shape)


def _render_relu_inplace(args) -> "MatrixManTensor":
    return _render_relu(args[0], inplace=True)


def _render_relu_functional(args) -> "MatrixManTensor":
    return _render_relu(args[0], inplace=False)


def _threshold_backward_program(params: tuple) -> tuple[int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.threshold_backward_programs:
        diagnostics.trace(f"gm45.compile -> threshold backward GLSL fragment shader params={params}")
        program = gm.make_program(_threshold_backward_shader_source(params))
        rt.threshold_backward_programs[params] = program
        rt.threshold_backward_uniforms[params] = (
            gm.glGetUniformLocation(program, b"grad_tex"),
            gm.glGetUniformLocation(program, b"self_tex"),
        )
    grad_loc, self_loc = rt.threshold_backward_uniforms[params]
    return rt.threshold_backward_programs[params], grad_loc, self_loc


def _threshold_backward_shader_source(params: tuple) -> bytes:
    (
        shape, grad_offset, grad_strides, grad_tex_w, grad_tex_h,
        self_offset, self_strides, self_tex_w, self_tex_h, out_tex_w, threshold,
    ) = params
    shape = tuple(int(value) for value in shape)
    grad_terms = []
    self_terms = []
    for axis, size in enumerate(shape):
        suffix = numel(shape[axis + 1:])
        coordinate = (
            f"((out_index / {max(1, suffix)}) - "
            f"(out_index / {max(1, suffix * size)}) * {size})"
        )
        grad_terms.append(f"({coordinate}) * {int(grad_strides[axis])}")
        self_terms.append(f"({coordinate}) * {int(self_strides[axis])}")
    grad_index = " + ".join(grad_terms) or "0"
    self_index = " + ".join(self_terms) or "0"
    source = """
#version 120
uniform sampler2D grad_tex;
uniform sampler2D self_tex;

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

float threshold_backward_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int grad_index = GRAD_OFFSET + __GRAD_INDEX__;
    int self_index = SELF_OFFSET + __SELF_INDEX__;
    float x = read_packed(self_tex, self_index, SELF_TEX_W, SELF_TEX_H);
    float g = read_packed(grad_tex, grad_index, GRAD_TEX_W, GRAD_TEX_H);
    return x > THRESHOLD ? g : 0.0;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        threshold_backward_at(base),
        threshold_backward_at(base + 1),
        threshold_backward_at(base + 2),
        threshold_backward_at(base + 3)
    );
}
"""
    replacements = {
        "OUT_NUMEL": numel(shape),
        "GRAD_OFFSET": grad_offset,
        "SELF_OFFSET": self_offset,
        "GRAD_TEX_W": grad_tex_w,
        "GRAD_TEX_H": grad_tex_h,
        "SELF_TEX_W": self_tex_w,
        "SELF_TEX_H": self_tex_h,
        "OUT_TEX_W": out_tex_w,
        "THRESHOLD": kernels.glsl_float(threshold),
        "__GRAD_INDEX__": grad_index,
        "__SELF_INDEX__": self_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_threshold_backward(args) -> "MatrixManTensor":
    if len(args) < 3:
        raise RuntimeError("gm45 threshold_backward requires grad_output, self, and threshold")
    grad_output, self_tensor, threshold = args[:3]
    if not isinstance(grad_output, MatrixManTensor) or not isinstance(self_tensor, MatrixManTensor):
        raise RuntimeError("gm45 threshold_backward requires MatrixManTensor operands")
    if grad_output.dtype != torch.float32 or self_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 threshold_backward supports only float32")
    if grad_output.shape != self_tensor.shape:
        raise RuntimeError("gm45 threshold_backward requires equal shapes")
    if grad_output._owner.layout.kind != "packed_rgba" or self_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 threshold_backward requires packed_rgba input storage")
    if isinstance(threshold, torch.Tensor):
        if threshold.device.type != "cpu" or threshold.numel() != 1:
            raise RuntimeError("gm45 threshold_backward threshold must be a numeric scalar")
        threshold = threshold.item()
    if not isinstance(threshold, (int, float)):
        raise RuntimeError("gm45 threshold_backward threshold must be a numeric scalar")
    shape = tuple(int(value) for value in grad_output.shape)
    if numel(shape) <= 0:
        raise RuntimeError("gm45 threshold_backward does not support empty tensors")
    out_owner = operation_context.output_texture(shape or (1,))
    params = (
        shape, grad_output._storage_offset, grad_output._logical_strides,
        grad_output._owner.layout.texture_width, grad_output._owner.layout.texture_height,
        self_tensor._storage_offset, self_tensor._logical_strides,
        self_tensor._owner.layout.texture_width, self_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width, float(threshold),
    )
    program, grad_loc, self_loc = _threshold_backward_program(params)
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 threshold_backward framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in ((gm.GL_TEXTURE0, grad_output, grad_loc), (gm.GL_TEXTURE1, self_tensor, self_loc)):
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after threshold_backward: 0x{err:04x}")
    return MatrixManTensor._from_owner(out_owner, shape)
