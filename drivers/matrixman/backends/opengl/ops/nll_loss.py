"""Packed GPU NLL loss for classification logits and integer targets."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context
from ..storage import numel
from ....tensor import MatrixManTensor


def _program(params):
    rt = operation_context.gl_runtime()
    if params not in rt.nll_programs:
        diagnostics.trace(f"gm45.compile -> NLL loss GLSL fragment shader params={params}")
        program = gm.make_program(_shader_source(params))
        rt.nll_programs[params] = program
        rt.nll_uniforms[params] = (
            gm.glGetUniformLocation(program, b"input_tex"),
            gm.glGetUniformLocation(program, b"target_tex"),
        )
    input_loc, target_loc = rt.nll_uniforms[params]
    return rt.nll_programs[params], input_loc, target_loc


def _shader_source(params):
    batch, classes, input_offset, input_strides, input_w, input_h, target_offset, target_stride, target_w, target_h, out_w, reduction, ignore_index, mode = params
    source = """
#version 120
uniform sampler2D input_tex;
uniform sampler2D target_tex;

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

int target_at(int batch_index)
{
    return int(floor(read_packed(target_tex, TARGET_OFFSET + batch_index * TARGET_STRIDE,
                                 TARGET_TEX_W, TARGET_TEX_H) + 0.5));
}

float selected_loss(int batch_index)
{
    int target_index = target_at(batch_index);
    if (target_index == IGNORE_INDEX || target_index < 0 || target_index >= CLASSES) return 0.0;
    int input_index = INPUT_OFFSET + batch_index * INPUT_STRIDE_BATCH
        + target_index * INPUT_STRIDE_CLASS;
    return -read_packed(input_tex, input_index, INPUT_TEX_W, INPUT_TEX_H);
}

float valid_count()
{
    float count = 0.0;
    for (int batch_index = 0; batch_index < BATCH; ++batch_index)
        if (target_at(batch_index) != IGNORE_INDEX) count += 1.0;
    return count;
}

float loss_at(int output_index)
{
    if (REDUCTION == 0) return selected_loss(output_index);
    float total = 0.0;
    for (int batch_index = 0; batch_index < BATCH; ++batch_index) total += selected_loss(batch_index);
    if (REDUCTION == 1) {
        float count = valid_count();
        return count > 0.0 ? total / count : 0.0;
    }
    return total;
}

float weight_at(int output_index)
{
    if (output_index > 0) return 0.0;
    return valid_count();
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        __VALUE_FN__(base), __VALUE_FN__(base + 1),
        __VALUE_FN__(base + 2), __VALUE_FN__(base + 3)
    );
}
"""
    replacements = {
        "BATCH": batch, "CLASSES": classes, "INPUT_OFFSET": input_offset,
        "INPUT_STRIDE_BATCH": input_strides[0], "INPUT_STRIDE_CLASS": input_strides[1],
        "INPUT_TEX_W": input_w, "INPUT_TEX_H": input_h,
        "TARGET_OFFSET": target_offset, "TARGET_STRIDE": target_stride,
        "TARGET_TEX_W": target_w, "TARGET_TEX_H": target_h,
        "OUT_TEX_W": out_w, "REDUCTION": reduction, "IGNORE_INDEX": ignore_index,
        "__VALUE_FN__": "loss_at" if mode == "loss" else "weight_at",
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render(args):
    if len(args) < 5:
        raise RuntimeError("gm45 nll_loss_forward requires input, target, weight, reduction, and ignore_index")
    input_tensor, target, weight, reduction, ignore_index = args[:5]
    if not isinstance(input_tensor, MatrixManTensor) or not isinstance(target, MatrixManTensor):
        raise RuntimeError("gm45 nll_loss_forward requires MatrixManTensor input and target")
    if input_tensor.dtype != torch.float32 or target.dtype != torch.int64:
        raise RuntimeError("gm45 nll_loss_forward requires float32 logits and int64 targets")
    if weight is not None:
        raise RuntimeError("gm45 nll_loss_forward currently supports weight=None only")
    if input_tensor.dim() != 2 or target.dim() != 1 or input_tensor.shape[0] != target.shape[0]:
        raise RuntimeError("gm45 nll_loss_forward supports input [B,C] and target [B]")
    if input_tensor._owner.layout.kind != "packed_rgba" or target._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 nll_loss_forward requires packed_rgba storage")
    integer_values = getattr(target._owner, "_integer_values", None)
    if integer_values is not None:
        invalid = (integer_values != int(ignore_index)) & ((integer_values < 0) | (integer_values >= input_tensor.shape[1]))
        if torch.any(invalid):
            raise RuntimeError("gm45 nll_loss_forward target contains an out-of-range class index")
    if int(reduction) not in (0, 1, 2):
        raise RuntimeError(f"gm45 nll_loss_forward unsupported reduction={reduction}")
    operation_context.require_contiguous(input_tensor, "nll_loss_forward")
    operation_context.require_contiguous(target, "nll_loss_forward")
    batch, classes = (int(input_tensor.shape[0]), int(input_tensor.shape[1]))
    out_shape = (batch,) if int(reduction) == 0 else ()

    def render(mode, shape, scalar_name):
        owner = operation_context.output_texture(shape or (1,))
        params = (
            batch, classes, input_tensor._storage_offset, input_tensor._logical_strides,
            input_tensor._owner.layout.texture_width, input_tensor._owner.layout.texture_height,
            target._storage_offset, target._logical_strides[0],
            target._owner.layout.texture_width, target._owner.layout.texture_height,
            owner.layout.texture_width, int(reduction), int(ignore_index), mode,
        )
        program, input_loc, target_loc = _program(params)
        operation_context.attach_output(owner)
        operation_context.framebuffer_complete(f"gm45 nll_loss {scalar_name} framebuffer incomplete")
        gm.glUseProgram(program)
        for unit, tensor, uniform in ((gm.GL_TEXTURE0, input_tensor, input_loc), (gm.GL_TEXTURE1, target, target_loc)):
            gm.glActiveTexture(unit)
            gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
            gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
        operation_context.draw_fullscreen_quad()
        return MatrixManTensor._from_owner(owner, shape)

    output = render("loss", out_shape, "output")
    total_weight = render("weight", (), "total_weight")
    return output, total_weight


def _backward_program(params):
    rt = operation_context.gl_runtime()
    if params not in rt.nll_backward_programs:
        diagnostics.trace(f"gm45.compile -> NLL backward GLSL fragment shader params={params}")
        program = gm.make_program(_backward_shader_source(params))
        rt.nll_backward_programs[params] = program
        rt.nll_backward_uniforms[params] = (
            gm.glGetUniformLocation(program, b"grad_output_tex"),
            gm.glGetUniformLocation(program, b"target_tex"),
            gm.glGetUniformLocation(program, b"total_weight_tex"),
        )
    grad_loc, target_loc, total_loc = rt.nll_backward_uniforms[params]
    return rt.nll_backward_programs[params], grad_loc, target_loc, total_loc


def _backward_shader_source(params):
    batch, classes, target_offset, target_stride, target_w, target_h, grad_offset, grad_w, grad_h, total_offset, total_w, total_h, out_w, reduction, ignore_index = params
    source = """
#version 120
uniform sampler2D grad_output_tex;
uniform sampler2D target_tex;
uniform sampler2D total_weight_tex;

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

int target_at(int batch_index)
{
    return int(floor(read_packed(target_tex, TARGET_OFFSET + batch_index * TARGET_STRIDE,
                                 TARGET_TEX_W, TARGET_TEX_H) + 0.5));
}

float upstream_at(int batch_index)
{
    return read_packed(grad_output_tex,
        GRAD_OFFSET + (REDUCTION == 0 ? batch_index : 0), GRAD_TEX_W, GRAD_TEX_H);
}

float grad_at(int output_index)
{
    if (output_index >= BATCH * CLASSES) return 0.0;
    int batch_index = output_index / CLASSES;
    int class_index = output_index - batch_index * CLASSES;
    int target_index = target_at(batch_index);
    if (target_index == IGNORE_INDEX || target_index < 0 || target_index >= CLASSES || class_index != target_index)
        return 0.0;
    float scale = upstream_at(batch_index);
    if (REDUCTION == 1) {
        float total_weight = read_packed(total_weight_tex, TOTAL_OFFSET, TOTAL_TEX_W, TOTAL_TEX_H);
        scale = total_weight > 0.0 ? scale / total_weight : 0.0;
    }
    return -scale;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(grad_at(base), grad_at(base + 1), grad_at(base + 2), grad_at(base + 3));
}
"""
    replacements = {
        "BATCH": batch, "CLASSES": classes, "TARGET_OFFSET": target_offset,
        "TARGET_STRIDE": target_stride, "TARGET_TEX_W": target_w, "TARGET_TEX_H": target_h,
        "GRAD_OFFSET": grad_offset, "GRAD_TEX_W": grad_w, "GRAD_TEX_H": grad_h,
        "TOTAL_OFFSET": total_offset, "TOTAL_TEX_W": total_w, "TOTAL_TEX_H": total_h,
        "OUT_TEX_W": out_w, "REDUCTION": reduction, "IGNORE_INDEX": ignore_index,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _render_backward(args):
    if len(args) < 7:
        raise RuntimeError("gm45 nll_loss_backward requires grad_output, input, target, weight, reduction, ignore_index, total_weight")
    grad_output, input_tensor, target, weight, reduction, ignore_index, total_weight = args[:7]
    if not isinstance(grad_output, MatrixManTensor) or not isinstance(input_tensor, MatrixManTensor) or not isinstance(target, MatrixManTensor) or not isinstance(total_weight, MatrixManTensor):
        raise RuntimeError("gm45 nll_loss_backward requires MatrixManTensor operands")
    if grad_output.dtype != torch.float32 or input_tensor.dtype != torch.float32 or target.dtype != torch.int64 or total_weight.dtype != torch.float32:
        raise RuntimeError("gm45 nll_loss_backward requires float32 tensors and int64 targets")
    if weight is not None:
        raise RuntimeError("gm45 nll_loss_backward currently supports weight=None only")
    if input_tensor.dim() != 2 or target.dim() != 1 or input_tensor.shape[0] != target.shape[0]:
        raise RuntimeError("gm45 nll_loss_backward supports input [B,C] and target [B]")
    if int(reduction) not in (0, 1, 2):
        raise RuntimeError(f"gm45 nll_loss_backward unsupported reduction={reduction}")
    for tensor, name in ((grad_output, "grad_output"), (input_tensor, "input"), (target, "target"), (total_weight, "total_weight")):
        if tensor._owner.layout.kind != "packed_rgba":
            raise RuntimeError(f"gm45 nll_loss_backward {name} requires packed_rgba storage")
        operation_context.require_contiguous(tensor, f"nll_loss_backward {name}")
    integer_values = getattr(target._owner, "_integer_values", None)
    if integer_values is not None:
        invalid = (integer_values != int(ignore_index)) & ((integer_values < 0) | (integer_values >= input_tensor.shape[1]))
        if torch.any(invalid):
            raise RuntimeError("gm45 nll_loss_backward target contains an out-of-range class index")
    shape = tuple(int(value) for value in input_tensor.shape)
    out_owner = operation_context.output_texture(shape)
    params = (
        shape[0], shape[1], target._storage_offset, target._logical_strides[0],
        target._owner.layout.texture_width, target._owner.layout.texture_height,
        grad_output._storage_offset, grad_output._owner.layout.texture_width,
        grad_output._owner.layout.texture_height, total_weight._storage_offset,
        total_weight._owner.layout.texture_width, total_weight._owner.layout.texture_height,
        out_owner.layout.texture_width, int(reduction), int(ignore_index),
    )
    program, grad_loc, target_loc, total_loc = _backward_program(params)
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 nll_loss_backward framebuffer incomplete")
    gm.glUseProgram(program)
    for unit, tensor, uniform in ((gm.GL_TEXTURE0, grad_output, grad_loc), (gm.GL_TEXTURE1, target, target_loc), (gm.GL_TEXTURE2, total_weight, total_loc)):
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)
    operation_context.draw_fullscreen_quad()
    return MatrixManTensor._from_owner(out_owner, shape)
