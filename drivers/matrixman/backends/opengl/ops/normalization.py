"""OpenGL inference BatchNorm operation."""

from __future__ import annotations

import numpy as np
import torch

from .. import diagnostics, gpumatrix as gm, operation_context
from ..storage import packed_atlas_size
from ..tensor import Gm45Tensor

def _batchnorm_program(params: tuple) -> tuple[int, int, int, int, int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.batchnorm_programs:
        diagnostics.trace(f"gm45.compile -> batchnorm GLSL fragment shader params={params}")
        program = gm.make_program(_batchnorm_shader_source(params))
        rt.batchnorm_programs[params] = program
        rt.batchnorm_uniforms[params] = (
            gm.glGetUniformLocation(program, b"input_tex"),
            gm.glGetUniformLocation(program, b"weight_tex"),
            gm.glGetUniformLocation(program, b"bias_tex"),
            gm.glGetUniformLocation(program, b"mean_tex"),
            gm.glGetUniformLocation(program, b"var_tex"),
        )
    return (rt.batchnorm_programs[params], *rt.batchnorm_uniforms[params])

def _batchnorm_shader_source(params: tuple) -> bytes:
    channels, height, width, eps, input_offset, input_tex_w, input_tex_h, out_tex_w, param_tex_w = params
    source = f"""
#version 120
uniform sampler2D input_tex;
uniform sampler2D weight_tex;
uniform sampler2D bias_tex;
uniform sampler2D mean_tex;
uniform sampler2D var_tex;

float pick_component(vec4 value, int component)
{{
    if (component == 0) return value.r;
    if (component == 1) return value.g;
    if (component == 2) return value.b;
    return value.a;
}}

float read_packed(sampler2D tex, int linear_index, int tex_width, int tex_height)
{{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / tex_width) * tex_width;
    int y = texel / tex_width;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(tex_width), float(tex_height));
    return pick_component(texture2D(tex, uv), component);
}}

float compute_output(int out_index)
{{
    if (out_index >= __OUT_NUMEL__) return 0.0;
    int spatial = __H__ * __W__;
    int channel = (out_index / spatial) - ((out_index / spatial) / __C__) * __C__;
    float x = read_packed(input_tex, out_index + __INPUT_OFFSET__, __INPUT_TEX_W__, __INPUT_TEX_H__);
    float gamma = read_packed(weight_tex, channel, __PARAM_TEX_W__, __PARAM_TEX_H__);
    float beta = read_packed(bias_tex, channel, __PARAM_TEX_W__, __PARAM_TEX_H__);
    float mean = read_packed(mean_tex, channel, __PARAM_TEX_W__, __PARAM_TEX_H__);
    float var = read_packed(var_tex, channel, __PARAM_TEX_W__, __PARAM_TEX_H__);
    return ((x - mean) / sqrt(var + __EPS__)) * gamma + beta;
}}

void main()
{{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * __OUT_TEX_W__ + tex_x) * 4;
    gl_FragColor = vec4(
        compute_output(base),
        compute_output(base + 1),
        compute_output(base + 2),
        compute_output(base + 3)
    );
}}
"""
    replacements = {
        "__C__": channels,
        "__H__": height,
        "__W__": width,
        "__EPS__": f"{float(eps):.10g}",
        "__OUT_NUMEL__": channels * height * width,
        "__INPUT_OFFSET__": input_offset,
        "__INPUT_TEX_W__": input_tex_w,
        "__INPUT_TEX_H__": input_tex_h,
        "__OUT_TEX_W__": out_tex_w,
        "__PARAM_TEX_W__": param_tex_w,
        "__PARAM_TEX_H__": packed_atlas_size(channels)[1],
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")

def _render_batch_norm(args):
    input_tensor, weight, bias, running_mean, running_var, training, momentum, eps = args[:8]
    del momentum
    if training:
        raise RuntimeError("gm45 native_batch_norm supports inference/eval mode only")
    if not isinstance(input_tensor, Gm45Tensor):
        raise RuntimeError("gm45 native_batch_norm requires input to be a Gm45Tensor")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 native_batch_norm requires packed_rgba input storage")
    operation_context.require_contiguous(input_tensor, "native_batch_norm")
    if len(input_tensor.shape) != 4 or int(input_tensor.shape[0]) != 1:
        raise RuntimeError("gm45 native_batch_norm supports batch-1 NCHW 4D tensors only")

    _, channels, height, width = (int(v) for v in input_tensor.shape)
    params = []
    for name, value, default in [
        ("weight", weight, 1.0),
        ("bias", bias, 0.0),
        ("running_mean", running_mean, 0.0),
        ("running_var", running_var, 1.0),
    ]:
        if value is None:
            params.append(torch.full((channels,), default, dtype=torch.float32))
            continue
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise RuntimeError(f"gm45 native_batch_norm {name} must be a CPU tensor")
        if value.dtype != torch.float32 or not value.is_contiguous() or tuple(value.shape) != (channels,):
            raise RuntimeError(f"gm45 native_batch_norm {name} must be contiguous float32 shape [{channels}]")
        params.append(value)

    weight_owner = operation_context.upload_raw_parameter(params[0].detach().numpy().astype(np.float32, copy=False), "bn_weight")
    bias_owner = operation_context.upload_raw_parameter(params[1].detach().numpy().astype(np.float32, copy=False), "bn_bias")
    mean_owner = operation_context.upload_raw_parameter(params[2].detach().numpy().astype(np.float32, copy=False), "bn_mean")
    var_owner = operation_context.upload_raw_parameter(params[3].detach().numpy().astype(np.float32, copy=False), "bn_var")
    out_owner = operation_context.output_texture(tuple(int(v) for v in input_tensor.shape))

    shader_params = (
        channels,
        height,
        width,
        float(eps),
        input_tensor._storage_offset,
        input_tensor._owner.layout.texture_width,
        input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
        weight_owner.layout.texture_width,
    )
    program, input_loc, weight_loc, bias_loc, mean_loc, var_loc = _batchnorm_program(shader_params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> native_batch_norm inference shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(input_tensor.shape)}\n"
        f"  params textures weight=#{weight_owner.texture} bias=#{bias_owner.texture} "
        f"mean=#{mean_owner.texture} var=#{var_owner.texture}\n"
        f"  -> output texture #{out_owner.texture} shape={list(input_tensor.shape)}"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 batch_norm framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    for unit, texture, uniform in [
        (gm.GL_TEXTURE0, input_tensor._owner.texture, input_loc),
        (gm.GL_TEXTURE1, weight_owner.texture, weight_loc),
        (gm.GL_TEXTURE2, bias_owner.texture, bias_loc),
        (gm.GL_TEXTURE3, mean_owner.texture, mean_loc),
        (gm.GL_TEXTURE4, var_owner.texture, var_loc),
    ]:
        gm.glActiveTexture(unit)
        gm.glBindTexture(gm.GL_TEXTURE_2D, texture)
        gm.glUniform1i(uniform, unit - gm.GL_TEXTURE0)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted BatchNorm fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after batch_norm: 0x{err:04x}")
    output = Gm45Tensor._from_owner(out_owner, tuple(int(v) for v in input_tensor.shape))
    return output, torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

