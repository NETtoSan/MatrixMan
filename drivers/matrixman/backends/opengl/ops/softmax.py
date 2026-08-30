"""OpenGL DFL-oriented softmax operation."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context
from ..tensor import Gm45Tensor

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

def _render_softmax(args) -> "Gm45Tensor":
    input_tensor = args[0]
    dim = int(args[1])
    half_to_float = bool(args[2]) if len(args) > 2 else False
    if not isinstance(input_tensor, Gm45Tensor):
        raise RuntimeError("gm45 softmax requires a Gm45Tensor input")
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
    return Gm45Tensor._from_owner(out_owner, out_shape)

