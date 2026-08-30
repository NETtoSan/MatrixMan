"""OpenGL nearest-neighbor resize operation."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context


def _upsample_nearest2d_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.upsample_programs:
        diagnostics.trace(f"gm45.compile -> upsample_nearest2d GLSL fragment shader params={params}")
        program = gm.make_program(_upsample_nearest2d_shader_source(params))
        rt.upsample_programs[params] = program
        rt.upsample_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.upsample_programs[params], rt.upsample_uniforms[params]


def _upsample_nearest2d_shader_source(params: tuple) -> bytes:
    channels, in_h, in_w, out_h, out_w, input_offset, input_tex_w, input_tex_h, out_tex_w = params
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
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) / vec2(float(INPUT_TEX_W), float(INPUT_TEX_H));
    return pick_component(texture2D(input_tex, uv), component);
}

float upsample_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int ox = out_index - (out_index / OUT_W) * OUT_W;
    int tmp0 = out_index / OUT_W;
    int oy = tmp0 - (tmp0 / OUT_H) * OUT_H;
    int c = tmp0 / OUT_H;
    int ix = ox / 2;
    int iy = oy / 2;
    int source_index = INPUT_OFFSET + ((c * IN_H) + iy) * IN_W + ix;
    return read_packed(source_index);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        upsample_at(base),
        upsample_at(base + 1),
        upsample_at(base + 2),
        upsample_at(base + 3)
    );
}
"""
    replacements = {
        "OUT_NUMEL": channels * out_h * out_w,
        "INPUT_OFFSET": input_offset,
        "INPUT_TEX_W": input_tex_w,
        "INPUT_TEX_H": input_tex_h,
        "OUT_TEX_W": out_tex_w,
        "IN_H": in_h,
        "IN_W": in_w,
        "OUT_H": out_h,
        "OUT_W": out_w,
    }
    for name, value in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        source = source.replace(name, str(value))
    return source.encode("ascii")


def render_upsample_nearest2d(args):
    input_tensor = args[0]
    output_size = tuple(int(v) for v in args[1])
    scale_h = args[2] if len(args) > 2 else None
    scale_w = args[3] if len(args) > 3 else None

    if not isinstance(input_tensor, operation_context.Gm45Tensor):
        raise RuntimeError("gm45 upsample_nearest2d requires a Gm45Tensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 upsample_nearest2d supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 upsample_nearest2d currently supports only packed_rgba tensor storage")
    operation_context.require_contiguous(input_tensor, "upsample_nearest2d")
    if len(input_tensor.shape) != 4 or int(input_tensor.shape[0]) != 1:
        raise RuntimeError("gm45 upsample_nearest2d supports only batch-1 NCHW 4D tensors")
    if len(output_size) != 2:
        raise RuntimeError("gm45 upsample_nearest2d expects a 2D output_size")

    _, channels, in_h, in_w = (int(v) for v in input_tensor.shape)
    out_h, out_w = output_size
    if out_h != in_h * 2 or out_w != in_w * 2:
        raise RuntimeError("gm45 upsample_nearest2d currently supports only exact 2x spatial scaling")
    if scale_h is not None and float(scale_h) != 2.0:
        raise RuntimeError("gm45 upsample_nearest2d currently supports only scale_h=2.0")
    if scale_w is not None and float(scale_w) != 2.0:
        raise RuntimeError("gm45 upsample_nearest2d currently supports only scale_w=2.0")

    out_shape = (1, channels, out_h, out_w)
    out_owner = operation_context.output_texture(out_shape)
    params = (
        channels, in_h, in_w, out_h, out_w, input_tensor._storage_offset,
        input_tensor._owner.layout.texture_width, input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _upsample_nearest2d_program(params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> upsample_nearest2d shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(input_tensor.shape)} "
        f"offset={input_tensor._storage_offset}\n"
        f"  output_size={list(output_size)} scale_h={scale_h} scale_w={scale_w}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 upsample_nearest2d framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted upsample_nearest2d fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after upsample_nearest2d: 0x{err:04x}")
    return operation_context.tensor_from_owner(out_owner, out_shape)
