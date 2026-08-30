"""OpenGL max-pooling operation."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context
from ..tensor import Gm45Tensor

def _maxpool_program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.maxpool_programs:
        diagnostics.trace(f"gm45.compile -> max_pool2d GLSL fragment shader params={params}")
        program = gm.make_program(_maxpool_shader_source(params))
        rt.maxpool_programs[params] = program
        rt.maxpool_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.maxpool_programs[params], rt.maxpool_uniforms[params]

def _maxpool_shader_source(params: tuple) -> bytes:
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

float pool_at(int out_index)
{
    if (out_index >= OUT_NUMEL) return 0.0;
    int ox = out_index - (out_index / OUT_W) * OUT_W;
    int tmp0 = out_index / OUT_W;
    int oy = tmp0 - (tmp0 / OUT_H) * OUT_H;
    int c = tmp0 / OUT_H;
    float best = -3.402823e+38;

    for (int ky = 0; ky < 5; ++ky) {
        int iy = oy + ky - 2;
        if (iy >= 0 && iy < IN_H) {
            for (int kx = 0; kx < 5; ++kx) {
                int ix = ox + kx - 2;
                if (ix >= 0 && ix < IN_W) {
                    int source_index = INPUT_OFFSET + ((c * IN_H) + iy) * IN_W + ix;
                    best = max(best, read_packed(source_index));
                }
            }
        }
    }
    return best;
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUT_TEX_W + tex_x) * 4;
    gl_FragColor = vec4(
        pool_at(base),
        pool_at(base + 1),
        pool_at(base + 2),
        pool_at(base + 3)
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

def _render_max_pool2d_with_indices(args) -> tuple["Gm45Tensor", torch.Tensor]:
    input_tensor = args[0]
    kernel_size = _as_pair(args[1], "kernel_size")
    stride = _as_pair(args[2], "stride")
    padding = _as_pair(args[3], "padding")
    dilation = _as_pair(args[4], "dilation") if len(args) > 4 else (1, 1)
    ceil_mode = bool(args[5]) if len(args) > 5 else False

    if not isinstance(input_tensor, Gm45Tensor):
        raise RuntimeError("gm45 max_pool2d requires a Gm45Tensor input")
    if input_tensor.dtype != torch.float32:
        raise RuntimeError("gm45 max_pool2d supports only float32")
    if input_tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 max_pool2d currently supports only packed_rgba tensor storage")
    operation_context.require_contiguous(input_tensor, "max_pool2d")
    if len(input_tensor.shape) != 4 or int(input_tensor.shape[0]) != 1:
        raise RuntimeError("gm45 max_pool2d supports only batch-1 NCHW 4D tensors")
    if kernel_size != (5, 5) or stride != (1, 1) or padding != (2, 2):
        raise RuntimeError("gm45 max_pool2d currently supports only kernel=5, stride=1, padding=2")
    if dilation != (1, 1):
        raise RuntimeError("gm45 max_pool2d currently supports dilation=(1,1) only")
    if ceil_mode:
        raise RuntimeError("gm45 max_pool2d currently supports ceil_mode=False only")

    _, channels, in_h, in_w = (int(v) for v in input_tensor.shape)
    out_h = (in_h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    out_shape = (1, channels, out_h, out_w)
    out_owner = operation_context.output_texture(out_shape)
    params = (
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        input_tensor._storage_offset,
        input_tensor._owner.layout.texture_width,
        input_tensor._owner.layout.texture_height,
        out_owner.layout.texture_width,
    )
    program, input_loc = _maxpool_program(params)
    rt = operation_context.gl_runtime()

    diagnostics.trace(
        "gm45.kernel -> max_pool2d values shader:\n"
        f"  input texture #{input_tensor._owner.texture} shape={list(input_tensor.shape)} "
        f"offset={input_tensor._storage_offset}\n"
        "  kernel=[5,5] stride=[1,1] padding=[2,2]\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} offset=0\n"
        "  indices: CPU empty int64 placeholder; YOLO MaxPool2d(return_indices=False) does not consume it"
    )

    operation_context.attach_output(out_owner)
    status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
    if status != gm.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"gm45 max_pool2d framebuffer incomplete: 0x{status:04x}")

    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, input_tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)

    operation_context.draw_fullscreen_quad()
    diagnostics.trace(f"gm45.opengl -> submitted max_pool2d fullscreen quad, output texture #{out_owner.texture}")

    err = gm.glGetError()
    if err:
        raise RuntimeError(f"gm45 OpenGL error after max_pool2d: 0x{err:04x}")
    values = Gm45Tensor._from_owner(out_owner, out_shape)
    indices = torch.empty((0,), dtype=torch.int64, device="cpu")
    return values, indices

def _as_pair(value, name: str) -> tuple[int, int]:
    """Normalize the shared two-dimensional operator argument form."""
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise RuntimeError(f"gm45 convolution expects {name} as int or pair")
