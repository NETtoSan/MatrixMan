"""Experimental GPU-side detection postprocessing reductions."""

from __future__ import annotations

import torch

from .. import diagnostics, gpumatrix as gm, operation_context, profiling
from ..tensor import Gm45Tensor


def _shader_source(channels: int, anchors: int, input_width: int, input_height: int,
                   output_width: int) -> bytes:
    classes = channels - 4
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

float read_input(int linear_index)
{
    int texel = linear_index / 4;
    int component = linear_index - texel * 4;
    int x = texel - (texel / INPUT_WIDTH) * INPUT_WIDTH;
    int y = texel / INPUT_WIDTH;
    vec2 uv = (vec2(float(x), float(y)) + vec2(0.5, 0.5)) /
              vec2(float(INPUT_WIDTH), float(INPUT_HEIGHT));
    return pick_component(texture2D(input_tex, uv), component);
}

float reduced_at(int linear_index)
{
    if (linear_index >= OUTPUT_NUMEL) return 0.0;
    int anchor = linear_index - (linear_index / ANCHORS) * ANCHORS;
    int channel = linear_index / ANCHORS;
    if (channel < 4) return read_input(channel * ANCHORS + anchor);

    float best_score = -1.0;
    int best_class = 0;
    for (int cls = 0; cls < CLASS_COUNT; ++cls) {
        float score = read_input((4 + cls) * ANCHORS + anchor);
        if (score > best_score) {
            best_score = score;
            best_class = cls;
        }
    }
    if (channel == 4) return best_score;
    return float(best_class);
}

void main()
{
    int tex_x = int(floor(gl_FragCoord.x));
    int tex_y = int(floor(gl_FragCoord.y));
    int base = (tex_y * OUTPUT_WIDTH + tex_x) * 4;
    gl_FragColor = vec4(reduced_at(base), reduced_at(base + 1),
                        reduced_at(base + 2), reduced_at(base + 3));
}
"""
    replacements = {
        "CLASS_COUNT": classes, "ANCHORS": anchors,
        "INPUT_WIDTH": input_width, "INPUT_HEIGHT": input_height,
        "OUTPUT_WIDTH": output_width, "OUTPUT_NUMEL": 6 * anchors,
    }
    for name, value in replacements.items():
        source = source.replace(name, str(value))
    return source.encode("ascii")


def _program(params: tuple) -> tuple[int, int]:
    rt = operation_context.gl_runtime()
    if params not in rt.postprocess_programs:
        channels, anchors, input_width, input_height, output_width = params
        diagnostics.trace(f"gm45.compile -> detection reduction GLSL shader params={params}")
        program = gm.make_program(_shader_source(channels, anchors, input_width, input_height, output_width))
        rt.postprocess_programs[params] = program
        rt.postprocess_uniforms[params] = gm.glGetUniformLocation(program, b"input_tex")
    return rt.postprocess_programs[params], rt.postprocess_uniforms[params]


def reduce_detection_output(tensor: "Gm45Tensor") -> "Gm45Tensor":
    """Reduce [1, 4+classes, anchors] to [1, 6, anchors] on the GPU."""
    if not isinstance(tensor, Gm45Tensor):
        raise RuntimeError("gm45 detection reduction requires a Gm45Tensor")
    if tensor.dtype != torch.float32 or tensor._owner.layout.kind != "packed_rgba":
        raise RuntimeError("gm45 detection reduction requires packed float32 storage")
    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] < 6 or shape[2] <= 0:
        raise RuntimeError("gm45 detection reduction requires shape [1,4+classes,anchors]")
    operation_context.require_contiguous(tensor, "detection reduction")
    channels, anchors = shape[1], shape[2]
    out_shape = (1, 6, anchors)
    out_owner = operation_context.output_texture(out_shape)
    params = (channels, anchors, tensor._owner.layout.texture_width,
              tensor._owner.layout.texture_height, out_owner.layout.texture_width)
    program, input_loc = _program(params)
    diagnostics.trace(
        "gm45.kernel -> detection reduction shader:\n"
        f"  input texture #{tensor._owner.texture} shape={list(shape)}\n"
        f"  -> output texture #{out_owner.texture} shape={list(out_shape)} "
        "channels=[cx,cy,w,h,best_confidence,best_class]"
    )
    operation_context.attach_output(out_owner)
    operation_context.framebuffer_complete("gm45 detection reduction framebuffer incomplete")
    gm.glUseProgram(program)
    gm.glActiveTexture(gm.GL_TEXTURE0)
    gm.glBindTexture(gm.GL_TEXTURE_2D, tensor._owner.texture)
    gm.glUniform1i(input_loc, 0)
    with profiling.gpu_timer("postprocess reduction"):
        operation_context.draw_fullscreen_quad()
    if (err := gm.glGetError()):
        raise RuntimeError(f"gm45 OpenGL error after detection reduction: 0x{err:04x}")
    return Gm45Tensor._from_owner(out_owner, out_shape)
