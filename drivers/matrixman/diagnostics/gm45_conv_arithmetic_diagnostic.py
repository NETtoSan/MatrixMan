#!/usr/bin/env python3
"""Standalone GM45 convolution arithmetic probes.

This does not call MatrixMan's convolution implementation.  It uses the
backend's packed upload/readback helpers so the address path is identical.
"""
from __future__ import annotations

import math
import ctypes
import sys
import numpy as np
import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman import gm45_backend as backend
from drivers.matrixman import gpumatrix as gl

gl.gl.glScissor.restype = None
gl.gl.glScissor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
gl.gl.glEnable.restype = None
gl.gl.glEnable.argtypes = [ctypes.c_uint]
gl.gl.glDisable.restype = None
gl.gl.glDisable.argtypes = [ctypes.c_uint]
gl.gl.glColorMask.restype = None
gl.gl.glColorMask.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte]
GL_SCISSOR_TEST = 0x0C11


def shader_source(in_c, in_h, in_w, input_tw, input_th, weight_tw, weight_th, body):
    return f"""
#version 120
uniform sampler2D input_tex;
uniform sampler2D weight_tex;
float pick(vec4 v, int c) {{ if(c==0)return v.r; if(c==1)return v.g; if(c==2)return v.b; return v.a; }}
float read_tex(sampler2D tex, int index, int width, int height) {{
  int texel=index/4; int component=index-texel*4;
  int x=texel-(texel/width)*width; int y=texel/width;
  vec2 uv=(vec2(float(x),float(y))+vec2(0.5,0.5))/vec2(float(width),float(height));
  return pick(texture2D(tex,uv),component);
}}
float input_at(int ic,int y,int x) {{ return read_tex(input_tex,((ic*{in_h}+y)*{in_w}+x),{input_tw},{input_th}); }}
float weight_at(int oc,int ic,int ky,int kx) {{ return read_tex(weight_tex,(((oc*{in_c}+ic)*3+ky)*3+kx),{weight_tw},{weight_th}); }}
void main() {{ {body} }}
""".encode("ascii")


def render_scalar(gx, gw, input_dims, weight_dims, body):
    in_c, in_h, in_w = input_dims
    input_tw, input_th = backend._packed_atlas_size(gx._owner.layout.numel)
    weight_tw, weight_th = backend._packed_atlas_size(gw._owner.layout.numel)
    owner = backend._new_empty_packed_texture((1,))
    rt = backend._runtime_required()
    program = gl.make_program(shader_source(in_c, in_h, in_w, input_tw, input_th, weight_tw, weight_th, body))
    try:
        gl.glViewport(0, 0, 1, 1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, rt.fbo.value)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, owner.texture, 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("arithmetic diagnostic framebuffer incomplete")
        gl.glUseProgram(program)
        for unit, tensor, name in ((gl.GL_TEXTURE0, gx, b"input_tex"), (gl.GL_TEXTURE1, gw, b"weight_tex")):
            gl.glActiveTexture(unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tensor._owner.texture)
            gl.glUniform1i(gl.glGetUniformLocation(program, name), unit - gl.GL_TEXTURE0)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1,-1); gl.glVertex2f(1,-1); gl.glVertex2f(1,1); gl.glVertex2f(-1,1)
        gl.glEnd()
        if (error := gl.glGetError()):
            raise RuntimeError(f"arithmetic diagnostic OpenGL error: 0x{error:04x}")
        return backend.Gm45Tensor._from_owner(owner, (1,)).cpu().item()
    finally:
        gl.glDeleteProgram(program)


def render_constant(iterations, looped):
    owner = backend._new_empty_packed_texture((1,))
    rt = backend._runtime_required()
    if looped:
        body = f"float acc=0.0; for(int i=0;i<{iterations};++i) acc+=0.125; gl_FragColor=vec4(acc,0,0,0);"
    else:
        body = "float acc=0.0; " + " ".join("acc+=0.125;" for _ in range(iterations)) + " gl_FragColor=vec4(acc,0,0,0);"
    program = gl.make_program(f"#version 120\nvoid main(){{{body}}}".encode("ascii"))
    try:
        gl.glViewport(0,0,1,1); gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,rt.fbo.value)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,gl.GL_COLOR_ATTACHMENT0,gl.GL_TEXTURE_2D,owner.texture,0)
        gl.glUseProgram(program); gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1,-1); gl.glVertex2f(1,-1); gl.glVertex2f(1,1); gl.glVertex2f(-1,1); gl.glEnd()
        return backend.Gm45Tensor._from_owner(owner,(1,)).cpu().item()
    finally:
        gl.glDeleteProgram(program)


def render_full_conv(x, weight):
    """Run the existing full-output shader source without dispatching aten.convolution."""
    gx = gm45.to_gm45(x)
    weight_owner = backend._upload_raw_packed_array(weight.numpy())
    out_shape = (1, weight.shape[0], x.shape[2], x.shape[3])
    out_owner = backend._new_empty_packed_texture(tuple(out_shape))
    params = (x.shape[1], x.shape[2], x.shape[3], weight.shape[0], x.shape[2], x.shape[3],
              3, 3, 1, 1, 1, 1, False, 1, gx._storage_offset,
              gx._owner.layout.texture_width, gx._owner.layout.texture_height,
              weight_owner.layout.texture_width, weight_owner.layout.texture_height,
              weight_owner.layout.texture_width, out_owner.layout.texture_width)
    program, input_loc, weight_loc, bias_loc = backend._conv_program(params)
    rt = backend._runtime_required()
    try:
        gl.glViewport(0, 0, out_owner.layout.texture_width, out_owner.layout.texture_height)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, rt.fbo.value)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, out_owner.texture, 0)
        gl.glUseProgram(program)
        gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, gx._owner.texture); gl.glUniform1i(input_loc, 0)
        gl.glActiveTexture(gl.GL_TEXTURE1); gl.glBindTexture(gl.GL_TEXTURE_2D, weight_owner.texture); gl.glUniform1i(weight_loc, 1)
        gl.glActiveTexture(gl.GL_TEXTURE2); gl.glBindTexture(gl.GL_TEXTURE_2D, weight_owner.texture); gl.glUniform1i(bias_loc, 2)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1,-1); gl.glVertex2f(1,-1); gl.glVertex2f(1,1); gl.glVertex2f(-1,1); gl.glEnd()
        if (error := gl.glGetError()):
            raise RuntimeError(f"full arithmetic diagnostic OpenGL error: 0x{error:04x}")
        return backend.Gm45Tensor._from_owner(out_owner, out_shape).cpu(), gx, weight_owner
    except Exception:
        gl.glDeleteTextures(1, backend.ctypes.byref(weight_owner.texture))
        raise


def render_full_dispatch(x, weight, *, tiles=1, reset=False, out_owner=None):
    """Production shader dispatch with optional scissor tiling/state reset."""
    gx = gm45.to_gm45(x)
    weight_owner = backend._upload_raw_packed_array(weight.numpy())
    out_shape = (1, weight.shape[0], x.shape[2], x.shape[3])
    if out_owner is None:
        out_owner = backend._new_empty_packed_texture(tuple(out_shape))
    in_tw, in_th = gx._owner.layout.texture_width, gx._owner.layout.texture_height
    params = (x.shape[1], x.shape[2], x.shape[3], weight.shape[0], x.shape[2], x.shape[3], 3, 3,
              1, 1, 1, 1, False, 1, gx._storage_offset, in_tw, in_th,
              weight_owner.layout.texture_width, weight_owner.layout.texture_height,
              weight_owner.layout.texture_width, out_owner.layout.texture_width)
    program, input_loc, weight_loc, bias_loc = backend._conv_program(params)
    rt = backend._runtime_required()
    width, height = out_owner.layout.texture_width, out_owner.layout.texture_height
    if reset:
        gl.gl.glDisable(0x0BE2)  # GL_BLEND
        gl.gl.glDisable(0x0B71)  # GL_DEPTH_TEST
        gl.gl.glDisable(GL_SCISSOR_TEST)
        gl.gl.glDisable(0x0BC0)  # GL_ALPHA_TEST
        gl.gl.glDisable(0x0BD0)  # GL_DITHER
        gl.gl.glColorMask(True, True, True, True)
    gl.glViewport(0, 0, width, height)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, rt.fbo.value)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, out_owner.texture, 0)
    complete = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
    gl.glUseProgram(program)
    gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, gx._owner.texture); gl.glUniform1i(input_loc, 0)
    gl.glActiveTexture(gl.GL_TEXTURE1); gl.glBindTexture(gl.GL_TEXTURE_2D, weight_owner.texture); gl.glUniform1i(weight_loc, 1)
    gl.glActiveTexture(gl.GL_TEXTURE2); gl.glBindTexture(gl.GL_TEXTURE_2D, weight_owner.texture); gl.glUniform1i(bias_loc, 2)
    if tiles == 1:
        regions = [(0, 0, width, height)]
    else:
        regions = [(0, y, width, min(height // tiles, height - y)) for y in range(0, height, max(1, height // tiles))]
        regions = regions[:tiles]
        gl.gl.glEnable(GL_SCISSOR_TEST)
    for x0, y0, tw, th in regions:
        if tiles != 1:
            gl.gl.glScissor(x0, y0, tw, th)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(-1,-1); gl.glVertex2f(1,-1); gl.glVertex2f(1,1); gl.glVertex2f(-1,1); gl.glEnd()
    if tiles != 1:
        gl.gl.glDisable(GL_SCISSOR_TEST)
    error = gl.glGetError()
    actual = backend.Gm45Tensor._from_owner(out_owner, out_shape).cpu()
    print(f"dispatch state: fbo={rt.fbo.value} complete={complete} viewport={width}x{height} active_texture=GL_TEXTURE2 input_tex={gx._owner.texture} weight_tex={weight_owner.texture} output_tex={out_owner.texture} program={program} samplers=(0,1,2) gl_error=0x{error:04x}")
    return actual, out_owner


def report(label, expected, actual):
    expected, actual = float(expected), float(actual)
    err = abs(expected-actual)
    print(f"{label}: expected={expected:.9g} actual={actual:.9g} error={err:.6g} allclose={math.isclose(expected,actual,rel_tol=1e-5,abs_tol=1e-5)}")


def setup(in_c, out_c=1, h=32, w=32):
    torch.manual_seed(10000+in_c+out_c+h)
    x=torch.randn((1,in_c,h,w),dtype=torch.float32)*0.1
    weight=torch.randn((out_c,in_c,3,3),dtype=torch.float32)*0.1
    return x,weight,gm45.to_gm45(x),gm45.to_gm45(weight.reshape(-1))


def tensor_probe(label, in_c, terms, looped=False, h=32, w=32):
    x,weight,gx,gw=setup(in_c,1,h,w); cy=cx=h//2
    expected=sum(x[0,ic,cy+ky-1,cx+kx-1]*weight[0,ic,ky,kx] for ic,ky,kx in terms)
    if looped:
        body=f"float acc=0.0; for(int t=0;t<{len(terms)};++t){{"
        for t,(ic,ky,kx) in enumerate(terms):
            body+=f"if(t=={t}) acc+=input_at({ic},{cy+ky-1},{cx+kx-1})*weight_at(0,{ic},{ky},{kx});"
        body+="} gl_FragColor=vec4(acc,0,0,0);"
    else:
        expr=" + ".join(f"input_at({ic},{cy+ky-1},{cx+kx-1})*weight_at(0,{ic},{ky},{kx})" for ic,ky,kx in terms)
        body=f"gl_FragColor=vec4({expr},0,0,0);"
    report(label,expected,render_scalar(gx,gw,(in_c,h,w),weight.shape[1:],body))


def main():
    gm45.set_trace(False)
    if "--dispatch-only" in sys.argv:
        x,w,_,_=setup(64,64,128,128); expected=F.conv2d(x,w,padding=1)
        tiles=int(sys.argv[sys.argv.index("--tiles")+1]) if "--tiles" in sys.argv else 1
        reset="--reset" in sys.argv
        actual,owner=render_full_dispatch(x,w,tiles=tiles,reset=reset)
        delta=(actual-expected).abs()
        print(f"dispatch-only tiles={tiles} reset={reset}: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g} rmse={delta.square().mean().sqrt().item():.6g} allclose={torch.allclose(actual,expected,rtol=1e-4,atol=1e-4)}")
        if "--reuse" in sys.argv:
            actual,_=render_full_dispatch(x,w,tiles=tiles,reset=reset,out_owner=owner)
            delta=(actual-expected).abs()
            print(f"dispatch-only reused texture: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g} rmse={delta.square().mean().sqrt().item():.6g} allclose={torch.allclose(actual,expected,rtol=1e-4,atol=1e-4)}")
        gm45.shutdown()
        return
    print("GM45 convolution arithmetic diagnostic; MatrixMan convolution is not called")
    print("EXPERIMENT 1: single product")
    x,w,gx,gw=setup(8,8,8,8)
    for oc,ky,kx in ((0,1,1),(3,0,2),(7,2,0)):
        expected=x[0,0,4+ky-1,4+kx-1]*w[oc,0,ky,kx]
        body=f"gl_FragColor=vec4(input_at(0,{4+ky-1},{4+kx-1})*weight_at({oc},0,{ky},{kx}),0,0,0);"
        report(f"single oc={oc} ky={ky} kx={kx}",expected,render_scalar(gx,gw,(8,8,8),w.shape[1:],body))
    terms=[(ic,ky,kx) for ic in range(8) for ky in range(3) for kx in range(3)]
    print("EXPERIMENT 2: fixed unrolled sums")
    for n in (2,4,8,16): tensor_probe(f"unrolled terms={n}",8,terms[:n])
    print("EXPERIMENT 3: unrolled versus genuine loop")
    tensor_probe("unrolled terms=8",8,terms[:8]); tensor_probe("looped terms=8",8,terms[:8],True)
    print("EXPERIMENT 4: channel loop only, center kernel")
    for c in (8,16,32,64):
        x,w,gx,gw=setup(c); expected=sum(x[0,ic,16,16]*w[0,ic,1,1] for ic in range(c))
        body=f"float acc=0.0; for(int ic=0;ic<{c};++ic) acc+=input_at(ic,16,16)*weight_at(0,ic,1,1); gl_FragColor=vec4(acc,0,0,0);"
        report(f"channel loop Cin={c}",expected,render_scalar(gx,gw,(c,32,32),w.shape[1:],body))
    print("EXPERIMENT 5: kernel loop only, Cin=1, 9 terms")
    x,w,gx,gw=setup(1); expected=F.conv2d(x,w,padding=1)[0,0,16,16]
    body="float acc=0.0; for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) acc+=input_at(0,16+ky-1,16+kx-1)*weight_at(0,0,ky,kx); gl_FragColor=vec4(acc,0,0,0);"
    report("kernel loop Cin=1 MACs=9",expected,render_scalar(gx,gw,(1,32,32),w.shape[1:],body))
    print("EXPERIMENT 6: full MAC loops")
    for c in (8,16,32,64):
        x,w,gx,gw=setup(c); expected=F.conv2d(x,w,padding=1)[0,0,16,16]
        body=f"float acc=0.0; for(int ic=0;ic<{c};++ic) for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) acc+=input_at(ic,16+ky-1,16+kx-1)*weight_at(0,ic,ky,kx); gl_FragColor=vec4(acc,0,0,0);"
        report(f"full loops Cin={c} MACs={c*9}",expected,render_scalar(gx,gw,(c,32,32),w.shape[1:],body))
    print("EXPERIMENT 7: texture-free accumulator")
    for n in (8,16,32,64,72,128,144,256,288,512,576):
        report(f"constant additions={n} unrolled",n*.125,render_constant(n,False)); report(f"constant additions={n} looped",n*.125,render_constant(n,True))
    print("EXPERIMENT 8: full-output production shader source and render-size sweep")
    for c in (8, 16):
        x,w,_,_=setup(c, c, 32, 32)
        expected=F.conv2d(x,w,padding=1)
        actual,_,_=render_full_conv(x,w)
        delta=(actual-expected).abs()
        print(f"full shader Cin=Cout={c}: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g} rmse={delta.square().mean().sqrt().item():.6g} allclose={torch.allclose(actual,expected,rtol=1e-4,atol=1e-4)}")
    print("render-size sweep: N=1 Cin=Cout=64, kernel=3 stride=1 padding=1 groups=1")
    for size in (32, 64, 128, 160):
        x,w,_,_=setup(64,64,size,size)
        expected=F.conv2d(x,w,padding=1)
        actual,_,_=render_full_conv(x,w)
        delta=(actual-expected).abs()
        tw,th=backend._packed_atlas_size(x.numel())
        print(f"size={size} input_atlas={tw}x{th} output_atlas={tw}x{th} fbo_viewport={tw}x{th}: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g} rmse={delta.square().mean().sqrt().item():.6g} allclose={torch.allclose(actual,expected,rtol=1e-4,atol=1e-4)}")
    print("dispatch controls at first failing size=128")
    x,w,_,_=setup(64,64,128,128); expected=F.conv2d(x,w,padding=1)
    for reset in (False, True):
        actual,_=render_full_dispatch(x,w,reset=reset)
        delta=(actual-expected).abs()
        print(f"one-shot reset={reset}: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g} rmse={delta.square().mean().sqrt().item():.6g} allclose={torch.allclose(actual,expected,rtol=1e-4,atol=1e-4)}")
    for tiles in (2,4,8,16):
        actual,_=render_full_dispatch(x,w,tiles=tiles,reset=True)
        delta=(actual-expected).abs()
        print(f"scissor tiles={tiles}: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g} rmse={delta.square().mean().sqrt().item():.6g} allclose={torch.allclose(actual,expected,rtol=1e-4,atol=1e-4)}")
    actual,owner=render_full_dispatch(x,w,reset=True)
    actual2,_=render_full_dispatch(x,w,reset=True,out_owner=owner)
    delta=(actual2-expected).abs()
    print(f"reused output texture #{owner.texture}: max_abs={delta.max().item():.6g} mean_abs={delta.mean().item():.6g} rmse={delta.square().mean().sqrt().item():.6g} allclose={torch.allclose(actual2,expected,rtol=1e-4,atol=1e-4)}")
    print("EXPERIMENT 8 source form: compute_output() contains literal integer loop bounds for ic/ky/kx, and main computes four output components independently in one fragment.")
    gm45.shutdown()


if __name__=="__main__": main()
