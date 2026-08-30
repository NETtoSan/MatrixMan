#!/usr/bin/env python3
"""Diagnose physical output-target limits without changing MatrixMan."""
from __future__ import annotations

import ctypes
import os
import sys
import math
import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman import gpumatrix as gl
from drivers.matrixman.backends.opengl import convolution, operation_context, resources, runtime, storage
from drivers.matrixman.backends.opengl import tensor as tensor_module

gl.gl.glScissor.restype = None
gl.gl.glScissor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
gl.gl.glDisable.restype = None
gl.gl.glDisable.argtypes = [ctypes.c_uint]
gl.gl.glColorMask.restype = None
gl.gl.glColorMask.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte]


def _params(x, w, out_owner, full_out_w, full_out_h, tile_base=0):
    inp = gm45.to_gm45(x)
    wo = resources.upload_raw_packed_array(w.numpy())
    params = (x.shape[1], x.shape[2], x.shape[3], w.shape[0], full_out_h, x.shape[3],
              3, 3, 1, 1, 1, 1, False, 1, inp._storage_offset,
              inp._owner.layout.texture_width, inp._owner.layout.texture_height,
              wo.layout.texture_width, wo.layout.texture_height, wo.layout.texture_width,
              full_out_w)
    source = convolution._conv_shader_source(params).decode("ascii")
    source = source.replace(
        "int base = (tex_y * " + str(full_out_w) + " + tex_x) * 4;",
        "int base = " + str(tile_base) + " + (tex_y * " + str(full_out_w) + " + tex_x) * 4;",
    )
    return inp, wo, params, source.encode("ascii")


def physical_owner(width, height):
    texture = resources.create_rgba32f_texture(width, height)
    layout = storage.StorageLayout("packed_rgba", width, height, width * height * 4)
    return tensor_module._TextureOwner(texture, layout)


def _draw_conv(x, w, physical_w, physical_h, *, tile_base=0, out_owner=None, reset=False):
    logical_h, logical_w = x.shape[2], x.shape[3]
    if out_owner is None:
        # One tile stores physical_w*physical_h RGBA scalars.
        out_owner = physical_owner(physical_w, physical_h)
    inp, wo, params, source = _params(x, w, out_owner, logical_w * 4, logical_h, tile_base)
    # The source helper expects the packed atlas width, not logical width.
    source = source.decode("ascii").replace(
        "int base = (tex_y * " + str(logical_w * 4) + " + tex_x) * 4;",
        "int base = " + str(tile_base) + " + (tex_y * " + str(logical_w * 4) + " + tex_x) * 4;"
    ).encode("ascii")
    program = gl.make_program(source)
    rt = runtime.runtime_required()
    if reset:
        gl.gl.glDisable(0x0BE2); gl.gl.glDisable(0x0B71); gl.gl.glDisable(0x0C11)
        gl.gl.glDisable(0x0BC0); gl.gl.glDisable(0x0BD0); gl.gl.glColorMask(True, True, True, True)
    gl.glViewport(0, 0, physical_w, physical_h)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, rt.fbo.value)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, out_owner.texture, 0)
    complete = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
    iloc = gl.glGetUniformLocation(program, b"input_tex")
    wloc = gl.glGetUniformLocation(program, b"weight_tex")
    bloc = gl.glGetUniformLocation(program, b"bias_tex")
    gl.glUseProgram(program)
    gl.glActiveTexture(gl.GL_TEXTURE0); gl.glBindTexture(gl.GL_TEXTURE_2D, inp._owner.texture); gl.glUniform1i(iloc, 0)
    gl.glActiveTexture(gl.GL_TEXTURE1); gl.glBindTexture(gl.GL_TEXTURE_2D, wo.texture); gl.glUniform1i(wloc, 1)
    gl.glActiveTexture(gl.GL_TEXTURE2); gl.glBindTexture(gl.GL_TEXTURE_2D, wo.texture); gl.glUniform1i(bloc, 2)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(-1,-1); gl.glVertex2f(1,-1); gl.glVertex2f(1,1); gl.glVertex2f(-1,1); gl.glEnd()
    err = gl.glGetError()
    result = tensor_module.Gm45Tensor._from_owner(out_owner, tuple(out_owner_shape(out_owner))).cpu()
    gl.glDeleteProgram(program)
    return result, out_owner, complete, err, inp, wo


def out_owner_shape(owner):
    # The diagnostic allocates every physical target with this logical shape.
    return (1, 1, owner.layout.texture_height, owner.layout.texture_width * 4)


def copy_tile_rows(stitched, tile, origin_x, origin_y, tile_w, tile_h, full_atlas):
    local = tile.reshape(tile_h, tile_w * 4)
    for row in range(tile_h):
        global_start = ((origin_y + row) * full_atlas + origin_x) * 4
        stitched[global_start:global_start + tile_w * 4] = local[row]


def target_only(size):
    owner = physical_owner(size, size)
    rt = runtime.runtime_required()
    source = f"#version 120\nvoid main(){{ int i=(int(floor(gl_FragCoord.y))*{size}+int(floor(gl_FragCoord.x)))*4; gl_FragColor=vec4(float(i),float(i+1),float(i+2),float(i+3)); }}".encode("ascii")
    program = gl.make_program(source)
    gl.glViewport(0,0,size,size); gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,rt.fbo.value)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,gl.GL_COLOR_ATTACHMENT0,gl.GL_TEXTURE_2D,owner.texture,0)
    complete=gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)==gl.GL_FRAMEBUFFER_COMPLETE
    gl.glUseProgram(program); gl.glBegin(gl.GL_QUADS)
    gl.glVertex2f(-1,-1); gl.glVertex2f(1,-1); gl.glVertex2f(1,1); gl.glVertex2f(-1,1); gl.glEnd()
    err=gl.glGetError(); actual=tensor_module.Gm45Tensor._from_owner(owner,out_owner_shape(owner)).cpu().reshape(-1)
    expected=torch.arange(size*size*4,dtype=torch.float32)
    d=(actual-expected).abs(); gl.glDeleteProgram(program)
    return complete,err,d.max().item(),d.mean().item()


def limits():
    gl.gl.glGetIntegerv.restype=None
    gl.gl.glGetIntegerv.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_int)]
    for name, enum, count in (("GL_MAX_TEXTURE_SIZE",0x0D33,1),("GL_MAX_VIEWPORT_DIMS",0x0D3A,2),
                              ("GL_MAX_RENDERBUFFER_SIZE",0x84E8,1),("GL_MAX_TEXTURE_IMAGE_UNITS",0x8872,1),
                              ("GL_MAX_FRAGMENT_UNIFORM_COMPONENTS",0x8B49,1)):
        values=(ctypes.c_int*count)(); gl.gl.glGetIntegerv(enum,values)
        print(f"{name}={tuple(values)}")


def main():
    gm45.set_trace(False)
    if "--production-tiles" in sys.argv:
        os.environ["MATRIXMAN_DIAGNOSTIC_TILES"] = "1"
        torch.manual_seed(5151)
        x=torch.randn((1,64,160,160),dtype=torch.float32)*0.03
        w=torch.randn((64,64,3,3),dtype=torch.float32)*0.03
        expected=F.conv2d(x,w,padding=1)
        result=F.conv2d(gm45.to_gm45(x),w,padding=1).cpu()
        print(f"production result: shape={list(result.shape)} max_abs={(result-expected).abs().max().item():.6g} allclose={torch.allclose(result,expected,rtol=1e-4,atol=1e-4)}")
        full_atlas=640
        for item in convolution._tile_diagnostic_snapshots:
            local=item["data"].reshape(item["height"],item["width"]*4)
            ref=expected.reshape(full_atlas,full_atlas*4)[item["origin_y"]:item["origin_y"]+item["height"], item["origin_x"]*4:(item["origin_x"]+item["width"])*4]
            d=(local-ref).abs()
            print(f"tile {item['tile_index']}: origin=({item['origin_x']},{item['origin_y']}) physical={item['width']}x{item['height']} texture=#{item['texture']} max_abs={d.max().item():.6g} mean_abs={d.mean().item():.6g} rmse={d.square().mean().sqrt().item():.6g} allclose={torch.allclose(local,ref,rtol=1e-4,atol=1e-4)}")
        print("production tile dispatch: input atlas=640x640, output logical atlas=640x640, viewport=tile dimensions, sampler units=(0 input, 1 weight, 2 bias), tile shader base=((local_y+origin_y)*640+local_x+origin_x)*4")
        gm45.shutdown()
        return
    print("GM45 physical convolution target diagnostic; no backend/demo changes")
    print("TEST B: target-only render")
    limits()
    for size in (256,384,448,480,496,504,508,512):
        complete,err,mx,mean=target_only(size)
        print(f"target={size}x{size} fbo_complete={complete} gl_error=0x{err:04x} max_abs={mx:.6g} mean_abs={mean:.6g} allclose={mx == 0.0}")
    torch.manual_seed(4242)
    x=torch.randn((1,64,128,128),dtype=torch.float32)*0.1
    w=torch.randn((64,64,3,3),dtype=torch.float32)*0.1
    expected=F.conv2d(x,w,padding=1)
    print("TEST A: 512x512 input atlas with physically small output target")
    for size in (128,256):
        owner=operation_context.output_texture((1,1,size,size*4))
        actual,_,complete,err,inp,wo=_draw_conv(x,w,size,size,out_owner=owner,reset=True)
        gathered=actual.reshape(-1)
        expected_tile=expected.reshape(-1).reshape(512, 512*4)[:size,:size*4].reshape(-1)
        d=(gathered-expected_tile).abs()
        print(f"input_atlas={inp._owner.layout.texture_width}x{inp._owner.layout.texture_height} output_target={size}x{size} fbo_complete={complete} gl_error=0x{err:04x} max_abs={d.max().item():.6g} mean_abs={d.mean().item():.6g} allclose={torch.allclose(gathered,expected_tile,rtol=1e-4,atol=1e-4)}")
    print("TEST C: four independent 256x256 physical targets reconstruct logical 512x512 atlas")
    stitched=torch.empty(expected.numel(),dtype=torch.float32)
    full_atlas=512
    for ty in (0,1):
        for tx in (0,1):
            base=(ty*256*full_atlas+tx*256)*4
            actual,_,complete,err,inp,wo=_draw_conv(x,w,256,256,tile_base=base,reset=True)
            copy_tile_rows(stitched, actual, tx * 256, ty * 256, 256, 256, full_atlas)
            print(f"tile=({tx},{ty}) base={base} fbo_complete={complete} gl_error=0x{err:04x}")
    d=(stitched-expected.reshape(-1)).abs()
    print(f"tiled reconstructed: max_abs={d.max().item():.6g} mean_abs={d.mean().item():.6g} rmse={d.square().mean().sqrt().item():.6g} allclose={torch.allclose(stitched,expected.reshape(-1),rtol=1e-4,atol=1e-4)}")
    print("TEST C2: 640x640 logical atlas reconstructed from <=256x256 physical targets")
    torch.manual_seed(4343)
    x640=torch.randn((1,64,160,160),dtype=torch.float32)*0.03
    w640=torch.randn((64,64,3,3),dtype=torch.float32)*0.03
    expected640=F.conv2d(x640,w640,padding=1)
    stitched640=torch.empty(expected640.numel(),dtype=torch.float32)
    full640=640
    for oy in (0,256,512):
        for ox in (0,256,512):
            tw=min(256,full640-ox); th=min(256,full640-oy)
            base=(oy*full640+ox)*4
            actual,_,complete,err,_,_=_draw_conv(x640,w640,tw,th,tile_base=base,reset=True)
            copy_tile_rows(stitched640,actual,ox,oy,tw,th,full640)
            print(f"tile origin=({ox},{oy}) physical={tw}x{th} fbo_complete={complete} gl_error=0x{err:04x}")
    d=(stitched640-expected640.reshape(-1)).abs()
    print(f"640 tiled reconstructed: max_abs={d.max().item():.6g} mean_abs={d.mean().item():.6g} rmse={d.square().mean().sqrt().item():.6g} allclose={torch.allclose(stitched640,expected640.reshape(-1),rtol=1e-4,atol=1e-4)}")
    gm45.shutdown()


if __name__=="__main__": main()
