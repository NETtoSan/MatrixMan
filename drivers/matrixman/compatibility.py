"""Capability and numerical compatibility probe for MatrixMan.

This module deliberately uses the existing public MatrixMan operations for
the self-tests. It does not replace them with CPU calculations; CPU tensors
are used only as references and for explicit readback verification.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import gm45_backend as backend
from . import gpumatrix as gm


GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION = 0x1F02
GL_EXTENSIONS = 0x1F03
GL_SHADING_LANGUAGE_VERSION = 0x8B8C
GL_MAX_TEXTURE_SIZE = 0x0D33
GL_MAX_VIEWPORT_DIMS = 0x0D3A
GL_MAX_RENDERBUFFER_SIZE = 0x84E8
GL_MAX_TEXTURE_IMAGE_UNITS = 0x8872
GL_MAX_FRAGMENT_UNIFORM_COMPONENTS = 0x8B49

_gl_get_integerv = gm.gl_proc(
    "glGetIntegerv", None, ctypes.c_uint, ctypes.POINTER(ctypes.c_int)
)

SELFTEST_RTOL = 1e-4
SELFTEST_ATOL = 1e-4


@dataclass
class Limits:
    texture_size: int
    viewport: tuple[int, int]
    renderbuffer_size: int
    texture_units: int
    fragment_uniforms: int


def _string(enum: int) -> str:
    value = gm.glGetString(enum)
    return value.decode("utf-8", "replace") if value else "unavailable"


def _integer(enum: int, count: int = 1):
    values = (ctypes.c_int * count)()
    _gl_get_integerv(enum, values)
    return tuple(int(values[i]) for i in range(count))


def _version_tuple(version: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _limits() -> Limits:
    return Limits(
        texture_size=_integer(GL_MAX_TEXTURE_SIZE)[0],
        viewport=_integer(GL_MAX_VIEWPORT_DIMS, 2),
        renderbuffer_size=_integer(GL_MAX_RENDERBUFFER_SIZE)[0],
        texture_units=_integer(GL_MAX_TEXTURE_IMAGE_UNITS)[0],
        fragment_uniforms=_integer(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS)[0],
    )


def _extensions() -> set[str]:
    return set(_string(GL_EXTENSIONS).split())


def _check(name: str, fn) -> bool:
    try:
        ok = bool(fn())
    except Exception as exc:  # A probe should report failures, not abort.
        print(f"{name:<30} FAIL ({exc})")
        return False
    print(f"{name:<30} {'PASS' if ok else 'FAIL'}")
    return ok


def _same(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return bool(torch.allclose(actual, expected, rtol=SELFTEST_RTOL, atol=SELFTEST_ATOL))


def _report_metrics(label: str, actual: torch.Tensor, expected: torch.Tensor) -> bool:
    error = (actual - expected).to(torch.float64)
    max_abs = float(error.abs().max()) if error.numel() else 0.0
    mean_abs = float(error.abs().mean()) if error.numel() else 0.0
    rmse = float(error.square().mean().sqrt()) if error.numel() else 0.0
    passed = _same(actual, expected)
    print(
        f"{label}: {'PASS' if passed else 'FAIL'}; "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rmse={rmse:.6g} "
        f"tolerance=rtol:{SELFTEST_RTOL:g},atol:{SELFTEST_ATOL:g}"
    )
    return passed


def _float_upload_readback() -> bool:
    source = torch.tensor([[0.25, -1.5], [2.0, 7.25]], dtype=torch.float32)
    return _report_metrics("Float texture", backend.to_gm45(source).cpu(), source)


def _elementwise_shader() -> bool:
    left = torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.float32)
    right = torch.tensor([[2.0, 5.0], [-1.5, 0.5]], dtype=torch.float32)
    actual = torch.add(backend.to_gm45(left), backend.to_gm45(right)).cpu()
    return _report_metrics("Elementwise add", actual, left + right)


def _conv() -> bool:
    torch.manual_seed(451)
    source = torch.randn((1, 2, 8, 8), dtype=torch.float32)
    weight = torch.randn((3, 2, 3, 3), dtype=torch.float32) * 0.1
    expected = F.conv2d(source, weight, padding=1)
    actual = F.conv2d(backend.to_gm45(source), weight, padding=1).cpu()
    return _report_metrics("Conv2D", actual, expected)


def _batch_norm() -> bool:
    torch.manual_seed(452)
    source = torch.randn((1, 4, 4, 4), dtype=torch.float32)
    mean = torch.randn(4, dtype=torch.float32)
    var = torch.rand(4, dtype=torch.float32) + 0.5
    weight = torch.randn(4, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)
    expected = F.batch_norm(source, mean, var, weight, bias, training=False, eps=1e-5)
    actual = F.batch_norm(
        backend.to_gm45(source), mean, var, weight, bias, training=False, eps=1e-5
    ).cpu()
    return _report_metrics("BatchNorm", actual, expected)


def _silu() -> bool:
    source = torch.linspace(-3.0, 3.0, 32, dtype=torch.float32).reshape(1, 2, 4, 4)
    expected = F.silu(source)
    actual = torch.nn.SiLU(inplace=True)(backend.to_gm45(source)).cpu()
    return _report_metrics("SiLU_", actual, expected)


def _large_conv(size: int) -> bool:
    torch.manual_seed(453 + size)
    source = torch.randn((1, 64, size, size), dtype=torch.float32) * 0.05
    weight = torch.randn((64, 64, 3, 3), dtype=torch.float32) * 0.05
    expected = F.conv2d(source, weight, padding=1)
    actual = F.conv2d(backend.to_gm45(source), weight, padding=1).cpu()
    atlas = size * 4
    return _report_metrics(f"Conv atlas {atlas}x{atlas}", actual, expected)


def _isolated_one_shot() -> bool:
    """Probe the known-unsafe large render in a disposable GL context."""
    configured_limit = backend.CONV_PHYSICAL_TILE_LIMIT
    backend.CONV_PHYSICAL_TILE_LIMIT = 1 << 30
    try:
        return _large_conv(128)
    finally:
        backend.CONV_PHYSICAL_TILE_LIMIT = configured_limit


def _run_isolated_one_shot() -> bool:
    command = [sys.executable, "-m", "drivers.matrixman.compatibility", "--internal-large-one-shot"]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("  512x512 one-shot isolated: FAIL (timed out; treated as unsafe)")
        return False
    output = completed.stdout.strip()
    if output:
        for line in output.splitlines():
            print(f"    {line}")
    passed = completed.returncode == 0
    if completed.returncode < 0:
        print(f"  512x512 one-shot isolated: FAIL (child terminated by signal {-completed.returncode})")
    elif not passed:
        print(f"  512x512 one-shot isolated: FAIL (child exit {completed.returncode}; treated as unsafe)")
    else:
        print("  512x512 one-shot isolated: PASS")
    return passed


def report() -> int:
    renderer = _string(GL_RENDERER)
    vendor = _string(GL_VENDOR)
    version = _string(GL_VERSION)
    glsl = _string(GL_SHADING_LANGUAGE_VERSION)
    limits = _limits()
    extensions = _extensions()
    gl_version = _version_tuple(version)
    glsl_version = _version_tuple(glsl)

    print("MatrixMan Compatibility Report")
    print(f"\nRenderer: {renderer}")
    print(f"Vendor: {vendor}")
    print(f"OpenGL version: {version}")
    print(f"GLSL version: {glsl}")
    print("\nRequired capabilities:")
    programmable = gl_version >= (2, 0)
    fbo_extension = {"GL_EXT_framebuffer_object", "GL_ARB_framebuffer_object"} & extensions
    float_extension = {
        "GL_ARB_texture_float",
        "GL_ATI_texture_float",
        "GL_EXT_color_buffer_float",
        "GL_ARB_color_buffer_float",
    } & extensions
    print(f"  programmable fragment shaders: {'PASS' if programmable else 'FAIL'}")
    fbo_capability = bool(fbo_extension or gl_version >= (3, 0))
    float_capability = bool(float_extension or gl_version >= (3, 0))
    texture_capability = (
        limits.texture_size >= 256
        and limits.viewport[0] >= 256
        and limits.viewport[1] >= 256
        and limits.renderbuffer_size >= 256
    )
    print(f"  framebuffer objects: {'PASS' if fbo_capability else 'FAIL'}")
    print(f"  floating-point textures: {'PASS' if float_capability else 'FAIL'}")
    print(f"  floating-point render targets: {'CHECK' if not float_capability else 'verified by self-test'}")
    print(f"  sufficient texture units: {'PASS' if limits.texture_units >= 3 else 'FAIL'}")
    print(f"  texture size: {'PASS' if texture_capability else 'FAIL'} ({limits.texture_size})")
    print(f"  viewport size: {'PASS' if texture_capability else 'FAIL'} ({limits.viewport[0]}x{limits.viewport[1]})")
    print("\nDetected limits:")
    print(f"  GL_MAX_TEXTURE_SIZE = {limits.texture_size}")
    print(f"  GL_MAX_VIEWPORT_DIMS = {limits.viewport[0]}x{limits.viewport[1]}")
    print(f"  GL_MAX_RENDERBUFFER_SIZE = {limits.renderbuffer_size}")
    print(f"  GL_MAX_TEXTURE_IMAGE_UNITS = {limits.texture_units}")
    print(f"  GL_MAX_FRAGMENT_UNIFORM_COMPONENTS = {limits.fragment_uniforms}")
    print("\nRelevant extensions:")
    for name in sorted(
        {
            "GL_EXT_framebuffer_object",
            "GL_ARB_framebuffer_object",
            "GL_ARB_texture_float",
            "GL_ATI_texture_float",
            "GL_EXT_color_buffer_float",
            "GL_ARB_color_buffer_float",
        }
    ):
        print(f"  {name}: {'yes' if name in extensions else 'no'}")

    print("\nNumerical self-tests:")
    tests = [
        ("Float texture upload/readback", _float_upload_readback),
        ("Float FBO", _elementwise_shader),
        ("Elementwise shader", _elementwise_shader),
        ("Conv2D", _conv),
        ("BatchNorm", _batch_norm),
        ("SiLU", _silu),
    ]
    results = [_check(name, fn) for name, fn in tests]

    print("\nLarge Conv Render:")
    print(f"  configured safe tile size: {backend.CONV_PHYSICAL_TILE_LIMIT}x{backend.CONV_PHYSICAL_TILE_LIMIT}")
    # Validate the normal production path before probing the known-unsafe
    # one-shot render. Each call allocates fresh input/output resources.
    large_256 = _check("  256x256 logical atlas", lambda: _large_conv(64))
    large_512_tiled = _check("  512x512 production tiled", lambda: _large_conv(128))
    one_shot = _run_isolated_one_shot()
    if one_shot:
        print("  Large Conv Render: PASS")
        print("  convolution tiling required: no")
    else:
        print(f"  Large Conv Render: {'PASS' if large_512_tiled else 'FAIL'}")
        print("  convolution tiling required: yes")

    # One-shot is a quirk-discovery test, not a required capability. A failed
    # isolated one-shot is acceptable when the normal tiled path passes.
    passed = all(results) and large_256 and large_512_tiled
    if (
        programmable
        and glsl_version >= (1, 20)
        and fbo_capability
        and float_capability
        and texture_capability
        and limits.texture_units >= 3
        and passed
    ):
        status = "COMPATIBLE"
    elif programmable and passed:
        status = "PARTIALLY COMPATIBLE"
    else:
        status = "UNSUPPORTED"
    print(f"\nMatrixMan status: {status}")
    return 0 if status != "UNSUPPORTED" else 1


def main() -> int:
    if "--internal-large-one-shot" in sys.argv:
        backend.set_trace(False)
        try:
            return 0 if _isolated_one_shot() else 2
        finally:
            backend.shutdown()
    if "--test-large-tiled" in sys.argv:
        backend.set_trace(False)
        try:
            print("MatrixMan fresh-context tiled convolution test")
            passed = _check("512x512 production tiled", lambda: _large_conv(128))
            print(f"Result: {'PASS' if passed else 'FAIL'}")
            return 0 if passed else 1
        finally:
            backend.shutdown()
    backend.set_trace(False)
    try:
        return report()
    finally:
        backend.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
