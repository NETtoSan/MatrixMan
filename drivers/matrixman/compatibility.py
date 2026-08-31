"""Capability and numerical compatibility probe for MatrixMan.

This module deliberately uses the existing public MatrixMan operations for
the self-tests. It does not replace them with CPU calculations; CPU tensors
are used only as references and for explicit readback verification.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import gm45_backend as backend
from . import gpumatrix as gm
from .backends.opengl import convolution, profiling, resources, runtime as runtime_module, tensor
from .backends.opengl.storage import StorageLayout, pack_linear_rgba, packed_atlas_size


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
    return _report_metrics("Float texture", backend.to_device(source).cpu(), source)


def _elementwise_shader() -> bool:
    left = torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.float32)
    right = torch.tensor([[2.0, 5.0], [-1.5, 0.5]], dtype=torch.float32)
    actual = torch.add(backend.to_device(left), backend.to_device(right)).cpu()
    return _report_metrics("Elementwise add", actual, left + right)


def _conv() -> bool:
    torch.manual_seed(451)
    source = torch.randn((1, 2, 8, 8), dtype=torch.float32)
    weight = torch.randn((3, 2, 3, 3), dtype=torch.float32) * 0.1
    expected = F.conv2d(source, weight, padding=1)
    actual = F.conv2d(backend.to_device(source), weight, padding=1).cpu()
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
        backend.to_device(source), mean, var, weight, bias, training=False, eps=1e-5
    ).cpu()
    return _report_metrics("BatchNorm", actual, expected)


def _silu() -> bool:
    source = torch.linspace(-3.0, 3.0, 32, dtype=torch.float32).reshape(1, 2, 4, 4)
    expected = F.silu(source)
    actual = torch.nn.SiLU(inplace=True)(backend.to_device(source)).cpu()
    return _report_metrics("SiLU_", actual, expected)


def _report_physical_tiles(expected: torch.Tensor) -> None:
    packed, _ = pack_linear_rgba(expected.numpy())
    print(f"physical tiles (consolidation destination texture=#{convolution._last_tile_output_texture}):")
    all_passed = True
    for snapshot in convolution._tile_diagnostic_snapshots:
        width, height = snapshot["width"], snapshot["height"]
        origin_x, origin_y = snapshot["origin_x"], snapshot["origin_y"]
        actual = snapshot["data"].reshape(height, width, 4).numpy()
        reference = packed[origin_y:origin_y + height, origin_x:origin_x + width]
        error = torch.from_numpy(actual - reference).to(torch.float64)
        max_abs = float(error.abs().max()) if error.numel() else 0.0
        mean_abs = float(error.abs().mean()) if error.numel() else 0.0
        rmse = float(error.square().mean().sqrt()) if error.numel() else 0.0
        passed = bool(torch.allclose(torch.from_numpy(actual), torch.from_numpy(reference), rtol=SELFTEST_RTOL, atol=SELFTEST_ATOL))
        all_passed = all_passed and passed
        print(
            f"  tile {snapshot['tile_index']}: grid={snapshot['grid']} "
            f"origin=({origin_x},{origin_y}) "
            f"size={width}x{height} texture=#{snapshot['texture']} "
            f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rmse={rmse:.6g} "
            f"result={'PASS' if passed else 'FAIL'}"
        )
    print(f"  physical tiles overall: {'PASS' if all_passed else 'FAIL'}")


def _large_conv(
    size: int,
    *,
    cin: int = 64,
    cout: int = 64,
    kernel: int = 3,
    report_tiles: bool = False,
) -> bool:
    torch.manual_seed(453 + size + cin + cout + kernel)
    source = torch.randn((1, cin, size, size), dtype=torch.float32) * 0.05
    weight = torch.randn((cout, cin, kernel, kernel), dtype=torch.float32) * 0.05
    padding = 1 if kernel == 3 else 0
    expected = F.conv2d(source, weight, padding=padding)
    actual = F.conv2d(backend.to_device(source), weight, padding=padding).cpu()
    if report_tiles:
        _report_physical_tiles(expected)
    atlas = packed_atlas_size(expected.numel())[0]
    atlas_height = packed_atlas_size(expected.numel())[1]
    label = f"Consolidated conv atlas {atlas}x{atlas_height}" if report_tiles else f"Conv atlas {atlas}x{atlas_height}"
    return _report_metrics(label, actual, expected)


def _diagnostic_workload(name: str) -> dict[str, int]:
    workloads = {
        "heavy": {"cin": 64, "cout": 64, "kernel": 3, "spatial": 128},
        "medium": {"cin": 32, "cout": 32, "kernel": 3, "spatial": 181},
        "light": {"cin": 16, "cout": 16, "kernel": 3, "spatial": 256},
        "one_by_one": {"cin": 64, "cout": 64, "kernel": 1, "spatial": 128},
    }
    try:
        return workloads[name]
    except KeyError as exc:
        raise RuntimeError(
            "MATRIXMAN_DIAG_CONV_WORKLOAD must be one of: "
            "heavy, medium, light, one_by_one"
        ) from exc


def _isolated_one_shot() -> bool:
    """Probe the known-unsafe large render in a disposable GL context."""
    configured_limit = convolution.CONV_PHYSICAL_TILE_LIMIT
    convolution.CONV_PHYSICAL_TILE_LIMIT = 1 << 30
    try:
        return _large_conv(128)
    finally:
        convolution.CONV_PHYSICAL_TILE_LIMIT = configured_limit


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


def _print_tile_geometry(width_limit: int, height_limit: int, atlas_width: int, atlas_height: int) -> None:
    tiles = convolution._last_tile_geometry
    print(f"logical output atlas dimensions: {atlas_width}x{atlas_height}")
    print(
        f"diagnostic width/height limit: {width_limit}x{height_limit}\n"
        f"physical tile grid: {((atlas_width + width_limit - 1) // width_limit)}x"
        f"{((atlas_height + height_limit - 1) // height_limit)}"
    )
    print(f"tile count: {len(tiles)}")
    if not tiles:
        print("execution path: one-shot Conv2D (no physical tiled render)")
        print("consolidation occurred: no")
        return
    print("execution path: physical tiled Conv2D")
    print("consolidation occurred: yes")
    areas = []
    for index, tile in enumerate(tiles):
        width, height = tile["width"], tile["height"]
        areas.append(width * height)
        print(
            f"render #{tile['render_sequence_index']}: grid={tile['grid']} "
            f"origin={tile['origin']} "
            f"physical_size={width}x{height} "
            f"logical/output region={tile['logical_region']} "
            f"texture=#{tile.get('texture', 'unknown')} size={tile['texture_size']}"
        )


def _consolidation_pattern(atlas: int) -> np.ndarray:
    """Create a deterministic packed RGBA atlas with coordinate-coded values."""
    y, x, component = np.indices((atlas, atlas, 4), dtype=np.float32)
    return y * 0.001 + x * 0.000001 + component * 0.1


def _consolidation_metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float, bool]:
    error = torch.from_numpy(actual - expected).to(torch.float64)
    max_abs = float(error.abs().max()) if error.numel() else 0.0
    mean_abs = float(error.abs().mean()) if error.numel() else 0.0
    rmse = float(error.square().mean().sqrt()) if error.numel() else 0.0
    passed = bool(torch.allclose(torch.from_numpy(actual), torch.from_numpy(expected), rtol=1e-6, atol=1e-6))
    return max_abs, mean_abs, rmse, passed


def _test_consolidation() -> bool:
    atlas = 512
    width_limit = int(os.environ.get("MATRIXMAN_DIAG_TILE_WIDTH", 256))
    height_limit = int(os.environ.get("MATRIXMAN_DIAG_TILE_HEIGHT", 256))
    if width_limit <= 0 or height_limit <= 0:
        raise RuntimeError("diagnostic tile dimensions must be positive")
    tiles_x = (atlas + width_limit - 1) // width_limit
    tiles_y = (atlas + height_limit - 1) // height_limit
    pattern = _consolidation_pattern(atlas)
    tile_owners = []
    destination = None
    geometries = []
    print("MatrixMan standalone consolidation test")
    print(f"atlas: {atlas}x{atlas}")
    print(f"tile limits: {width_limit}x{height_limit}")
    print(f"tile grid: {tiles_x}x{tiles_y} ({tiles_x * tiles_y} tiles)")
    try:
        for tile_y in range(tiles_y):
            origin_y = tile_y * height_limit
            tile_h = min(height_limit, atlas - origin_y)
            for tile_x in range(tiles_x):
                origin_x = tile_x * width_limit
                tile_w = min(width_limit, atlas - origin_x)
                tile_data = np.ascontiguousarray(pattern[origin_y:origin_y + tile_h, origin_x:origin_x + tile_w])
                texture = resources.create_rgba32f_texture(tile_w, tile_h, tile_data)
                owner = tensor._TextureOwner(
                    texture, StorageLayout("packed_rgba", tile_w, tile_h, tile_w * tile_h * 4)
                )
                tile_owners.append(owner)
                geometries.append({
                    "grid": (tile_x, tile_y),
                    "origin": (origin_x, origin_y),
                    "width": tile_w,
                    "height": tile_h,
                    "texture": texture,
                })
        for geometry, owner in zip(geometries, tile_owners):
            uploaded = tensor.readback_tensor(
                owner, (1, 1, geometry["height"], geometry["width"] * 4)
            ).reshape(geometry["height"], geometry["width"], 4).numpy()
            expected = pattern[
                geometry["origin"][1]:geometry["origin"][1] + geometry["height"],
                geometry["origin"][0]:geometry["origin"][0] + geometry["width"],
            ]
            _, _, _, passed = _consolidation_metrics(uploaded, expected)
            print(f"source tile grid={geometry['grid']} texture=#{owner.texture} upload={'PASS' if passed else 'FAIL'}")
            if not passed:
                return False

        destination_texture = resources.create_rgba32f_texture(atlas, atlas)
        destination = tensor._TextureOwner(
            destination_texture,
            StorageLayout("packed_rgba", atlas, atlas, atlas * atlas * 4),
        )
        runtime = runtime_module.runtime_required()
        gm.glFinish()
        convolution._consolidate_tiles(
            tile_owners, geometries, destination, atlas, atlas,
            width_limit, height_limit, runtime,
        )
        gm.glFinish()
        actual = tensor.readback_tensor(
            destination, (1, 1, atlas, atlas * 4)
        ).reshape(atlas, atlas, 4).numpy()
        max_abs, mean_abs, rmse, passed = _consolidation_metrics(actual, pattern)
        print(f"destination texture=#{destination.texture}")
        print(f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rmse={rmse:.6g}")
        print(f"consolidation: {'PASS' if passed else 'FAIL'}")
        return passed
    finally:
        for owner in tile_owners:
            texture = ctypes.c_uint(owner.texture)
            gm.glDeleteTextures(1, ctypes.byref(texture))
        if destination is not None:
            texture = ctypes.c_uint(destination.texture)
            gm.glDeleteTextures(1, ctypes.byref(texture))
    widths = [tile["width"] for tile in tiles]
    heights = [tile["height"] for tile in tiles]
    remainder_width = atlas % width_limit or width_limit
    remainder_height = atlas % height_limit or height_limit
    print(f"maximum tile area: {max(areas)}")
    print(f"minimum tile area: {min(areas)}")
    print(f"remainder width: {remainder_width}")
    print(f"remainder height: {remainder_height}")
    for divisor in (2, 4, 8, 16, 32, 64):
        print(
            f"divisible by {divisor}: "
            f"widths={'yes' if all(width % divisor == 0 for width in widths) else 'no'} "
            f"heights={'yes' if all(height % divisor == 0 for height in heights) else 'no'}"
        )


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
    print(f"  configured safe tile size: {convolution.CONV_PHYSICAL_TILE_LIMIT}x{convolution.CONV_PHYSICAL_TILE_LIMIT}")
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
            os.environ["MATRIXMAN_DIAGNOSTIC_TILES"] = "1"
            os.environ["MATRIXMAN_DIAGNOSTIC_RECT_TILES"] = "1"
            print("MatrixMan fresh-context tiled convolution test")
            limit_text = os.environ.get("MATRIXMAN_TILE_LIMIT", "256")
            try:
                limit = int(limit_text)
            except ValueError:
                print(f"Invalid MATRIXMAN_TILE_LIMIT: {limit_text!r}")
                return 1
            if limit <= 0:
                print("Invalid MATRIXMAN_TILE_LIMIT: must be positive")
                return 1
            try:
                diag_width = int(os.environ.get("MATRIXMAN_DIAG_TILE_WIDTH", limit))
                diag_height = int(os.environ.get("MATRIXMAN_DIAG_TILE_HEIGHT", limit))
            except ValueError:
                print("Invalid diagnostic tile dimensions: must be positive integers")
                return 1
            if diag_width <= 0 or diag_height <= 0:
                print("Invalid diagnostic tile dimensions: must be positive")
                return 1
            workload_name = os.environ.get("MATRIXMAN_DIAG_CONV_WORKLOAD", "heavy").strip().lower() or "heavy"
            try:
                workload = _diagnostic_workload(workload_name)
            except RuntimeError as exc:
                print(str(exc))
                return 1
            macs = workload["cin"] * workload["kernel"] * workload["kernel"]
            logical_numel = workload["cout"] * workload["spatial"] * workload["spatial"]
            atlas_width, atlas_height = packed_atlas_size(logical_numel)
            tiles_x = (atlas_width + diag_width - 1) // diag_width
            tiles_y = (atlas_height + diag_height - 1) // diag_height
            print(f"MATRIXMAN_TILE_LIMIT: {limit}")
            print(f"MATRIXMAN_TILE_SYNC: {os.environ.get('MATRIXMAN_TILE_SYNC', 'per_tile')}")
            print(f"MATRIXMAN_DIAG_TILE_WIDTH/HEIGHT: {diag_width}x{diag_height}")
            print(f"MATRIXMAN_DIAG_TILE_ORDER: {os.environ.get('MATRIXMAN_DIAG_TILE_ORDER', 'normal')}")
            print(f"MATRIXMAN_DIAG_CONV_WORKLOAD: {workload_name}")
            print(
                f"Conv2D: input=[1,{workload['cin']},{workload['spatial']},{workload['spatial']}] "
                f"weight=[{workload['cout']},{workload['cin']},{workload['kernel']},{workload['kernel']}] "
                f"kernel={workload['kernel']}x{workload['kernel']} groups=1 "
                f"stride=1 padding={1 if workload['kernel'] == 3 else 0} "
                f"MACs/output={macs} texture_samples/output={macs}"
            )
            print(f"logical output atlas dimensions: {atlas_width}x{atlas_height}")
            print(f"configured physical tile grid: {tiles_x}x{tiles_y} ({tiles_x * tiles_y} tiles)")
            started = time.perf_counter()
            passed = _check(
                "Conv2D diagnostic",
                lambda: _large_conv(
                    workload["spatial"],
                    cin=workload["cin"],
                    cout=workload["cout"],
                    kernel=workload["kernel"],
                    report_tiles=True,
                ),
            )
            print(f"elapsed: {time.perf_counter() - started:.3f}s")
            _print_tile_geometry(diag_width, diag_height, atlas_width, atlas_height)
            if backend.profile_enabled():
                counters = profiling.counters
                print(f"glFinish count: {int(counters['glFinish_calls'])}")
                print(f"glFinish time: {counters['glFinish_seconds']:.3f}s")
                print(f"pre-consolidation glFinish executed: {int(counters['pre_consolidation_sync_calls'])}")
                print(f"pre-consolidation glFinish skipped: {int(counters['pre_consolidation_sync_skips'])}")
                print(f"glFlush count: {int(counters['glFlush_calls'])}")
                print(f"glFlush time: {counters['glFlush_seconds']:.3f}s")
            else:
                print("glFinish counters: unavailable (set MATRIXMAN_PROFILE=1)")
            print(f"Result: {'PASS' if passed else 'FAIL'}")
            return 0 if passed else 1
        finally:
            backend.shutdown()
    if "--test-consolidation" in sys.argv:
        backend.set_trace(False)
        try:
            return 0 if _test_consolidation() else 1
        finally:
            backend.shutdown()
    backend.set_trace(False)
    try:
        return report()
    finally:
        backend.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
