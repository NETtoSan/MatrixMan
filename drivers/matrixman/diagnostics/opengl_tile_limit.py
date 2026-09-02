"""Validate MatrixMan OpenGL Conv physical tile sizes on the current GPU.

Each candidate is run in a fresh child process because an allocation or driver
reset at an aggressive size must not take down the parent sweep.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F


DEFAULT_SIZES = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
DEFAULT_TIMEOUT = 30.0
MAX_ESTIMATED_BYTES = 768 * 1024 * 1024
AUTOTUNE_SCHEMA_VERSION = 1
_GL_LIMITS = {
    "GL_MAX_TEXTURE_SIZE": (0x0D33, 1),
    "GL_MAX_RENDERBUFFER_SIZE": (0x84E8, 1),
    "GL_MAX_VIEWPORT_DIMS": (0x0D3A, 2),
    "GL_MAX_TEXTURE_IMAGE_UNITS": (0x8872, 1),
    "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS": (0x8B4D, 1),
    "GL_MAX_FRAGMENT_UNIFORM_COMPONENTS": (0x8B49, 1),
}


def _text(value) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _limits(gl) -> dict[str, int | tuple[int, int]]:
    gl.gl.glGetIntegerv.restype = None
    gl.gl.glGetIntegerv.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
    result = {}
    for name, (enum, count) in _GL_LIMITS.items():
        values = (ctypes.c_int * count)()
        gl.gl.glGetIntegerv(enum, values)
        result[name] = tuple(values) if count > 1 else int(values[0])
    return result


def _memory_estimate(size: int) -> tuple[int, int]:
    one = size * size * 16
    # Input, output, tile/consolidation scratch, and a conservative attachment
    # allowance. These are estimates, not promises about driver allocations.
    practical = one * 4
    return one, practical


def _stats(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    delta = (actual - reference).to(torch.float64)
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt((delta * delta).mean())),
    }


def _classify_exception(exc: BaseException) -> str:
    message = str(exc).lower()
    if "framebuffer incomplete" in message:
        return "FAIL framebuffer incomplete"
    if "opengl error" in message or "gl error" in message:
        return "FAIL GL error"
    if any(word in message for word in ("alloc", "out of memory", "outofmemory")):
        return "FAIL allocation"
    return "FAIL"


def _child(size: int, spatial: int | None = None) -> int:
    os.environ["MATRIXMAN_BACKEND"] = "opengl"
    os.environ["MATRIXMAN_TILE_LIMIT"] = str(size)
    os.environ["MATRIXMAN_TILE_SYNC"] = "per_tile"
    os.environ["MATRIXMAN_DIAGNOSTIC_TILES"] = "1"
    # Keep the primary validation about ordinary Conv dispatch, independent
    # of the caller's opt-in experimental spatial-reuse setting.
    os.environ["MATRIXMAN_CONV_SPATIAL_REUSE"] = "0"
    try:
        from drivers import matrixman
        from drivers.matrixman.backend import get_backend
        from drivers.matrixman.backends.opengl import convolution

        matrixman.config.reloadFromEnvironment()
        matrixman.prefer("opengl")
        selected = get_backend()
        if selected.name != "opengl":
            raise RuntimeError(f"selected backend is {selected.name}")
        spatial = spatial or size // 4
        torch.manual_seed(7000 + size + spatial)
        x = torch.randn((1, 64, spatial, spatial), dtype=torch.float32) * 0.03
        weight = torch.randn((64, 64, 3, 3), dtype=torch.float32) * 0.03
        reference = F.conv2d(x, weight, padding=1)
        started = time.perf_counter()
        actual = F.conv2d(matrixman.to_device(x), weight, padding=1).cpu()
        runtime = time.perf_counter() - started
        telemetry = convolution.diagnostic_tile_geometry()
        stats = _stats(reference, actual)
        tolerance = 1e-4
        passed = torch.allclose(actual, reference, rtol=tolerance, atol=tolerance)
        tiles = telemetry.get("tiles", [])
        largest = max(((int(t["width"]), int(t["height"])) for t in tiles), default=(0, 0))
        result = {
            "size": size, "input_shape": list(x.shape), "output_shape": list(actual.shape),
            "atlas": list(telemetry.get("atlas", (0, 0))),
            "path": "direct" if not telemetry.get("tiled") else "tiled",
            "physical_tiles": int(telemetry.get("physical_tile_count", 0)),
            "largest_physical_tile": list(largest), "runtime": runtime,
            "gl_error": "0x0000", "stats": stats, "tolerance": tolerance,
            "result": "PASS" if passed else "FAIL numerical mismatch",
        }
        print(json.dumps(result), flush=True)
        matrixman.shutdown()
        return 0 if passed else 2
    except BaseException as exc:
        print(json.dumps({"size": size, "result": _classify_exception(exc), "error": str(exc)}), flush=True)
        try:
            from drivers import matrixman
            matrixman.shutdown()
        except BaseException:
            pass
        return 3


def _run_child(size: int, timeout: float, spatial: int | None = None) -> dict:
    command = [sys.executable, "-m", "drivers.matrixman.diagnostics.opengl_tile_limit", "--child", "--size", str(size)]
    if spatial is not None:
        command += ["--spatial", str(spatial)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"size": size, "result": "FAIL timeout"}
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "result" in value:
            if completed.returncode < 0:
                value["result"] = "FAIL child process crash"
            return value
    return {"size": size, "result": "FAIL child process crash", "error": completed.stderr[-500:]}


def _cache_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "matrixman" / "opengl_tile_autotune.json"


def _cache_key(info: dict[str, str]) -> str:
    fields = ("vendor", "renderer", "opengl", "glsl")
    return json.dumps({field: str(info.get(field, "unavailable")) for field in fields}, sort_keys=True)


def autotune_tile_limit(info: dict[str, str], *, refresh: bool = False, timeout: float = DEFAULT_TIMEOUT) -> int:
    """Return the largest passing size, using the same isolated Conv tests as the CLI."""
    path = _cache_path()
    key = _cache_key(info)
    if not refresh:
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            entry = cache.get("entries", {}).get(key, {})
            if entry.get("schema") == AUTOTUNE_SCHEMA_VERSION and int(entry.get("tile_limit", 0)) > 0:
                return int(entry["tile_limit"])
        except (OSError, ValueError, TypeError):
            pass

    command = [
        sys.executable, "-m", "drivers.matrixman.diagnostics.opengl_tile_limit",
        "--autotune-worker", "--timeout", str(timeout),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True,
            timeout=max(60.0, timeout * len(DEFAULT_SIZES) + 20.0), check=False,
        )
        worker = next((json.loads(line) for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), None)
        passed = [int(value) for value in (worker or {}).get("passed", [])]
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        passed = []
    resolved = max(passed) if passed else 256
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(cache, dict):
            cache = {}
        entries = cache.setdefault("entries", {})
        entries[key] = {"schema": AUTOTUNE_SCHEMA_VERSION, "tile_limit": resolved, "validated_sizes": passed}
        path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass
    return resolved


def _autotune_worker(timeout: float) -> int:
    """Probe once, close that context, then run isolated candidates."""
    os.environ["MATRIXMAN_BACKEND"] = "opengl"
    try:
        from drivers import matrixman
        from drivers.matrixman.backend import get_backend

        matrixman.config.reloadFromEnvironment()
        # Prevent the worker's capability probe from recursively autotuning.
        matrixman.config.tileLimit = 256
        matrixman.prefer("opengl")
        backend = get_backend()
        backend.device_info()
        matrixman.shutdown()
        passed = []
        for size in DEFAULT_SIZES:
            _one, practical = _memory_estimate(size)
            if practical <= MAX_ESTIMATED_BYTES and _run_child(size, timeout).get("result") == "PASS":
                passed.append(size)
        print(json.dumps({"passed": passed}), flush=True)
        return 0
    except BaseException as exc:
        print(json.dumps({"passed": [], "error": str(exc)}), flush=True)
        return 1


def _print_row(row: dict) -> None:
    stats = row.get("stats", {})
    print(
        f"{row['size']:4d}  {row['result']:<28} "
        f"actual={row.get('largest_physical_tile', ['-', '-'])[0]}x{row.get('largest_physical_tile', ['-', '-'])[1]} "
        f"path={row.get('path', '-')} atlas={row.get('atlas', ['-', '-'])[0]}x{row.get('atlas', ['-', '-'])[1]} "
        f"tiles={row.get('physical_tiles', '-')} runtime={row.get('runtime', 0):.3f}s "
        f"max_abs={stats.get('max_abs', float('nan')):.3g} mean_abs={stats.get('mean_abs', float('nan')):.3g} rmse={stats.get('rmse', float('nan')):.3g}"
    )
    if row.get("error"):
        print(f"      detail: {row['error']}")


def _parent(args: argparse.Namespace) -> int:
    os.environ["MATRIXMAN_BACKEND"] = "opengl"
    try:
        from drivers import matrixman
        from drivers.matrixman.backend import get_backend
        from drivers.matrixman.backends.opengl import gpumatrix as gl

        matrixman.config.reloadFromEnvironment()
        matrixman.prefer("opengl")
        backend = get_backend()
        if backend.name != "opengl":
            print(f"OpenGL diagnostic requires OpenGL; selected {backend.name}")
            return 1
        info = backend.device_info()
        limits = _limits(gl)
        print("MatrixMan OpenGL physical tile validation")
        for label in ("renderer", "vendor", "opengl", "glsl"):
            print(f"{label}: {_text(info.get(label))}")
        for name, value in limits.items():
            print(f"{name}: {value if not isinstance(value, tuple) else 'x'.join(map(str, value))}")
        print("Memory estimates per candidate: one RGBA32F texture = size*size*16 bytes; practical Conv footprint ~= 4x (not exact allocations)")
        matrixman.shutdown()
    except BaseException as exc:
        print(f"OpenGL diagnostic unavailable: {exc}")
        return 1

    rows = []
    max_texture = int(limits["GL_MAX_TEXTURE_SIZE"])
    viewport = limits["GL_MAX_VIEWPORT_DIMS"]
    max_viewport = min(viewport)
    for size in args.sizes:
        one, practical = _memory_estimate(size)
        print(f"candidate {size}: estimated texture={one / 2**20:.1f} MiB, practical Conv={practical / 2**20:.1f} MiB")
        if size > args.max_size:
            row = {"size": size, "result": "SKIP exceeds --max-size"}
        elif size > max_texture or size > max_viewport:
            row = {"size": size, "result": "SKIP exceeds reported GL limits"}
        elif practical > MAX_ESTIMATED_BYTES:
            row = {"size": size, "result": "SKIP estimated memory unreasonable"}
        else:
            row = _run_child(size, args.timeout)
        rows.append(row)
        _print_row(row)

    print("\nYOLO Conv atlas check ([1,64,80,80] -> [1,64,80,80], atlas 320x320)")
    for limit in (256, 512):
        row = _run_child(limit, args.timeout, spatial=80)
        print(f"tile limit {limit}: {row.get('path', row['result'])}; actual={row.get('largest_physical_tile', ['-', '-'])[0]}x{row.get('largest_physical_tile', ['-', '-'])[1]}")

    passed_sizes = [row["size"] for row in rows if row.get("result") == "PASS"]
    print(f"\nlargest tile size validated by this diagnostic: {max(passed_sizes) if passed_sizes else 'none'}")
    print("MatrixMan production defaults remain: MATRIXMAN_TILE_LIMIT=256, MATRIXMAN_TILE_SYNC=per_tile")
    return 0 if passed_sizes else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)), help="comma-separated physical sizes")
    parser.add_argument("--max-size", type=int, default=max(DEFAULT_SIZES))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="child timeout in seconds")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--autotune-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--spatial", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if not args.size or args.size <= 0:
            parser.error("--child requires positive --size")
        return _child(args.size, args.spatial)
    if args.autotune_worker:
        return _autotune_worker(args.timeout)
    try:
        args.sizes = tuple(sorted({int(item) for item in args.sizes.split(",") if item.strip()}))
        if not args.sizes or any(size <= 0 for size in args.sizes):
            raise ValueError
    except ValueError:
        parser.error("--sizes must be a comma-separated list of positive integers")
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
