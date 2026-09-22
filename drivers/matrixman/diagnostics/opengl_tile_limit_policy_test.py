"""Pure policy/cache tests for OpenGL physical tile-limit resolution."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from drivers.matrixman.diagnostics import opengl_tile_limit as tiles


LIMITS = {
    "GL_MAX_TEXTURE_SIZE": 4096,
    "GL_MAX_VIEWPORT_DIMS": (4096, 4096),
    "GL_MAX_RENDERBUFFER_SIZE": 4096,
}


def main() -> int:
    assert tiles.resolve_tile_limit_policy(256, {}, LIMITS).resolved == 256
    assert tiles.resolve_tile_limit_policy(512, {}, LIMITS).resolved == 512

    for renderer in (
        "Mesa Mobile Intel(R) GM45 Express Chipset (CTG)",
        "Intel GM45",
        "Intel GMA 4500MHD",
    ):
        policy = tiles.resolve_tile_limit_policy("auto", {"renderer": renderer}, LIMITS, 4096)
        assert policy.resolved == 4096, (renderer, policy)

    bounded = dict(LIMITS)
    bounded["GL_MAX_TEXTURE_SIZE"] = 768
    bounded["GL_MAX_VIEWPORT_DIMS"] = (1024, 768)
    bounded["GL_MAX_RENDERBUFFER_SIZE"] = 2048
    assert tiles.resolve_tile_limit_policy("auto", {"renderer": "Unknown GPU"}, bounded, 4096).resolved == 768
    assert tiles.resolve_tile_limit_policy("auto", {"renderer": "Intel GM45"}, LIMITS, 2048).resolved == 2048
    try:
        tiles.resolve_tile_limit_policy(4097, {}, LIMITS)
    except ValueError:
        pass
    else:
        raise AssertionError("explicit tile limit above GL bounds was accepted")

    original_path = tiles._cache_path
    original_child = tiles._run_child
    try:
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "tile-cache.json"
            cache_path.write_text(json.dumps({
                "entries": {tiles._cache_key({"vendor": "v", "renderer": "r", "opengl": "2.1", "glsl": "1.20"}): {
                    "schema": 1, "tile_limit": 4096,
                }}
            }))
            tiles._cache_path = lambda: cache_path
            calls = []
            progress = []
            def fake_child(size, timeout, spatial=None):
                calls.append(size)
                return {"result": "FAIL" if size == 512 else "PASS"}
            tiles._run_child = fake_child
            assert tiles.autotune_tile_limit(
                {"vendor": "v", "renderer": "r", "opengl": "2.1", "glsl": "1.20"},
                progress=progress.append,
            ) == 2048
            assert calls == list(tiles.DEFAULT_SIZES)
            assert "trying 512 ... [FAIL]" in progress
            assert "trying 2048 ... [OK]" in progress
            refreshed = json.loads(cache_path.read_text())
            entry = next(iter(refreshed["entries"].values()))
            assert entry["schema"] == tiles.AUTOTUNE_SCHEMA_VERSION
            assert entry["policy_version"] == tiles.AUTOTUNE_POLICY_VERSION
            tiles._run_child = lambda size, timeout, spatial=None: (_ for _ in ()).throw(AssertionError("cache miss"))
            cached_progress = []
            assert tiles.autotune_tile_limit(
                {"vendor": "v", "renderer": "r", "opengl": "2.1", "glsl": "1.20"},
                progress=cached_progress.append,
            ) == 2048
            assert cached_progress == ["cached result: 2048"]

            fail_path = Path(directory) / "failed-cache.json"
            tiles._cache_path = lambda: fail_path
            tiles._run_child = lambda size, timeout, spatial=None: {"result": "FAIL"}
            failed_progress = []
            assert tiles.autotune_tile_limit(
                {"vendor": "v", "renderer": "r", "opengl": "2.1", "glsl": "1.20"},
                progress=failed_progress.append,
            ) == 256
            assert failed_progress[-1] == "autotune failed; falling back to tileLimit: 256"
    finally:
        tiles._cache_path = original_path
        tiles._run_child = original_child

    print("tile-limit policy tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
