"""Focused checks for the public Python configuration surface."""

from __future__ import annotations

import os
import subprocess
import sys


def _child(source: str, **environment: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    env["PYTHONPATH"] = os.getcwd()
    return subprocess.run(
        [sys.executable, "-c", source], env=env, text=True,
        capture_output=True, check=False,
    )


def main() -> int:
    from drivers import matrixman
    from drivers.matrixman.backend import Backend, set_backend

    assert matrixman.profile == "off"
    assert matrixman.config.tile_limit == 256
    matrixman.profile = "summary"
    matrixman.trace = False
    matrixman.config.tile_limit = 128
    assert matrixman.profile == "summary"
    assert matrixman.config.tile_limit == 128

    for name, value in (("sync_policy", "banana"), ("tile_limit", -5)):
        try:
            setattr(matrixman.config, name, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {name} was accepted")

    listing = matrixman.config.show(as_text=True)
    assert "profile: summary [python override]" in listing
    assert "tile_limit: 128 [python override]" in listing
    assert "scratch_policy: safe [default]" in listing
    assert "sync_policy" in dir(matrixman.config)
    assert "activation_pool" in matrixman.config.keys()
    assert dict(matrixman.config.items())["sync_policy"] == "safe"
    assert "MatrixMan configuration" in str(matrixman.config)
    assert "MatrixManConfig(" in repr(matrixman.config)
    assert matrixman.config.options("sync_policy") == ("safe", "auto", "relaxed")
    assert matrixman.config.options("tile_limit")["range"] == "positive integer"
    assert matrixman.config.metadata("activation_pool")["environment"] == "MATRIXMAN_ACTIVATION_POOL"
    assert "activation_pool" in matrixman.config.describe("activation_pool")
    assert "sync_policy" in matrixman.config.describe()
    assert matrixman.config.help("tile_limit")
    try:
        matrixman.config.describe("not_a_setting")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid describe name was accepted")
    assert matrixman.profile == "summary"
    assert matrixman.trace is False

    result = _child(
        """
from drivers import matrixman
assert matrixman.profile == 'detail'
assert matrixman.config.tile_limit == 128
assert matrixman.config.source('tile_limit') == 'environment'
matrixman.config.tile_limit = 256
assert matrixman.config.source('tile_limit') == 'python override'
assert matrixman.config.tile_limit == 256
print('environment precedence: PASS')
""",
        MATRIXMAN_PROFILE="detail", MATRIXMAN_TILE_LIMIT="128",
    )
    if result.returncode:
        print(result.stdout, result.stderr)
        return result.returncode

    class FakeBackend(Backend):
        name = "opengl"

    set_backend(FakeBackend())
    try:
        matrixman.config.tile_limit = 512
    except RuntimeError as exc:
        assert "must be set before backend initialization" in str(exc)
    else:
        raise AssertionError("protected setting changed after initialization")

    print("MatrixMan Python configuration tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
