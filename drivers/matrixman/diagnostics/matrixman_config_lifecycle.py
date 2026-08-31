"""Small subprocess checks for MatrixMan's lazy backend lifecycle."""

from __future__ import annotations

import os
import subprocess
import sys


_IMPORT_CHECK = """
from drivers import matrixman
from drivers.matrixman.backend import active_backend
assert active_backend() is None
matrixman.prefer(BACKEND)
assert active_backend() is None
print('preference recorded before backend initialization')
"""

_USE_CHECK = """
import torch
from drivers import matrixman
from drivers.matrixman.backend import active_backend
matrixman.prefer(BACKEND)
matrixman.to_device(torch.zeros((1,), dtype=torch.float32))
print('initialized=' + active_backend().name)
"""

_SELECTION_COUNT_CHECK = """
import drivers.matrixman.selector as selector
from drivers.matrixman.backend import Backend, active_backend, get_backend, set_backend
calls = []
def fake_select():
    calls.append(1)
    return set_backend(type('TestBackend', (Backend,), {'name': 'opengl'})())
selector.select_backend = fake_select
from drivers import matrixman
matrixman.prefer('opengl')
assert not calls
assert active_backend() is None
get_backend()
assert len(calls) == 1
get_backend()
assert len(calls) == 1
print('selection_calls=1')
"""

_PROFILE_HOOK_CHECK = """
import atexit
registered = []
atexit.register = lambda callback: registered.append(callback)
from drivers import matrixman
from drivers.matrixman.backends.cuda import profiling as cuda_profile
from drivers.matrixman.backends.opengl import profiling as opengl_profile
profile_hooks = [
    callback for callback in registered
    if 'matrixman.backends.' in (getattr(callback, '__module__', '') or '')
]
assert not profile_hooks
before = len(registered)
cuda_profile.register_exit_hook()
cuda_profile.register_exit_hook()
assert len(registered) == before + 1
print('import_hooks=0 active_hook_count=1')
"""

_PREFERENCE_CHECK = """
import os
from drivers import matrixman
from drivers.matrixman.config import preferred_backend
assert preferred_backend() is None
matrixman.profiling = True
assert preferred_backend() is None
matrixman.prefer('auto')
assert preferred_backend() is None
matrixman.prefer('cuda')
assert preferred_backend() == 'cuda'
matrixman.profiling = False
assert preferred_backend() == 'cuda'
print('profiling_orthogonal=True')
"""

_DEFAULT_PROFILING_CHECK = """
import os
os.environ.pop('MATRIXMAN_PROFILE', None)
os.environ.pop('MATRIXMAN_CUDA_PROFILE', None)
from drivers import matrixman
assert matrixman.profiling is False
matrixman.prefer('auto')
assert matrixman.profiling is False
print('default_profiling=False')
"""

_TRACE_CHECK = """
import os
os.environ.pop('MATRIXMAN_TRACE', None)
os.environ.pop('MATRIXMAN_PROFILE', None)
os.environ.pop('MATRIXMAN_CUDA_PROFILE', None)
from drivers import matrixman
assert matrixman.trace is False
assert matrixman.profiling is False
matrixman.prefer('auto')
assert matrixman.trace is False
matrixman.trace = True
assert matrixman.trace is True
assert matrixman.profiling is False
matrixman.profiling = True
assert matrixman.trace is True
matrixman.trace = False
assert matrixman.trace is False
assert matrixman.profiling is True
matrixman.profiling = False
assert matrixman.trace is False
print('trace_default=False python_override_and_orthogonal=True')
"""

_TRACE_ENV_CHECK = """
from drivers import matrixman
assert matrixman.trace is True
matrixman.trace = False
assert matrixman.trace is False
matrixman.trace = True
assert matrixman.trace is True
print('trace_env_and_python_precedence=True')
"""

_TRACE_DISABLED_ENV_CHECK = """
from drivers import matrixman
assert matrixman.trace is False
print('trace_env_disabled=True')
"""

_OPENGL_POLICY_CHECK = """
from drivers.matrixman.backends.opengl.backend import classify_renderer
policy = classify_renderer('NVIDIA Corporation', 'GeForce GT 720M/PCIe/SSE2')
assert 'NVIDIA fallback' in policy
print('nvidia_available=True policy=' + policy)
"""


def _run(source: str, backend: str, environment: dict[str, str] | None = None):
    code = source.replace("BACKEND", repr(backend), 1)
    env = os.environ.copy()
    if environment:
        env.update(environment)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    print("MatrixMan configuration lifecycle")
    result = _run(_SELECTION_COUNT_CHECK, "auto")
    if result.returncode:
        print("  prefer-before-use selection count: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  prefer-before-use selection count: PASS")
    result = _run(_PROFILE_HOOK_CHECK, "auto")
    if result.returncode:
        print("  profiler import/registration isolation: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  profiler import/registration isolation: PASS")
    result = _run(_PREFERENCE_CHECK, "auto")
    if result.returncode:
        print("  profiling/preference orthogonality: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  profiling/preference orthogonality: PASS")
    result = _run(_DEFAULT_PROFILING_CHECK, "auto")
    if result.returncode:
        print("  unset profiling default: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  unset profiling default: PASS")
    result = _run(_TRACE_CHECK, "auto")
    if result.returncode:
        print("  trace default/orthogonality: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  trace default/orthogonality: PASS")
    result = _run(_TRACE_ENV_CHECK, "auto", {"MATRIXMAN_TRACE": "1"})
    if result.returncode:
        print("  trace environment precedence: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  trace environment precedence: PASS")
    result = _run(_TRACE_DISABLED_ENV_CHECK, "auto", {"MATRIXMAN_TRACE": "0"})
    if result.returncode:
        print("  trace environment disabled: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  trace environment disabled: PASS")
    result = _run(_OPENGL_POLICY_CHECK, "auto")
    if result.returncode:
        print("  OpenGL renderer policy classification: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  OpenGL renderer policy classification: PASS")
    for backend in ("cuda", "opengl"):
        result = _run(_IMPORT_CHECK, backend)
        if result.returncode:
            print(f"  prefer({backend!r}) before use: unavailable")
            print(result.stderr.strip() or result.stdout.strip())
        else:
            print(f"  prefer({backend!r}) before use: PASS")

    result = _run(_USE_CHECK, "opengl", {"MATRIXMAN_BACKEND": "opengl"})
    if result.returncode:
        print("  environment OpenGL selection: unavailable")
        print(result.stderr.strip() or result.stdout.strip())
    else:
        print("  environment OpenGL selection: PASS")

    switch_check = """
from drivers import matrixman
from drivers.matrixman.backend import set_backend, Backend
set_backend(type('CudaTestBackend', (Backend,), {'name': 'cuda'})())
try:
    matrixman.prefer('opengl')
except RuntimeError as exc:
    assert 'already initialized as CUDA' in str(exc)
    print('live switch rejected')
else:
    raise AssertionError('live backend switch was accepted')
"""
    result = _run(switch_check, "auto")
    if result.returncode:
        print("  initialized-backend switch rejection: FAIL")
        print(result.stderr.strip() or result.stdout.strip())
        return 1
    print("  initialized-backend switch rejection: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
