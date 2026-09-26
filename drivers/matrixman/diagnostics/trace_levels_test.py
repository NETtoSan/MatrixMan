"""Hardware-independent checks for MatrixMan's three trace verbosity levels."""

from __future__ import annotations

import contextlib
import io
import os

from drivers import matrixman
from drivers.matrixman.backends.opengl import diagnostics, runtime
from drivers.matrixman.config import config


def _capture(level, *, context=False) -> str:
    matrixman.trace = level
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        diagnostics.kernel_log("high-level operation")
        diagnostics.trace("internal storage detail")
        if context:
            runtime._context_log("release", depth=0)
    return output.getvalue()


def main() -> int:
    old_trace = matrixman.trace
    old_environment = os.environ.get("MATRIXMAN_TRACE")
    try:
        os.environ.pop("MATRIXMAN_TRACE", None)

        quiet = _capture(False, context=True)
        if quiet:
            raise AssertionError(f"trace=False emitted output: {quiet!r}")

        concise = _capture(True, context=True)
        if "high-level operation" not in concise:
            raise AssertionError("trace=True did not emit the high-level event")
        if "internal storage detail" in concise or "gl_context release" in concise:
            raise AssertionError("trace=True emitted detailed output")

        detailed = _capture("detailed", context=True)
        if "high-level operation" not in detailed:
            raise AssertionError("detailed trace omitted the high-level event")
        if "internal storage detail" not in detailed or "gl_context release" not in detailed:
            raise AssertionError("detailed trace omitted internal output")

        try:
            matrixman.trace = "verbose"
        except ValueError:
            pass
        else:
            raise AssertionError("invalid trace level was accepted")

        os.environ["MATRIXMAN_TRACE"] = "detailed"
        config.reloadFromEnvironment()
        if matrixman.trace != "detailed":
            raise AssertionError(f"environment trace parsing failed: {matrixman.trace!r}")
    finally:
        if old_environment is None:
            os.environ.pop("MATRIXMAN_TRACE", None)
        else:
            os.environ["MATRIXMAN_TRACE"] = old_environment
        config.reloadFromEnvironment()
        matrixman.trace = old_trace

    print("Trace levels test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
