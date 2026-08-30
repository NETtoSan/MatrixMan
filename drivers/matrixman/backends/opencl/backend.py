"""Placeholder for a future cross-vendor/legacy compute backend."""

from ...backend import Backend


class OpenCLBackend(Backend):
    """Unavailable placeholder; OpenCL support is not implemented."""

    name = "opencl"

    @classmethod
    def probe(cls) -> bool:
        return False
