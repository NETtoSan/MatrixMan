"""Placeholder for a future NVIDIA compute backend."""

from ...backend import Backend


class CudaBackend(Backend):
    """Unavailable placeholder; CUDA support is not implemented."""

    name = "cuda"

    @classmethod
    def probe(cls) -> bool:
        return False
