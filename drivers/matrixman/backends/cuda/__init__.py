"""Reserved for a future NVIDIA CUDA MatrixMan backend.

CUDA is not currently supported by MatrixMan.
"""

from .backend import CudaBackend

__all__ = ["CudaBackend"]
