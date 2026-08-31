"""MatrixMan CUDA backend façade."""

from ...backend import Backend
from .gpumatrix import CudaExecutionBackend, check as check_cuda, detect_device


class CudaBackend(Backend):
    """Backend interface adapter around the reusable CUDA execution runtime."""

    name = "cuda"

    def __init__(self):
        self.execution = CudaExecutionBackend()

    def device_info(self) -> dict[str, str]:
        return {"backend": "CUDA", **self.execution.info}

    def synchronize(self):
        check_cuda(
            self.execution.driver,
            self.execution.driver.cuCtxSynchronize(),
            "cuCtxSynchronize",
        )

    def close(self) -> None:
        self.execution.close()

    @classmethod
    def probe(cls) -> bool:
        try:
            detect_device()
        except Exception:
            return False
        return True
