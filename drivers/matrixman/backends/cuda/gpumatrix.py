#!/usr/bin/env python3
"""Reusable legacy-CUDA execution helpers and a small standalone diagnostic.

The execution path uses only the CUDA Driver API through ``ctypes`` and an
embedded PTX module targeted at ``sm_21``.  It does not require ``nvcc`` or a
CUDA Toolkit at runtime, and it does not fall back to CPU arithmetic.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

import numpy as np


CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUdeviceptr = ctypes.c_uint64
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CUDA_BLOCK_SIZE = 128


PTX = r"""
.version 3.0
.target sm_21
.address_size 64

.visible .entry matrix_add(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 p_count
)
{
    .reg .pred %p<1>;
    .reg .u32 %r<5>;
    .reg .u64 %rd<5>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_count];
    // Legacy PTX requires special registers to be read with mov/cvt first.
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.u32 %r2, %r2, %r3;
    mov.u32 %r4, %tid.x;
    add.u32 %r2, %r2, %r4;
    setp.ge.u32 %p0, %r2, %r1;
    @%p0 bra DONE;
    mul.wide.u32 %rd4, %r2, 4;
    add.u64 %rd1, %rd1, %rd4;
    add.u64 %rd2, %rd2, %rd4;
    add.u64 %rd3, %rd3, %rd4;
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd3], %f3;
DONE:
    ret;
}

.visible .entry matrix_mul(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 p_m,
    .param .u32 p_k,
    .param .u32 p_n
)
{
    .reg .pred %p<2>;
    .reg .u32 %r<14>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<5>;
    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_m];
    ld.param.u32 %r2, [p_k];
    ld.param.u32 %r3, [p_n];
    mov.u32 %r4, %ctaid.x;
    mov.u32 %r11, %ntid.x;
    mul.lo.u32 %r4, %r4, %r11;
    mov.u32 %r12, %tid.x;
    add.u32 %r4, %r4, %r12;
    mul.lo.u32 %r5, %r1, %r3;
    setp.ge.u32 %p0, %r4, %r5;
    @%p0 bra DONE;
    div.u32 %r6, %r4, %r3;
    rem.u32 %r7, %r4, %r3;
    mov.u32 %r8, 0;
    mov.f32 %f1, 0.0;
LOOP:
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra STORE;
    mul.lo.u32 %r9, %r6, %r2;
    add.u32 %r9, %r9, %r8;
    mul.wide.u32 %rd4, %r9, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.lo.u32 %r10, %r8, %r3;
    add.u32 %r10, %r10, %r7;
    mul.wide.u32 %rd6, %r10, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r8, %r8, 1;
    bra LOOP;
STORE:
    mul.wide.u32 %rd4, %r4, 4;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f1;
DONE:
    ret;
}
""".encode("ascii")


def load_driver() -> ctypes.CDLL:
    path = ctypes.util.find_library("cuda") or "libcuda.so.1"
    try:
        return ctypes.CDLL(path)
    except OSError as exc:
        raise RuntimeError("CUDA Driver API unavailable") from exc


def configure_driver(driver: ctypes.CDLL) -> None:
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = CUresult
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = CUresult
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
    driver.cuDeviceGet.restype = CUresult
    driver.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, CUdevice]
    driver.cuDeviceGetName.restype = CUresult
    driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, CUdevice]
    driver.cuDeviceGetAttribute.restype = CUresult
    driver.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), CUdevice]
    driver.cuDeviceTotalMem_v2.restype = CUresult
    driver.cuCtxCreate_v2.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
    driver.cuCtxCreate_v2.restype = CUresult
    driver.cuCtxDestroy_v2.argtypes = [CUcontext]
    driver.cuCtxDestroy_v2.restype = CUresult
    driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
    driver.cuMemAlloc_v2.restype = CUresult
    driver.cuMemFree_v2.argtypes = [CUdeviceptr]
    driver.cuMemFree_v2.restype = CUresult
    driver.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]
    driver.cuMemcpyHtoD_v2.restype = CUresult
    driver.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
    driver.cuMemcpyDtoH_v2.restype = CUresult
    driver.cuModuleLoadData.argtypes = [ctypes.POINTER(CUmodule), ctypes.c_void_p]
    driver.cuModuleLoadData.restype = CUresult
    driver.cuModuleUnload.argtypes = [CUmodule]
    driver.cuModuleUnload.restype = CUresult
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p]
    driver.cuModuleGetFunction.restype = CUresult
    driver.cuLaunchKernel.argtypes = [
        CUfunction, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    driver.cuLaunchKernel.restype = CUresult
    driver.cuCtxSynchronize.argtypes = []
    driver.cuCtxSynchronize.restype = CUresult
    driver.cuGetErrorName.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorName.restype = CUresult
    driver.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorString.restype = CUresult


def check(driver: ctypes.CDLL, result: int, action: str) -> None:
    if result == CUDA_SUCCESS:
        return
    name = ctypes.c_char_p()
    message = ctypes.c_char_p()
    driver.cuGetErrorName(result, ctypes.byref(name))
    driver.cuGetErrorString(result, ctypes.byref(message))
    raise RuntimeError(
        f"{action}: {name.value.decode() if name.value else result} "
        f"({message.value.decode() if message.value else 'unknown error'})"
    )


def detect_device(driver: ctypes.CDLL | None = None) -> tuple[ctypes.CDLL, dict[str, str]]:
    """Initialize CUDA and return the first device's capability information."""
    if driver is None:
        driver = load_driver()
        configure_driver(driver)
    check(driver, driver.cuInit(0), "cuInit")
    count = ctypes.c_int()
    check(driver, driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    if count.value < 1:
        raise RuntimeError("CUDA initialized, but no CUDA devices were found")
    device = CUdevice()
    check(driver, driver.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet(0)")
    name = ctypes.create_string_buffer(256)
    major = ctypes.c_int()
    minor = ctypes.c_int()
    total_mem = ctypes.c_size_t()
    check(driver, driver.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
    check(driver, driver.cuDeviceGetAttribute(ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "query compute capability major")
    check(driver, driver.cuDeviceGetAttribute(ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "query compute capability minor")
    check(driver, driver.cuDeviceTotalMem_v2(ctypes.byref(total_mem), device), "cuDeviceTotalMem")
    return driver, {
        "name": name.value.decode(errors="replace"),
        "compute_capability": f"{major.value}.{minor.value}",
        "memory_mib": f"{total_mem.value / (1024 * 1024):.1f}",
    }


class CudaExecutionBackend:
    """Minimal device-memory and kernel execution boundary for MatrixMan."""

    def __init__(self, device_index: int = 0):
        if device_index != 0:
            raise ValueError("only CUDA device 0 is currently supported")
        self.driver = load_driver()
        configure_driver(self.driver)
        self.driver, self.info = detect_device(self.driver)
        self.device = CUdevice()
        check(self.driver, self.driver.cuDeviceGet(ctypes.byref(self.device), device_index), "cuDeviceGet")
        self.context = CUcontext()
        self.module = CUmodule()
        self.closed = False
        try:
            check(self.driver, self.driver.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device), "cuCtxCreate")
            ptx = ctypes.create_string_buffer(PTX)
            check(
                self.driver,
                self.driver.cuModuleLoadData(
                    ctypes.byref(self.module), ctypes.cast(ptx, ctypes.c_void_p)
                ),
                "cuModuleLoadData (embedded sm_21 PTX: matrix_add, matrix_mul)",
            )
            self.add_function = CUfunction()
            self.matmul_function = CUfunction()
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.add_function), self.module, b"matrix_add"), "get matrix_add")
            check(self.driver, self.driver.cuModuleGetFunction(ctypes.byref(self.matmul_function), self.module, b"matrix_mul"), "get matrix_mul")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        if self.module:
            check(self.driver, self.driver.cuModuleUnload(self.module), "cuModuleUnload")
            self.module = CUmodule()
        if self.context:
            check(self.driver, self.driver.cuCtxDestroy_v2(self.context), "cuCtxDestroy")
            self.context = CUcontext()
        self.closed = True

    def __enter__(self) -> "CudaExecutionBackend":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def allocate(self, nbytes: int) -> CUdeviceptr:
        if nbytes <= 0:
            raise ValueError("device allocation size must be positive")
        pointer = CUdeviceptr()
        check(self.driver, self.driver.cuMemAlloc_v2(ctypes.byref(pointer), nbytes), "cuMemAlloc")
        return pointer

    def free(self, pointer: CUdeviceptr) -> None:
        if pointer and pointer.value:
            check(self.driver, self.driver.cuMemFree_v2(pointer), "cuMemFree")
            pointer.value = 0

    @staticmethod
    def _float32_array(array: np.ndarray, label: str) -> np.ndarray:
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise TypeError(f"{label} must be a NumPy float32 array")
        if not array.flags.c_contiguous:
            raise ValueError(f"{label} must be C-contiguous")
        return array

    def to_device(self, array: np.ndarray) -> CUdeviceptr:
        array = self._float32_array(array, "host array")
        pointer = self.allocate(array.nbytes)
        try:
            check(self.driver, self.driver.cuMemcpyHtoD_v2(pointer, array.ctypes.data_as(ctypes.c_void_p), array.nbytes), "cuMemcpyHtoD")
        except Exception:
            self.free(pointer)
            raise
        return pointer

    def from_device(self, pointer: CUdeviceptr, shape: tuple[int, ...]) -> np.ndarray:
        output = np.empty(shape, dtype=np.float32)
        check(self.driver, self.driver.cuMemcpyDtoH_v2(output.ctypes.data_as(ctypes.c_void_p), pointer, output.nbytes), "cuMemcpyDtoH")
        return output

    def _launch(self, function: CUfunction, args: list[ctypes._SimpleCData], work_items: int) -> None:
        if work_items <= 0:
            raise ValueError("kernel work size must be positive")
        params = (ctypes.c_void_p * len(args))(
            *(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in args)
        )
        grid = (work_items + CUDA_BLOCK_SIZE - 1) // CUDA_BLOCK_SIZE
        check(self.driver, self.driver.cuLaunchKernel(function, grid, 1, 1, CUDA_BLOCK_SIZE, 1, 1, 0, None, params, None), "cuLaunchKernel")
        check(self.driver, self.driver.cuCtxSynchronize(), "cuCtxSynchronize")

    def add(self, a: CUdeviceptr, b: CUdeviceptr, output: CUdeviceptr, count: int) -> None:
        self._launch(self.add_function, [a, b, output, ctypes.c_uint(count)], count)

    def matmul(self, a: CUdeviceptr, b: CUdeviceptr, output: CUdeviceptr, m: int, k: int, n: int) -> None:
        if min(m, k, n) <= 0:
            raise ValueError("matrix dimensions must be positive")
        self._launch(
            self.matmul_function,
            [a, b, output, ctypes.c_uint(m), ctypes.c_uint(k), ctypes.c_uint(n)],
            m * n,
        )


def print_check(name: str, gpu: np.ndarray, cpu: np.ndarray) -> None:
    matches = np.allclose(gpu, cpu, rtol=1e-5, atol=1e-5)
    print(f"\n{name} CUDA result:")
    print(gpu)
    print(f"\n{name} NumPy reference:")
    print(cpu)
    print("max abs diff:", float(np.max(np.abs(gpu - cpu))))
    print("matches CPU reference:", bool(matches))
    if not matches:
        raise RuntimeError(f"{name} result does not match CPU reference")


def run_diagnostic() -> int:
    n = 4
    a = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
    b = np.arange(n * n, 0, -1, dtype=np.float32).reshape(n, n)
    np.set_printoptions(precision=3, suppress=True)
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        print("\nA:")
        print(a)
        print("\nB:")
        print(b)
        a_dev = cuda.to_device(a)
        b_dev = cuda.to_device(b)
        add_dev = cuda.allocate(a.nbytes)
        mul_dev = cuda.allocate(a.nbytes)
        try:
            cuda.add(a_dev, b_dev, add_dev, a.size)
            cuda.matmul(a_dev, b_dev, mul_dev, n, n, n)
            print_check("C = A + B", cuda.from_device(add_dev, a.shape), a + b)
            print_check("C = A x B", cuda.from_device(mul_dev, a.shape), a @ b)
        finally:
            for pointer in (a_dev, b_dev, add_dev, mul_dev):
                cuda.free(pointer)

        m, k, p = 3, 5, 7
        non_square_a = np.arange(1, m * k + 1, dtype=np.float32).reshape(m, k)
        non_square_b = np.arange(1, k * p + 1, dtype=np.float32).reshape(k, p)
        a_dev = cuda.to_device(non_square_a)
        b_dev = cuda.to_device(non_square_b)
        output_dev = cuda.allocate(m * p * np.dtype(np.float32).itemsize)
        try:
            cuda.matmul(a_dev, b_dev, output_dev, m, k, p)
            print_check(
                "C = (3x5) A x (5x7) B",
                cuda.from_device(output_dev, (m, p)),
                non_square_a @ non_square_b,
            )
        finally:
            for pointer in (a_dev, b_dev, output_dev):
                cuda.free(pointer)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_diagnostic())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
