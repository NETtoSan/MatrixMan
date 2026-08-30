#!/usr/bin/env python3
"""Tiny legacy-CUDA matrix smoke test for MatrixMan.

This is intentionally independent of PyTorch and the MatrixMan PrivateUse1
frontend.  It uses the CUDA Driver API through ``ctypes`` and loads embedded
PTX targeted at Compute Capability 2.1:

    NumPy host memory -> cuMemAlloc device memory -> CUDA kernel
    -> cuMemcpyDtoH host memory -> NumPy validation

The PTX is deliberately small and naive.  It is a compatibility probe, not a
production CUDA backend.  In particular, this file does not use torch.cuda,
unified memory, tensor cores, cooperative groups, or CPU arithmetic as a
fallback.
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


# CUDA 3.x PTX syntax is supported by the legacy driver range this probe is
# intended for.  Each of the 16 threads handles one element of a 4x4 matrix.
PTX = r"""
.version 3.0
.target sm_21
.address_size 64

.visible .entry matrix_add(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 p_n
)
{
    .reg .pred %p<1>;
    .reg .u32 %r<5>;
    .reg .u64 %rd<5>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %tid.x;
    setp.ge.u32 %p0, %r2, 16;
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
    .param .u32 p_n
)
{
    .reg .pred %p<2>;
    .reg .u32 %r<12>;
    .reg .u64 %rd<8>;
    .reg .f32 %f<5>;

    ld.param.u64 %rd1, [p_a];
    ld.param.u64 %rd2, [p_b];
    ld.param.u64 %rd3, [p_out];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %tid.x;
    setp.ge.u32 %p0, %r2, 16;
    @%p0 bra DONE;
    div.u32 %r3, %r2, %r1;
    rem.u32 %r4, %r2, %r1;
    mov.u32 %r5, 0;
    mov.f32 %f1, 0.0;
LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra STORE;
    mul.lo.u32 %r6, %r3, %r1;
    add.u32 %r6, %r6, %r5;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.lo.u32 %r7, %r5, %r1;
    add.u32 %r7, %r7, %r4;
    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    mul.f32 %f4, %f2, %f3;
    add.f32 %f1, %f1, %f4;
    add.u32 %r5, %r5, 1;
    bra LOOP;
STORE:
    mul.wide.u32 %rd4, %r2, 4;
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
        raise RuntimeError(
            "CUDA Driver API unavailable; install/load a legacy NVIDIA driver "
            "before running this probe"
        ) from exc


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


def launch(driver, function, a_ptr, b_ptr, out_ptr, n: int) -> None:
    n_arg = ctypes.c_uint(n)
    params = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.byref(a_ptr), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(b_ptr), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(out_ptr), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
    )
    check(driver, driver.cuLaunchKernel(function, 1, 1, 1, 16, 1, 1, 0, None, params, None), "cuLaunchKernel")
    check(driver, driver.cuCtxSynchronize(), "cuCtxSynchronize")


def run_kernel(driver, function, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    size = a.nbytes
    a_ptr = CUdeviceptr()
    b_ptr = CUdeviceptr()
    out_ptr = CUdeviceptr()
    for ptr, label in ((a_ptr, "A"), (b_ptr, "B"), (out_ptr, "output")):
        check(driver, driver.cuMemAlloc_v2(ctypes.byref(ptr), size), f"allocate {label}")
    try:
        check(driver, driver.cuMemcpyHtoD_v2(a_ptr, a.ctypes.data_as(ctypes.c_void_p), size), "upload A")
        check(driver, driver.cuMemcpyHtoD_v2(b_ptr, b.ctypes.data_as(ctypes.c_void_p), size), "upload B")
        launch(driver, function, a_ptr, b_ptr, out_ptr, a.shape[0])
        output = np.empty_like(a)
        check(driver, driver.cuMemcpyDtoH_v2(output.ctypes.data_as(ctypes.c_void_p), out_ptr, size), "read output")
        return output
    finally:
        for ptr in (a_ptr, b_ptr, out_ptr):
            if ptr.value:
                check(driver, driver.cuMemFree_v2(ptr), "free device memory")


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


def main() -> int:
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
    print("CUDA device:", name.value.decode(errors="replace"))
    print("compute capability:", f"{major.value}.{minor.value}")
    print("device memory (MiB):", total_mem.value / (1024 * 1024))
    if (major.value, minor.value) != (2, 1):
        print("warning: embedded PTX targets sm_21; this is not the requested GT 720M capability")

    context = CUcontext()
    module = CUmodule()
    try:
        check(driver, driver.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
        ptx = ctypes.create_string_buffer(PTX)
        check(driver, driver.cuModuleLoadData(ctypes.byref(module), ctypes.cast(ptx, ctypes.c_void_p)), "cuModuleLoadData")
        add_function = CUfunction()
        mul_function = CUfunction()
        check(driver, driver.cuModuleGetFunction(ctypes.byref(add_function), module, b"matrix_add"), "get matrix_add")
        check(driver, driver.cuModuleGetFunction(ctypes.byref(mul_function), module, b"matrix_mul"), "get matrix_mul")

        n = 4
        a = np.arange(1, n * n + 1, dtype=np.float32).reshape(n, n)
        b = np.arange(n * n, 0, -1, dtype=np.float32).reshape(n, n)
        np.set_printoptions(precision=3, suppress=True)
        print("\nA:")
        print(a)
        print("\nB:")
        print(b)
        print_check("C = A + B", run_kernel(driver, add_function, a, b), a + b)
        print_check("C = A x B", run_kernel(driver, mul_function, a, b), a @ b)
        return 0
    finally:
        if module:
            check(driver, driver.cuModuleUnload(module), "cuModuleUnload")
        if context:
            check(driver, driver.cuCtxDestroy_v2(context), "cuCtxDestroy")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
