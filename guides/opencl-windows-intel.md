# OpenCL on Windows with Intel HD Graphics 530

This is a bring-up guide for running OpenCL kernels on the Intel HD Graphics 530 GPU. It is intentionally separate from the MatrixMan backend: do not change `drivers/matrixman/` until the standalone test succeeds.

## Bottom line

Yes. Intel HD Graphics 530 is the Gen9/Skylake integrated GPU and supports OpenCL on Windows when a suitable Intel graphics driver is installed. Intel's Windows API table lists HD Graphics 520/530 as OpenCL 2.0. Intel's older 2019 SDK documentation describes the Skylake graphics runtime as OpenCL 2.1. The exact value to trust on a particular machine is the device's runtime-reported `CL_DEVICE_VERSION` from `clinfo` or PyOpenCL.

Intel's current Graphics Compute Runtime (NEO) documentation lists Gen9/Skylake as a **legacy** platform, with OpenCL 3.0 in the current runtime support table. That table is primarily the current compute-runtime project/release line; Windows binaries are shipped through Intel graphics driver packages and may lag the GitHub code. Do not assume OpenCL 3.0 is available on this Windows installation—measure it.

For Windows, install the Intel graphics driver package for 6th–10th Generation Intel Processor Graphics (or the computer manufacturer's validated equivalent). Intel says the Intel Graphics Compute Runtime for OpenCL is included in the Intel Graphics Driver package for Windows. A separate Intel GPU OpenCL download is normally not required. The Intel CPU Runtime for OpenCL is a different implementation and is not needed to run kernels on the HD 530 GPU.

References:

- [Intel supported APIs for Intel Graphics](https://www.intel.com/content/www/us/en/support/articles/000005524/graphics.html)
- [Intel runtimes for Intel processors](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)
- [Intel Run OpenCL Code](https://www.intel.com/content/www/us/en/developer/tools/opencl/run.html)
- [Intel Graphics Compute Runtime / NEO](https://github.com/intel/compute-runtime)
- [NEO legacy platform table](https://github.com/intel/compute-runtime/blob/master/documentation/LEGACY_PLATFORMS.md)

## GPU versus CPU: keep them explicit

OpenCL is an API, not a GPU-only technology. A machine can expose an Intel CPU OpenCL platform, an Intel GPU platform, both, or neither. Installing `intel-opencl-rt`/Intel CPU Runtime only gives a CPU device; it does not enable HD 530 GPU execution.

The Intel graphics driver/ICD should expose a device whose type is `GPU` and whose name contains `Intel(R) HD Graphics 530` (wording can vary). In Python, select by `cl.device_type.GPU` and inspect the device name. Never select `platform.devices[0]` without checking its type and name.

## A. Install and verify the graphics driver

1. Identify the exact GPU in Device Manager (`Display adapters`) and record the driver version. Use Intel's [List of Drivers for Intel Graphics](https://www.intel.com/content/www/us/en/support/articles/000090440/graphics.html) and select the **Intel Graphics Driver for Windows [15.45]** entry for 6th Gen HD Graphics 520/530, or prefer the OEM driver for a laptop/embedded system.
2. Reboot after installation.
3. Confirm that Device Manager reports `Intel(R) HD Graphics 530` without a warning icon.

HD 530 is an older/legacy device. Newer Intel driver branches may not support it; use the latest driver package that explicitly includes 6th Gen/Skylake, rather than forcing an unrelated modern package. Intel's current NEO project calls Gen9 maintenance/legacy support; this is different from the current production platforms.

## B. Verify the Windows ICD loader and registration

Run these in 64-bit PowerShell:

```powershell
$sys = Join-Path $env:WINDIR 'System32'
Test-Path (Join-Path $sys 'OpenCL.dll')
Get-Item (Join-Path $sys 'OpenCL.dll') | Select-Object FullName,Length,VersionInfo

# Traditional Khronos ICD registration (may be empty on newer Windows drivers)
reg query 'HKLM\SOFTWARE\Khronos\OpenCL\Vendors'
reg query 'HKLM\SOFTWARE\WOW6432Node\Khronos\OpenCL\Vendors'

# Look for Intel's ICD implementation in installed driver packages
Get-ChildItem "$env:WINDIR\System32\DriverStore\FileRepository" -Recurse `
  -Filter 'Intel_OpenCL_ICD64.dll' -ErrorAction SilentlyContinue |
  Select-Object FullName,Length,LastWriteTime
```

`OpenCL.dll` is the Khronos ICD loader, not the Intel implementation. The loader discovers vendor ICDs through Windows registry data. Current Khronos documentation says it checks PnP Display Adapter/Software Component `OpenCLDriverName` values as well as the traditional `HKLM\SOFTWARE\Khronos\OpenCL\Vendors` key. Therefore an empty traditional key is not conclusive.

Reference: [Khronos OpenCL ICD installation and Windows registration](https://github.com/KhronosGroup/OpenCL-Docs/blob/main/OpenCL_ICD_Installation.txt) and [current ICD enumeration rules](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/cl_khr_icd.html).

## C. Run `clinfo`

Use the lightweight `clinfo` utility. The commonly used implementation is [Oblomov/clinfo](https://github.com/Oblomov/clinfo); use a trusted prebuilt Windows binary or build it from source. Then run:

```powershell
clinfo -l                 # short platform/device list
clinfo                    # full report
clinfo | Select-String 'Platform Name|Device Name|Device Type|Device Version|Global memory|Max work-group size|Compute Units'
```

The report must contain all of the following for the target device:

- device name: Intel HD Graphics 530;
- device type: `GPU`;
- device version: record the reported OpenCL version;
- global memory size;
- maximum work-group size;
- max compute units.

If `clinfo` says `PLATFORM_NOT_FOUND_KHR`, the Python package is not the problem: the loader has no usable registered ICD. If it lists only an Intel CPU device, the CPU runtime is working but the GPU driver/ICD is missing or not exposing the GPU. If it lists both, choose the GPU explicitly.

## D–G. Python bring-up

PyOpenCL is appropriate for a standalone MatrixMan experiment. It is a Python binding; it does not install an Intel GPU driver. Use a 64-bit CPython matching the 64-bit driver process.

The normal install is:

```powershell
py -3.13 -m venv .venv-opencl
.\.venv-opencl\Scripts\python.exe -m pip install --upgrade pip
.\.venv-opencl\Scripts\python.exe -m pip install pyopencl numpy
```

PyOpenCL publishes Windows wheels, so Visual Studio/C++ is normally not required. If pip falls back to a source build, stop and check the Python version and wheel availability before installing a compiler toolchain. Very new Python releases can temporarily lack compatible wheels. This machine currently has Python 3.14; use 3.13 (or 3.12/3.11) for the first experiment unless the current PyPI page confirms a regular `cp314-win_amd64` wheel for the selected PyOpenCL release. Do not mix a PyPI PyOpenCL environment with unrelated Conda OpenCL loader packages.

First enumerate and validate the GPU:

```python
import pyopencl as cl

for p in cl.get_platforms():
    print("PLATFORM", p.name, "|", p.version)
    for d in p.get_devices():
        print("  DEVICE", d.name)
        print("    type:", cl.device_type.to_string(d.type))
        print("    version:", d.version)
        print("    global memory:", d.global_mem_size)
        print("    max work-group:", d.max_work_group_size)
        print("    compute units:", d.max_compute_units)

gpu = [d for p in cl.get_platforms() for d in p.get_devices()
       if d.type & cl.device_type.GPU and "530" in d.name]
if not gpu:
    raise RuntimeError("Intel HD Graphics 530 GPU device was not found")
device = gpu[0]
print("SELECTED GPU:", device.name)
```

Then run a tiny vector-add kernel on that exact device:

```python
import numpy as np
import pyopencl as cl

ctx = cl.Context(devices=[device])
queue = cl.CommandQueue(ctx)
src = r"""
__kernel void add(__global const float *a,
                  __global const float *b,
                  __global float *out) {
    size_t i = get_global_id(0);
    out[i] = a[i] + b[i];
}
"""
program = cl.Program(ctx, src).build()
n = 1024
a = np.arange(n, dtype=np.float32)
b = np.ones(n, dtype=np.float32)
out = np.empty_like(a)
mf = cl.mem_flags
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
bo = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
program.add(queue, (n,), None, ba, bb, bo)
cl.enqueue_copy(queue, out, bo).wait()
np.testing.assert_allclose(out, a + b)
print("vector-add OK on", device.name)
```

Only after this succeeds should the experiment add MatrixMan-specific code, followed by matrix multiplication. For every experiment, print the selected platform/device and assert `GPU` before timing or benchmarking.

## Current machine snapshot

The repository workstation was inspected without changing MatrixMan code. At inspection time it had:

- `C:\Windows\System32\OpenCL.dll` and `SysWOW64\OpenCL.dll`;
- Intel `Intel_OpenCL_ICD64.dll`/`Intel_OpenCL_ICD32.dll` files under the Windows DriverStore;
- no `clinfo` command on PATH;
- Python 3.14.7, with PyOpenCL not installed;
- no values returned by the traditional Khronos vendor registry queries.

The GPU/device result still needs to be established by running `clinfo` on the actual desktop session. The practical next action is therefore: install/repair the HD 530-compatible Intel graphics driver, install/run `clinfo`, and require an Intel HD 530 **GPU** device in its output. If that appears, this machine is suitable for first MatrixMan OpenCL experiments.
