# MatrixMan

> **Does it run? YES.**  
> **Is it fast? NO.**  
> **Did modern PyTorch drop your GPU? MatrixMan may still have ideas.**

MatrixMan is an experimental PyTorch execution backend/runtime for obsolete and
unsupported GPUs that modern PyTorch compute stacks have dropped.

Modern PyTorch dropped your GPU? If it still exposes a usable low-level
interface, MatrixMan may still be able to run on it. The project implements a
deliberately limited set of tensor operations itself through older GPU
interfaces, keeping computationally capable hardware useful after its normal
framework/toolchain support window has closed.

The runtime is not itself OpenGL, CUDA, or OpenCL. Those are execution backends
under MatrixMan:

```text
PyTorch
   ↓
MatrixMan (PrivateUse1 / MatrixManTensor frontend)
   ├── OpenGL / GLSL fragment shaders
   │      legacy Intel / AMD / NVIDIA-class graphics
   ├── CUDA Driver API / PTX
   │      legacy NVIDIA GPUs
   └── OpenCL
          experimental / future; not implemented
   ↓
GPU
```

OpenGL and CUDA are separate implementations behind the same MatrixMan-facing
selection and tensor-dispatch layer. PrivateUse1 supplies PyTorch's device
identity and dispatch boundary; MatrixMan owns the tensor/runtime work below
that boundary.

OpenGL: “I am rendering pixels.”  
MatrixMan: “You are doing tensor math.”

This is not universal OpenGL magic. Compatibility depends on the driver,
OpenGL/GLSL version, floating-point texture support, and floating-point
framebuffer support.

## Backend Status

### OpenGL / GLSL

OpenGL is the current, most mature MatrixMan backend. It abuses GLSL fragment
shaders for tensor arithmetic: tensors are packed into floating-point textures,
the shader performs the operation, and framebuffer-backed textures hold the
results. This is deliberately not generic native OpenGL compute.

The OpenGL backend is integrated with PyTorch through `PrivateUse1`,
`MatrixManTensor`, and `__torch_dispatch__`. Its supported subset includes the
operations listed below, including Conv2D, grouped/depthwise convolution,
inference BatchNorm, activation, pooling, resize, concatenation, softmax, and
metadata-only view paths. Unsupported arithmetic fails explicitly instead of
silently moving to the CPU.

The OpenGL path has been verified on Linux and Windows. Confirmed hardware
evidence currently includes Intel GM45/GMA 4500MHD, Intel HD Graphics 4400,
and NVIDIA GT 720M-class hardware with the drivers listed in the compatibility
table. A custom Ultralytics YOLO detection checkpoint has also been run through
the PyTorch-facing path. These are validated configurations and model paths,
not a claim that every OpenGL GPU, YOLO model, or PyTorch workload is supported.

Known limitations include the narrow ATen operator subset, float32-oriented
storage, driver- and shader-dependent behavior, and conservative physical
tiling/synchronization requirements. GM45 can corrupt large one-shot
convolution renders, so the production baseline uses small tiles.

### CUDA / PTX

CUDA is a separate legacy execution backend for NVIDIA hardware. It uses the
CUDA Driver API directly through `ctypes`, embedded/hand-managed PTX, and CUDA
device allocations. It does not depend on `torch.cuda` for execution. The
validated PTX target is `sm_21`; the legacy test system is a GeForce GT 720M
(GF117M/Fermi, Compute Capability 2.1) with NVIDIA driver 390.157.

The CUDA backend is connected to MatrixMan's PyTorch-facing
`PrivateUse1`/`MatrixManTensor` dispatch layer. The frontend is shared, but the
CUDA tensor owner, kernels, transfers, and synchronization are not the
OpenGL implementation. The current CUDA scope covers a limited float32
runtime and dispatch subset, including elementwise arithmetic, 2D matmul,
selected views/splits/cat/stack, sigmoid/SiLU, softmax, nearest-neighbor
upsampling, inference BatchNorm, and a constrained NCHW Conv2D path. CUDA and
OpenGL do not have full feature parity.

Important CUDA limits remain: uploads require contiguous CPU float32 tensors;
matmul is limited to contiguous 2D float32 inputs; convolution does not support
transposed convolution or output padding and has restricted groups/layout
support; BatchNorm is inference-only; tensor-tensor division is not
implemented; and unsupported operators fail explicitly. CUDA uses the default
stream, with asynchronous kernel queueing by default and synchronous Driver
API transfers/readback. CUDA-specific runtime and operator coverage is still
smaller than the OpenGL path, and the CUDA backend should not be read as
arbitrary PyTorch or general model support.

### OpenCL

OpenCL is a planned/experimental backend only. `drivers/matrixman/backends/opencl/`
is currently a placeholder, the OpenCL probe is not implemented, and no
OpenCL fallback path should be inferred from the directory. OpenGL remains the
only backend with the project's broadest validated model evidence.

## Implemented Operation Subset

MatrixMan is experimental and intentionally strict. It implements only the
operations needed by its current examples and model experiments, including:

- Conv2D, grouped convolution, and depthwise convolution
- inference-mode BatchNorm
- in-place SiLU (`aten.silu_.default`)
- implemented elementwise add, subtract, multiply, divide, sigmoid, and scalar add paths
- selected concatenation and stack paths
- MaxPool and nearest-neighbor upsampling paths
- the traced Softmax path
- selected view, reshape, transpose, squeeze, unsqueeze, flatten, expand, split, and broadcast metadata paths
- explicit GPU-to-CPU readback

Unsupported operations fail explicitly. They do not quietly run tensor
arithmetic on the CPU.

## Current Status

The working PyTorch-facing MatrixMan frontend can select CUDA when its legacy
driver is available and otherwise falls back to OpenGL. OpenGL remains the
most mature and broadly validated backend; CUDA is a separate, lower-level
implementation with a connected but smaller PrivateUse1 operator surface.
The following are verified
hardware results, not a claim that all GPUs from these vendors or families are
supported:

| Hardware / driver | OpenGL / GLSL | MatrixMan result | YOLO 320x320 evidence |
| --- | --- | --- | --- |
| Intel GM45 / GMA 4500MHD, Mesa | 2.1 / 1.20 | Compatibility and YOLO path verified; conservative tiling required | historically about 5.4–5.7 s forward |
| Intel HD Graphics 4400, Mesa i915 | 4.6 compatibility profile / 4.60 | Compatibility suite PASS; 512x512 one-shot Conv PASS; tiling not required by that test | 2.210 s forward, 2.623 s total, about 0.38 FPS |
| NVIDIA GT 720M / NVD7, Mesa nouveau | 4.3 compatibility profile / 4.30 | Compatibility suite PASS; 512x512 one-shot Conv PASS; tiling not required by that test | 1.519 s forward, 1.975 s total, about 0.51 FPS |

The GM45 can corrupt large one-shot convolution renders. MatrixMan therefore
keeps physically small convolution tiles as its production baseline. Newer
tested GPUs tolerate larger one-shot renders in the compatibility test, but
that does not change the conservative defaults.

MatrixMan feeds the GM45 smaller tiles. It likes tiny bites.

Run the compatibility probe on a new machine before treating its result as
evidence for a model workload:

```bash
python3 -m drivers.matrixman.compatibility
```

### Experimental / Unverified

Other Intel, NVIDIA, AMD/Radeon, Apple, and accelerator configurations remain
unverified unless listed above. OpenGL support alone does not guarantee
MatrixMan compatibility: driver behavior, floating-point framebuffer support,
shader behavior, texture limits, and the supported ATen operator subset all
matter.

## Check Compatibility

Run the capability and numerical probe:

```bash
python3 -m drivers.matrixman.compatibility
```

Or:

```bash
python3 -m drivers.matrixman --check
```

The probe reports the renderer, OpenGL and GLSL versions, relevant extensions,
hardware limits, numerical self-tests, and whether large convolution tiling is
required. A known GM45-style result looks like:

```text
Float texture upload/readback  PASS
Elementwise shader             PASS
Conv2D                         PASS
BatchNorm                      PASS
SiLU_                          PASS

512x512 production tiled       PASS
512x512 one-shot               FAIL

convolution tiling required: yes
MatrixMan status: COMPATIBLE
```

An optional fast path can fail while MatrixMan remains compatible when its
validated fallback passes.

See [Compatibility and diagnostics](drivers/matrixman/COMPATIBILITY.md) for
verified hardware, diagnostic modes, and the GM45 findings behind these
defaults.

## Architecture

MatrixMan has a small explicit backend interface. It selects CUDA first when
the legacy CUDA capability is available, then falls back to OpenGL. The
OpenGL implementation is split into modules
under `drivers/matrixman/backends/opengl/`:

```text
backend.py          backend façade and public entrypoint
runtime.py          SDL/OpenGL context and context-owned lifetime
resources.py        textures, uploads, readback resources, and caches
../tensor.py        backend-neutral MatrixManTensor wrapper and metadata
tensor.py           OpenGL texture ownership and readback helpers
metadata.py         logical views, strides, offsets, and split metadata
kernels.py          shader/program lookup and shared kernel helpers
render.py           shared framebuffer and fullscreen-render plumbing
operation_context.py shared operation services
diagnostics.py      tracing and unsupported-operation reporting
profiling.py        timing and OpenGL counters
factories.py        PyTorch PrivateUse1 factory registration
dispatch.py         PyTorch/ATen dispatch bridge
convolution.py      dedicated tiled Conv2D subsystem
ops/                arithmetic, activation, normalization, pooling, resize,
                    concat, softmax, and matmul implementations
```

`implementation.py` is now only a small compatibility re-export façade. It is
not in the normal runtime execution path. The `backends/cuda/` package contains
the CUDA Driver API/PTX runtime and backend-specific operations. The
`backends/opencl/` package remains a reserved placeholder.

## Basic PyTorch Usage

Application code only needs the public package entry point:

```python
import torch
from drivers import matrixman

a_cpu = torch.randn((2, 2), dtype=torch.float32)
b_cpu = torch.randn((2, 2), dtype=torch.float32)

a = matrixman.to_device(a_cpu)
b = matrixman.to_device(b_cpu)

c = a + b                 # MatrixMan GPU tensor add
result = c.cpu()          # explicit readback to CPU

print(result)
```

The small example is also available as:

```bash
python3 demo/example/pytorch_example.py
```

## YOLO / Ultralytics Example

The minimal Ultralytics example sends a preprocessed input tensor into the
underlying PyTorch model through MatrixMan:

```python
import torch
from ultralytics import YOLO
from drivers import matrixman

model = YOLO("model.pt").model.eval()
input_mm = matrixman.to_device(input_cpu)

with torch.no_grad():
    output_mm = model(input_mm)

# Ultralytics configurations may return a tensor, tuple, list, or dict.
# The runnable example extracts the prediction tensor first, then reads back.
output_cpu = first_tensor(output_mm).cpu()
```

The runnable example accepts a local Ultralytics YOLO detection checkpoint and
an image. It does not download models automatically. A model is only expected
to work when the operations exercised by that particular model are supported:

```bash
python3 demo/example/yolo_example.py model.pt path/to/image.jpg --imgsz 320
```

The execution boundary is deliberately visible:

```text
image -> CPU preprocessing -> PyTorch tensor -> matrixman.to_device()
      -> MatrixManTensor -> Ultralytics/PyTorch forward
      -> supported tensor arithmetic through MatrixMan/OpenGL/GLSL
      -> explicit CPU readback -> CPU postprocessing
```

Python `model.forward` and its control flow still execute normally on the
CPU/PyTorch side. MatrixMan intercepts supported tensor operations involving a
`MatrixManTensor` and executes their arithmetic through OpenGL/GLSL.

MatrixMan is not specific to VisDrone. Ultralytics YOLO models can be
attempted when the ATen operations they exercise are supported; this does not
mean every YOLO or Ultralytics model is supported. The current VisDrone-style
checkpoint is tested evidence, not a general Ultralytics compatibility claim.

The backend-neutral names are `MatrixManTensor`, `to_device()`,
`is_matrixman_tensor()`, and the PyTorch device name `matrixman:0`. The
historical public names `Gm45Tensor`, `to_gm45()`, and `is_gm45_tensor()` remain
deprecated forwarding aliases. These names identify the current PyTorch
integration and do not imply that execution is restricted to GM45 hardware.

### Tested Model Evidence

The project's confirmed real-model validation uses a custom Ultralytics YOLO
detection checkpoint trained for VisDrone-style detection. The 320x320 timings
listed above are measurements from the verified OpenGL machines; the older
GM45 result is historical. VisDrone is evidence of a working trained model,
not a requirement for using MatrixMan.

## CPU Fallback Policy

CPU work is acceptable for dispatch, Python control flow, metadata, initial
uploads, explicit final readback, application preprocessing/postprocessing,
and narrowly required framework bookkeeping.

Supported tensor arithmetic must not silently move through CPU arithmetic. If
an arithmetic operation is not implemented, MatrixMan fails loudly so the
execution path is visible.

## Installation

Clone the repository:

```bash
git clone https://github.com/NETtoSan/MatrixMan.git
cd MatrixMan
```

This repository currently has no `setup.py`, `pyproject.toml`, or
`requirements.txt`. Installation is therefore manual: provide a compatible
Python environment with PyTorch and NumPy, and install Ultralytics/OpenCV for
the YOLO examples. MatrixMan also needs an SDL2 library and an OpenGL driver
that expose the required functions and floating-point framebuffer/texture
capabilities.

Run the compatibility probe before assuming a particular machine will work.

## Performance

Does it work? **YES.**

Is an Intel GMA 4500MHD a good neural-network accelerator?

**HAHAHAHAHAHAHAHAHA.**

Performance work is ongoing. The reported YOLO timings above are measurements
from the listed machines and are not benchmarks for other hardware or models.
The GM45-safe path retains synchronization and tiling overhead even on newer
tested GPUs because the production defaults must remain correct on GM45.

## Safe Defaults and Compatibility Notes

The conservative production defaults remain:

```text
MATRIXMAN_TILE_LIMIT=256
MATRIXMAN_TILE_SYNC=per_tile
```

The 256x256 physical tile limit and per-tile synchronization are required by
the validated GM45 behavior. The HD 4400 and GT 720M compatibility tests did
not require tiling for their 512x512 one-shot checks, but those results do not
promote larger renders to the general production baseline. Diagnostic tile
controls remain separate from ordinary model execution.

MatrixMan supports only a deliberately limited set of PyTorch operations and
tensor shapes. Unsupported operations fail explicitly rather than silently
performing CPU tensor arithmetic. Model compatibility therefore depends on the
actual ATen operations exercised by that model.

## Future Backends

The backend layout leaves room for future sibling implementations:

```text
drivers/matrixman/backends/
├── opengl/   verified current backend
├── cuda/     legacy Driver API/PTX backend; limited frontend integration
└── opencl/   placeholder only; not implemented
```

OpenCL support is not currently provided. CUDA has its own tensor owner,
Driver API/PTX execution path, dispatch handlers, operator implementations,
and factory registration. The current OpenGL `MatrixManTensor` storage owner
is not shared by CUDA or intended to be shared by OpenCL.

## What MatrixMan Is Not

MatrixMan is not:

- CUDA
- ROCm
- a replacement for a properly supported native PyTorch backend
- feature-complete
- currently fast
- guaranteed to work on every OpenGL GPU

If your GPU already works properly with CUDA, ROCm, MPS, or another maintained
PyTorch accelerator, please use that.

MatrixMan is for the situation where you look at an unsupported GPU and think:

> “But technically it can multiply numbers.”

## Philosophy

```text
Modern ML frameworks:
    This GPU is unsupported.

MatrixMan:
    It has shaders.

Modern ML frameworks:
    There is no supported compute runtime.

MatrixMan:
    It has shaders.

Modern ML frameworks:
    This architecture is from 2008.

MatrixMan:
    IT. HAS. SHADERS.
```

> **PyTorch says your GPU isn't supported?**  
> **Does it draw triangles? Give it to MatrixMan.**

That last line is a project slogan, not a compatibility guarantee. Whether a
GPU actually works is what the compatibility checker is for.
