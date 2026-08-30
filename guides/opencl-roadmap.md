# OpenCL roadmap

## Current status

OpenCL is the next planned MatrixMan backend, but it does not currently work.

```text
OpenCL probe: not implemented
```

The `drivers/matrixman/backends/opencl/` directory is a placeholder only. No
OpenCL implementation or fallback path should be inferred from its presence.

## Why OpenCL follows OpenGL

The OpenGL backend demonstrated that PyTorch tensor computation can run on
otherwise unsupported graphics hardware when programmable floating-point
rendering is available. OpenCL is a more direct compute model:

- `cl_mem` buffers represent tensor storage instead of textures/framebuffers;
- kernels execute through explicit workgroups;
- local memory can stage reusable neighborhoods and tiles;
- vector types can express valid packed operations without confusing storage
  lanes with semantic channels;
- queues and events provide explicit synchronization and profiling;
- Conv2D and matmul can be expressed as compute kernels rather than fragment
  programs.

OpenCL still depends on the actual device, driver, extensions, float support,
and runtime quality. It is not automatically available merely because OpenGL
works.

## What can be reused

The OpenCL backend should reuse the backend-neutral portions of MatrixMan:

- the PyTorch `PrivateUse1` frontend and dispatch concepts;
- tensor shape, stride, offset, and view metadata concepts;
- backend selection and compatibility reporting;
- the benchmark runner and its JSON schema;
- focused diagnostics and reference-comparison philosophy;
- the public YOLO demo and shared detection helpers;
- the invariant that unsupported tensor arithmetic fails explicitly instead of
  silently moving to CPU.

The OpenGL `Gm45Tensor` and its texture owner should not be shared as the
OpenCL tensor representation.

## What will differ

An OpenCL backend will need its own:

| Concern | OpenCL direction |
| --- | --- |
| Storage | `cl_mem` buffers with a documented scalar/vector layout. |
| Runtime | OpenCL platform/device/context/command queue ownership. |
| Programs | Kernel compilation, build diagnostics, and binary/program cache. |
| Dispatch | ND-range kernel launches and explicit event dependencies. |
| Operations | Kernels for elementwise ops, reductions, Conv2D, matmul, and required YOLO paths. |
| Transfers | Explicit buffer upload and readback. |
| Profiling | OpenCL event profiling mapped into the existing backend-neutral JSON schema. |

The first implementation should establish one validated buffer layout and a
small correctness path before attempting broad operator coverage or automatic
device-specific tuning.

## Proposed phases

1. Probe platforms, devices, queue capabilities, and required extensions.
2. Implement context lifecycle, buffer ownership, upload/readback, and a
   minimal tensor wrapper.
3. Implement one elementwise kernel and one matmul/Conv2D correctness path
   against CPU references.
4. Add PrivateUse1 dispatch for the validated operator subset.
5. Integrate event timing with `yolo_benchmark.py` and compare OpenCL and
   OpenGL using the same JSON schema.
6. Expand YOLO coverage only after repeated hardware correctness tests.

Work should begin when hardware and runtime access are available for actual
OpenCL validation. Until then, OpenGL remains the only implemented backend.
