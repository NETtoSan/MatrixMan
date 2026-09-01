# MatrixMan CUDA Optimization Notes

This note records the CUDA runtime and optimization results measured on the
legacy test system. Measurements are separated into production behavior,
diagnostic-only experiments, and rejected designs.

## Test hardware

- GPU: GeForce GT 720M
- Architecture: GF117M / Fermi
- Compute Capability: 2.1
- NVIDIA driver: 390.157
- Device memory: approximately 964.6 MiB

## CUDA execution architecture

MatrixMan uses the CUDA Driver API directly through `ctypes`. It does not use
`torch.cuda`. PyTorch's `PrivateUse1` dispatch layer sits above MatrixMan and
routes supported tensor operations to the backend.

The backend loads an embedded PTX module targeting:

```text
.version 3.0
.target sm_21
.address_size 64
```

Kernel launches use the CUDA default stream. The CUDA Driver API and legacy
NVIDIA 390 JIT impose stricter PTX compatibility requirements than modern
CUDA environments.

## Allocation pool optimization

MatrixMan has an exact-size pool for temporary, non-parameter CUDA storage:

```text
free_blocks[size_bytes] -> list[CUdeviceptr]
```

Temporary allocations reuse an idle pointer of exactly the requested size.
Parameter-cache allocations remain persistent and are not placed in the
temporary pool. The pool drains and calls `cuMemFree_v2` at backend shutdown or
when required for out-of-memory recovery.

Before pooling, a representative 20-frame profile reported approximately:

```text
Alloc calls: 5244
Free calls:  5234
HtoD calls:  2957
```

After parameter caching, most allocation and upload traffic was model
parameter traffic. Exact-size temporary pooling removed repeated driver
allocation/free work for recurring activation sizes. Across hardware runs,
steady YOLO throughput improved from approximately 6.4-6.5 FPS to roughly
7.0-7.5 FPS, depending on run conditions.

Pooling does not change tensor ownership semantics: an owner must be released
before its pointer can become a free pool block, and views retain the owner.

## Async queue and deferred reclamation

The old launch behavior was:

```text
cuLaunchKernel
cuCtxSynchronize
return to Python
repeat
```

The current default behavior is:

- kernels queue asynchronously on the CUDA default stream;
- a released temporary pointer enters pending reclamation;
- pending pointers are not reused or freed while work may still reference them;
- after a successful synchronization, pending pointers return to the exact-size
  pool or are directly freed;
- final CPU readback is the principal steady-state synchronization point.

The compatibility switch restores synchronization after every kernel:

```bash
MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE=1
```

HtoD and DtoH transfers remain synchronous Driver API operations. No async
memcpy was added as part of this optimization.

## Async A/B result

In a 100-frame, 320x320 unprofiled YOLO test:

| Mode | Observed throughput |
|---|---:|
| Async queueing | generally 8.8-9.5 FPS; later frames often 9.3-9.7 FPS |
| Synchronized compatibility mode | generally 7.0-7.6 FPS |

This is roughly a 25% throughput improvement. Detection counts matched
frame-for-frame in the tested sequence.

## Synchronization profiling

A profiled 20-frame run reported:

```text
Synchronization calls: 210
total: 942.853 ms

readback:
  calls=20
  total=942.014 ms
  avg=47.101 ms
  pending_reclaimed=2936
  pending_bytes=635012540

parameter_replacement:
  calls=188
  total=0.757 ms

shutdown:
  calls=2
  total=0.082 ms
```

The 188 parameter-replacement boundaries are effectively negligible. The
meaningful queue drain occurs at final readback. The approximately 47 ms
readback synchronization time mostly represents waiting for previously queued
GPU work, not the DtoH copy itself.

The actual DtoH transfer timing was:

```text
calls=20
total=8.227 ms
avg=0.411 ms
```

## Deferred memory cost

The same profiling work showed:

```text
peak pending releases: 162
peak pending bytes:    31,830,435
peak cached bytes:     32,403,875
```

Deferred reclamation trades roughly tens of MiB of retained VRAM for better
execution overlap. This is acceptable on the measured device, but memory
pressure and OOM recovery remain important because the GT 720M has only about
964.6 MiB of device memory.

## Profiler semantics

In async mode, per-operator timings are host enqueue/submission times. They are
not GPU kernel execution times. Conv2D GMAC/s must not be derived from those
enqueue timings; async reports omit derived GMAC/s and label the timing as
host enqueue time.

True per-kernel timing requires synchronized compatibility mode:

```bash
MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE=1 MATRIXMAN_PROFILE=1 \
python3 -m drivers.matrixman.backends.cuda.conv2d
```

## Canonical c64 3x3 kernel

The production specialization is:

```text
conv2d_3x3_s1_p1_c64_plane
```

It uses one block per output-channel plane, 128 threads per block, and stages
576 weights per output channel in shared memory. Compiled resource attributes
on GF117 were:

```text
max_threads_per_block = 1024
shared_size_bytes     = 2304
num_regs              = 25
```

Representative synchronized timings were:

| Spatial size | Average kernel time |
|---|---:|
| 20x20 | approximately 2.4 ms |
| 40x40 | approximately 7.7-7.9 ms |
| 80x80 | approximately 26.7-31.8 ms, depending on run/context |

## Rejected `plane_2block` experiment

`plane_2block` was correct but diagnostic-only. It used two blocks per output
channel, duplicating the shared-weight staging for each partition.

| Spatial size | Canonical | `plane_2block` |
|---|---:|---:|
| 20x20 | 2.425 ms | 3.828 ms |
| 40x40 | 7.704 ms | 11.579 ms |
| 80x80 | 26.695 ms | 42.623 ms |

Its compiled attributes were `shared=2304` and `regs=26`. The additional block
parallelism did not compensate for duplicated weight staging and synchronization
work, so this design was rejected for production.

## Diagnostic-only `plane_256` experiment

The diagnostic kernel is:

```text
conv2d_3x3_s1_p1_c64_plane_256
```

It retains one block per output channel and one weight-staging pass, but uses
256 threads per block:

```text
spatial0 = tid + 512*k
spatial1 = spatial0 + 256
```

Its compiled attributes were `shared=2304` and `regs=25`. It matched the
canonical kernel exactly for the tested sizes:

```text
plane vs plane_256 max_abs_diff = 0
tested: 4x4, 20x20, 40x40, 80x80
```

Synchronized performance was:

| Spatial size | Canonical | `plane_256` | Difference |
|---|---:|---:|---:|
| 20x20 | 2.425 ms | 2.347 ms | approximately 3.2% faster |
| 40x40 | 7.704 ms | 7.106 ms | approximately 7.8% faster |
| 80x80 | 26.695 ms | 26.670 ms | effectively unchanged |

Despite the small-size results, `plane_256` remains diagnostic-only and must
not replace the canonical production kernel globally without broader validation.

## Current recommendation

- Keep exact-size temporary allocation pooling.
- Keep async default-stream queueing as the default execution mode.
- Keep `MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE=1` for synchronized A/B tests and
  true per-kernel profiling.
- Keep `plane_256` diagnostic-only.
- Treat `plane_2block` as a negative-result diagnostic.
- Do not continue inventing c64 thread-layout variants without new profiling
  evidence.
- Focus future work on broader backend capability, regression benchmarking, or
  clearly measured bottlenecks.

## Useful commands

Unprofiled async YOLO:

```bash
MATRIXMAN_BACKEND=cuda \
python3 demo/main-tracking.py --imgsz 320 --frames 100
```

Synchronized YOLO comparison:

```bash
MATRIXMAN_BACKEND=cuda \
MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE=1 \
python3 demo/main-tracking.py --imgsz 320 --frames 100
```

Profiled CUDA run:

```bash
MATRIXMAN_BACKEND=cuda \
MATRIXMAN_PROFILE=1 \
python3 demo/main-tracking.py --imgsz 320 --frames 20
```

Synchronized Conv2D diagnostic:

```bash
MATRIXMAN_BACKEND=cuda \
MATRIXMAN_CUDA_DISABLE_ASYNC_QUEUE=1 \
python3 -m drivers.matrixman.backends.cuda.conv2d
```

Conv2D microbenchmarks must run in synchronized mode, or explicitly synchronize
around timed kernel execution. Otherwise their timing measures host submission
overhead rather than GPU execution.
