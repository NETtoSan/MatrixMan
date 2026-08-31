# OpenGL `implementation.py` architecture audit

This is an architecture-only audit of `drivers/matrixman/backends/opengl/implementation.py`. No runtime code was changed for this report.

At the time of the audit the file contains 3,752 lines and approximately 151 KB. It defines 127 top-level functions/methods and is the compatibility assembly point for nearly every OpenGL backend concern.

## Major internal sections

| Section | Approximate location | Current responsibility |
|---|---:|---|
| Backend and PrivateUse1 registration | 92–160 | `OpenGLBackend`, backend registration, device module hooks |
| Diagnostics and profiling bridge | 164–275 | tracing, kernel/error logs, unsupported-op reporting, profiling aliases |
| Runtime compatibility | 277–297 | wrappers around `runtime.py` |
| Program/shader helpers | 299–1950 | program caches, uniform caches, GLSL source generation |
| Resource compatibility and allocation | 1963–2092 | resource wrappers, shape validation, output allocation, CPU upload |
| GPU readback | 2093–2179 | synchronization, FBO readback, logical reconstruction |
| Legacy metadata | 2180–2389 | inert legacy helpers plus active split handling |
| Scalar/general utilities | 2396–2435 | scalar extraction, shape/pair helpers |
| Arithmetic execution | 2437–2604 | matrix and packed arithmetic rendering |
| Stack/fill/concat | 2605–3079 | stack, fill, concat variants and copy shaders |
| Pooling/resize | 3080–3222 | max pool and nearest-neighbor upsample |
| Factory/arange | 3223–3299 | factory validation and arange shader execution |
| BatchNorm/SiLU | 3300–3435 | normalization and activation rendering |
| Dispatch bridge | 3436–3699 | `_DispatchBridge.__torch_dispatch__` and tensor routing |
| Public tensor factories | 3700–3750 | `tensor`, `randn`, `to_gm45`, factory installation |

## Function and group inventory

### Backend and registration

Functions/classes:

- `OpenGLBackend`
- `_register_privateuse_name`
- `init`
- `shutdown`
- `_runtime_required`

`OpenGLBackend.probe()` initializes through the compatibility lifecycle. `matmul()` routes to `_render_binary("matmul", ...)`; `synchronize()` calls `gm.glFinish()`.

The registration code creates the `gm45` PrivateUse1 device module and exposes availability/device-count methods that read the implementation `_runtime` compatibility mirror.

Dependencies include `torch`, `Backend`, `runtime.py`, and the implementation runtime mirror. This group should eventually be split between `backend.py` and a PyTorch framework adapter. The lifecycle wrappers should remain temporarily for compatibility, then be deleted after callers migrate.

Moveability: mostly already separated; framework registration must be rewritten behind an explicit framework boundary.

### Diagnostics, tracing, and unsupported operations

Functions/state:

- `_env_flag`
- `_trace_enabled`
- `set_trace`
- `debug_enabled`
- `_trace`
- `_kernel_log`
- `_error_log`
- `_shape_text`
- `_summarize_dispatch_value`
- `_record_unsupported`
- `reset_unsupported_report`
- `unsupported_report`
- `_unsupported_counts`
- `_unsupported_examples`

These are called from initialization, operators, readback, and dispatch. `_summarize_dispatch_value` depends on `MatrixManTensor`, while unsupported reports depend on the dispatch argument structure. Tracing is a direct module-global side effect rather than an operation service.

Proposed destination: a small diagnostics utility, with profiling remaining in `profiling.py`. Do not put this collection into `operation_context.py`.

Moveability: direct after replacing the concrete tensor type check with a narrow summary protocol.

### Profiling bridge

Imported aliases:

- `_profile_counters`
- `_profile_conv`
- `_profile_dispatch`
- `_profile_enabled`
- `_profile_parameter_keys`
- `_profile_parameters`
- `profile_report`
- `profile_reset`

Operators and resource wrappers directly read and mutate these names. The actual containers and instrumentation live in `profiling.py`, but `implementation.py` remains the common access surface.

Proposed destination: `profiling.py`, accessed through a small explicit profiler interface or stable module-level API.

Moveability: the storage is already moved; direct references should be rewritten before operator extraction. Counter names and semantics must not change.

### Runtime compatibility

`init`, `shutdown`, and `_runtime_required` delegate to `runtime.py`. `_runtime` remains a compatibility mirror because `_TextureOwner`, registration callbacks, and old operator code still inspect it.

Proposed destination: retain only temporary wrappers in `implementation.py`.

Moveability: already extracted. The mirror and wrappers can be removed last.

### GLSL formatting and storage primitives

Important dependencies:

- `_glsl_float`
- imported `_numel`
- imported `_packed_atlas_size`
- imported `_contiguous_strides`
- imported `_max_storage_index`
- imported `_StorageLayout`

`_glsl_float` produces shader literals and is used by arithmetic, fill, arange, and other source generators. Storage functions provide packed atlas dimensions, logical strides, and bounds calculations and are used by allocation, readback, metadata, and shader generation.

Proposed destination:

- storage calculations remain in `storage.py`
- `_glsl_float` should move to a small GLSL utility or `kernels.py`

Moveability: storage calculations are already reusable; `_glsl_float` can move directly only if generated shader bytes remain identical.

### Program and uniform cache helpers

Functions:

- `_program`
- `_batchnorm_program`
- `_silu_program`
- `_packed_add_program`
- `_packed_sub_program`
- `_packed_strided_add_program`
- `_packed_scalar_div_program`
- `_packed_broadcast_mul_program`
- `_packed_sigmoid_program`
- `_scalar_add_program`
- `_stack_program`
- `_fill_program`
- `_cat_program`
- `_cat_dim0_2d_program`
- `_cat_lastdim_program`
- `_cat_dim1_3d_program`
- `_maxpool_program`
- `_upsample_nearest2d_program`
- `_arange_program`
- `_softmax_program`

These functions select runtime-owned dictionaries, compile generated GLSL with `gm.make_program`, and cache uniform locations with exact parameter tuples as keys. `_program` delegates the generic add/matmul lookup to `kernels.py`; the remaining helpers are still operator-specific.

Proposed destination:

- generic cache mechanics: `kernels.py`
- operator-specific lookup and source generation: corresponding `ops/<family>.py`

Moveability: generic lookup is movable. Each operator-specific helper must move together with its exact shader generator and uniform schema. They should not be combined into a generic kernel abstraction.

### Shader source generation

Arithmetic sources:

- `_packed_add_shader_source`
- `_packed_sub_shader_source`
- `_packed_strided_add_shader_source`
- `_packed_scalar_div_shader_source`
- `_packed_broadcast_mul_shader_source`
- `_scalar_add_shader_source`

Other sources:

- `_stack_shader_source`
- `_fill_shader_source`
- `_cat_shader_source`
- `_cat_dim0_2d_shader_source`
- `_cat_lastdim_shader_source`
- `_cat_dim1_3d_shader_source`
- `_maxpool_shader_source`
- `_upsample_nearest2d_shader_source`
- `_arange_shader_source`
- `_softmax_shader_source`
- `_silu_shader_source`
- `_batchnorm_shader_source`

Each generator constructs GLSL 1.20 source, substitutes dimensions, offsets, strides, texture sizes, and scalar literals, then returns ASCII bytes. Several generators embed operation-specific assumptions such as rank-3 layouts, DFL dimensions, 16-bin softmax, or packed RGBA addressing.

Proposed destination: the matching operator module. Shared `read_packed` GLSL snippets could eventually use a source utility, but that should be done only if it preserves exact generated source.

Moveability: source generators must be moved with their program lookup and parameter tuple. They must be tested for byte-for-byte output before and after migration.

### Resource and allocation compatibility

Functions:

- `_validate_supported_shape`
- `_create_rgba32f_texture`
- `_acquire_scratch_texture`
- `_release_scratch_texture`
- `_upload_array_to_texture`
- `_upload_raw_packed_array`
- `_parameter_cache_key`
- `_cached_parameter_texture`
- `_new_empty_packed_texture`
- `_new_zero_element_placeholder`
- `_empty_gm45`
- `_new_empty_matrix_texture`
- `_texture_from_cpu`
- `_validate_cpu_input`

The first seven resource functions delegate to `resources.py`. `_new_empty_packed_texture` still combines supported-shape validation, packed atlas sizing, texture allocation, `_TextureOwner` construction, and tracing. `_empty_gm45` is a PyTorch factory implementation. `_new_empty_matrix_texture` is matrix/matmul-specific. CPU upload combines validation, resource upload, and tracing.

Proposed destination:

- raw allocation/upload/cache: `resources.py`
- owner and wrapper creation: `tensor.py`
- shape validation: `metadata.py` or a small shape utility
- PrivateUse1 factory registration: a future PyTorch adapter
- matrix-specific allocation: `ops/matmul.py`

Moveability: `_new_empty_packed_texture` must be rewritten into resource allocation plus tensor wrapping before multiple operators move. This is the largest shared allocation bottleneck.

### GPU readback

Function: `_read_texture`.

It calls `_runtime_required`, performs `glFinish`, handles matrix and packed texture readback, binds the shared FBO for packed data, uses `glReadPixels`, reconstructs logical offsets/strides, creates CPU tensors, and updates readback profiling counters.

Dependencies:

- runtime FBO
- `gpu_stress.read_texture`
- `gm.glFinish`, `glReadPixels`, and framebuffer attachment
- `_TextureOwner`
- storage bounds helpers
- profiling/tracing/kernel logging
- PyTorch CPU tensor construction

Caller: `_DispatchBridge.__torch_dispatch__` for `_to_copy`.

Proposed destination: split raw transfer into `resources.py` and logical CPU reconstruction into `tensor.py` or a tensor readback adapter.

Moveability: must be rewritten first; moving the whole function would reproduce the monolith.

### Metadata and views

Functions:

- `_legacy_is_contiguous_logical`
- `_legacy_require_contiguous_logical`
- `_legacy_normalize_shape`
- `_legacy_metadata_view`
- `_legacy_metadata_transpose`
- `_legacy_metadata_unsqueeze`
- `_legacy_metadata_squeeze`
- `_legacy_metadata_expand`
- `_metadata_split`
- `_squeeze_shape`
- `_unsqueeze_shape`

The active view helpers are routed through `metadata.py`; the `_legacy_*` bodies remain inert compatibility remnants. `_metadata_split` remains active and handles supported batch-1 NCHW and DFL-shaped cases, including storage offsets and alias construction.

Dependencies include `MatrixManTensor`, `_TextureOwner.layout`, storage strides, `_numel`, shape validation, and tracing. Split also contains model-trace-specific compatibility cases.

Proposed destination: `metadata.py`; split should move after its dispatch-specific supported cases are isolated.

Moveability: view/reshape/transpose/expand/squeeze are movable. Split requires a compatibility test and a narrower shape policy first.

### Scalar and shape utilities

Functions:

- `_is_scalar_operand`
- `_scalar_value`
- `_as_pair`

These are consumed by arithmetic, fill, pooling, and dispatch. `_scalar_value` is a shared PyTorch scalar conversion helper. `_as_pair` is pooling argument normalization.

Proposed destination: `_scalar_value` and `_is_scalar_operand` in `operation_context.py` or a small PyTorch utility; `_as_pair` in `ops/pooling.py`.

Moveability: direct after preserving exact accepted scalar types and errors.

### Arithmetic execution

Functions:

- `_render_binary`
- `_render_scalar_add`
- `_render_packed_add`
- `_render_packed_sub`
- `_render_packed_strided_add`
- `_render_packed_scalar_div`
- `_render_packed_broadcast_mul`

These create outputs, validate packed layouts/strides, select arithmetic programs, attach the shared FBO, bind textures and uniforms, draw fullscreen quads, check errors, and construct `MatrixManTensor` outputs.

Dependencies:

- arithmetic shader generators and caches
- `_new_empty_packed_texture` and `_new_empty_matrix_texture`
- `_runtime_required`
- `_is_scalar_operand`, `_scalar_value`
- metadata contiguity checks
- `render.py`
- `MatrixManTensor._from_owner`
- direct `gm` state and profiling/tracing

Caller: `_DispatchBridge.__torch_dispatch__`; `_render_binary` is also called by `OpenGLBackend.matmul`.

Destination: `ops/arithmetic.py`.

Moveability: rewrite first around the operation context. No arithmetic implementation should move until output allocation, scalar utilities, profiling, and shader helper dependencies are independent.

### Stack, fill, and concat

Functions:

- `_render_stack`
- `_render_fill_scalar`
- `_cat_dim1_3d_program`
- `_cat_dim1_3d_shader_source`
- `_render_cat_dim1_3d`
- `_render_cat`
- `_render_cat_dim0_2d`
- `_render_cat_lastdim_3d`

Related program/source helpers are the stack, fill, and concat helpers listed above. These paths use logical offsets/strides, branch-generating GLSL, multiple sampler uniforms, packed output allocation, and direct GL binding.

Destination: `ops/concat.py`.

Moveability: move as one family after allocation and metadata contracts are stable. Do not move branch-generating shader code into generic infrastructure.

### Pooling

Functions:

- `_maxpool_program`
- `_maxpool_shader_source`
- `_render_max_pool2d_with_indices`
- `_as_pair`

The path implements the existing supported max-pool subset and returns a CPU index placeholder. It depends on packed contiguous tensors, allocation, cache lookup, render plumbing, and exact placeholder semantics.

Destination: `ops/pooling.py`.

Moveability: medium; direct after allocation/context extraction and tests for values plus indices placeholder.

### Resize

Functions:

- `_upsample_nearest2d_program`
- `_upsample_nearest2d_shader_source`
- `_render_upsample_nearest2d`

Destination: `ops/resize.py`.

Moveability: one of the easiest families after the operation context is complete. It has one input sampler and a limited parameter schema.

### Factory/arange

Functions:

- `_validate_factory_options`
- `_arange_length`
- `_arange_program`
- `_arange_shader_source`
- `_render_arange`
- `_arange_default_gm45`
- `_arange_start_gm45`
- `_arange_start_step_gm45`

Destination: a future small factories module or PyTorch framework adapter, not arithmetic. PrivateUse1 registration is separately framework-specific.

Moveability: shader execution can move after allocation/context stabilization; factory registration should move last.

### Normalization

Functions:

- `_batchnorm_program`
- `_batchnorm_shader_source`
- `_render_batch_norm`

The implementation uploads five parameter textures, constructs a parameterized shader, binds input/weight/bias/mean/variance samplers, renders, and returns a tensor.

Destination: `ops/normalization.py`.

Moveability: medium-high. Parameter upload/cache behavior and profiling must be made explicit first. No BatchNorm folding is implied.

### Activation

Functions:

- `_silu_program`
- `_silu_shader_source`
- `_render_silu_inplace`
- `_packed_sigmoid_program`
- `_packed_sigmoid_shader_source`
- `_render_packed_sigmoid`

Destination: `ops/activation.py`.

Moveability: sigmoid is relatively self-contained. SiLU must preserve in-place owner/shape/storage-offset mutation and should move only after the tensor service exposes that contract.

### Dispatch bridge

Class: `_DispatchBridge`.

Methods:

- `__new__`
- `__init__`
- `_from_owner`
- `__torch_dispatch__`
- `__repr__`

`__torch_dispatch__` routes arithmetic, Conv2D, BatchNorm, SiLU, split, concat, stack, fill, pooling, resize, softmax, readback, view/reshape, and matmul. It also owns unsupported-op reporting and many compact kernel logs.

Dependencies: every operator, metadata, readback, convolution, tensor class, scalar helpers, profiling decorator, tracing, and `torch.ops.aten` symbols.

Destination: `dispatch.py`.

Moveability: last. It should eventually contain routing only and no OpenGL calls or allocation logic.

### Public tensor factories

Functions:

- `tensor`
- `randn`
- `to_gm45`
- `is_matrixman_tensor`
- `install_tensor_method`
- `_install_privateuse1_factory_kernels`

Destination: `tensor.py` for tensor-facing API and a future PyTorch adapter for factory registration.

Moveability: medium after resource upload and wrapper construction are stable. Keep compatibility re-exports during migration.

## Dependency graph

```text
storage.py
  ├── packed atlas/layout/stride/bounds calculations
  └── GLSL indexing assumptions

runtime.py
  ├── OpenGL context and shared FBO
  ├── program/uniform cache containers
  ├── scratch texture pool
  └── persistent parameter cache

resources.py
  └── texture creation, upload, scratch pool, parameter cache operations

tensor.py
  ├── _TextureOwner
  └── MatrixManTensor wrapper and owner construction

metadata.py
  └── logical shape/stride/storage-offset aliases

kernels.py
  └── generic cached add/matmul program lookup

render.py
  └── shared FBO attachment, viewport, fullscreen quad

profiling.py
  └── counters, dispatch timing, parameter and GL instrumentation

operation_context.py
  ├── assembles the above services
  └── still lazily calls implementation.py for output allocation and shape validation

implementation.py
  ├── operator-specific shader sources and caches
  ├── operator execution
  ├── readback
  ├── factory behavior
  └── dispatch bridge

dispatch bridge
  ├── arithmetic / activation / normalization / pooling / resize
  ├── concat / stack / softmax / matmul
  ├── metadata and readback
  └── convolution.py
```

Operational dependency chains:

```text
operator execution
  -> operator shader source generator
  -> GLSL formatting + storage indexing
  -> runtime program/uniform cache
  -> operation context
  -> resource output allocation
  -> MatrixManTensor construction
  -> render/FBO helpers
  -> profiling/tracing

readback
  -> runtime FBO
  -> GL synchronization and transfer
  -> storage offset/stride reconstruction
  -> CPU tensor construction
  -> profiling

dispatch
  -> all operator chains above
  -> metadata
  -> readback
  -> convolution
```

## Shared bottlenecks affecting multiple families

1. `_new_empty_packed_texture` combines validation, layout, resource allocation, ownership, and tracing.
2. `_runtime_required` is referenced by nearly every program and render path.
3. `_glsl_float` and storage helpers affect many shader generators.
4. Profiling/tracing globals are read directly by resource and operator code.
5. Program and uniform dictionaries are runtime-owned, but selection and insertion logic is still operator-local.
6. `MatrixManTensor._from_owner` is the common return path for almost every operator.
7. Framebuffer checks, texture binding, uniform setup, and `glGetError` handling are repeated inside operator bodies.
8. The dispatch bridge names every private implementation helper and prevents deletion until routing is migrated.
9. Lazy callbacks from `operation_context.py`, `metadata.py`, `resources.py`, and `tensor.py` preserve compatibility but keep the monolith reachable.

## Safest rewrite/extraction order

### Stage 1: stabilize low-level pure services

First rewrite/extract:

- explicit GLSL literal formatting (`_glsl_float`)
- explicit scalar conversion/validation
- explicit supported-shape validation
- explicit profiler/tracer access

Destination: `kernels.py`, `metadata.py`, `profiling.py`, or small utilities. Do not expand `operation_context.py` beyond delegation.

Risk: medium.

### Stage 2: split output allocation

Separate `_new_empty_packed_texture` into:

```text
storage.py       layout calculation
resources.py     RGBA32F allocation
../tensor.py     MatrixManTensor wrapper and owner construction
tensor.py        _TextureOwner and OpenGL readback
operation_context.py  narrow assembly function
```

Preserve all layout, tracing, and allocation counters.

Risk: medium-high because every operator uses it.

### Stage 3: stabilize operation context

Remove its direct dependency on implementation for allocation and validation. Expose only stable services, not operator-specific shader functions.

Risk: medium-high.

### Stage 4: migrate one small operator family

Recommended first family: nearest-neighbor resize or sigmoid.

Reason: each has a small parameter surface and limited sampler/uniform behavior.

Risk: medium.

### Stage 5: migrate arithmetic shader/program helpers and execution

Move each arithmetic shader generator with its program lookup and then move execution bodies. Preserve exact source bytes, cache tuple keys, uniform order, and error behavior.

Risk: high.

### Stage 6: migrate pooling, resize, and softmax

Move each as a complete operator family, including compatibility restrictions and placeholder behavior.

Risk: medium-high.

### Stage 7: migrate activation and normalization

Move sigmoid/SiLU and BatchNorm after tensor mutation and parameter-cache services are explicit.

Risk: high.

### Stage 8: migrate concat and stack

Move all branch-generating packed-copy variants together.

Risk: high.

### Stage 9: migrate matmul

Separate matrix texture allocation from generic packed output allocation, then route `OpenGLBackend.matmul` through `ops/matmul.py`.

Risk: medium-high.

### Stage 10: migrate factories

Move arange execution and then PrivateUse1 factory registration into the framework-facing layer.

Risk: medium.

### Stage 11: extract dispatch

Move `_DispatchBridge.__torch_dispatch__` to `dispatch.py` after all operator destinations are stable.

Risk: high.

### Stage 12: remove compatibility remnants

Delete wrappers, mirrors, legacy metadata bodies, stale imports, and old private aliases only after static and runtime compatibility checks pass.

Risk: medium.

## First thing to rewrite

The first rewrite should be the output/tensor allocation service, specifically `_new_empty_packed_texture`, together with shape validation and the operation-context callback.

This is the lowest-level service used by arithmetic, activation, normalization, pooling, resize, concat, softmax, and portions of factory execution. Until it is independent, every family remains coupled to `implementation.py`.

The first independent contract should be:

```text
allocate_packed_output(shape)
    -> resource allocation + StorageLayout + _TextureOwner

tensor_from_owner(owner, shape, offset, strides)
    -> MatrixManTensor
```

The contract must preserve packed atlas dimensions, ownership registration, tracing, and profiling counters.

## Eventual deletions

After migration and compatibility verification, the following can likely be deleted:

- `_runtime` compatibility mirror
- `init`, `shutdown`, `_runtime_required` wrappers
- resource compatibility wrappers
- `_legacy_*` metadata functions
- old operator aliases and wrappers
- implementation-local program helpers
- implementation-local tensor factory registration
- `_DispatchBridge`
- unused imports of `gm`, `gpu_stress`, and storage helpers

The final `implementation.py` should ideally be empty or a very small compatibility façade containing re-exports for old private imports. It should not contain operator math, OpenGL calls, shader generation, tensor construction, or dispatch routing.

## Observation-only notes

### PERFORMANCE

- Program and uniform caching already exists and is context-owned.
- Operator paths still repeat FBO attachment, texture binding, uniform setup, framebuffer checks, and error checks.
- No source-only observation proves these are bottlenecks; existing profiling counters should be used before optimization.
- Readback always synchronizes with `glFinish`, which is correctness-sensitive and not a refactor target.

### COMPATIBILITY

- Supported operator subsets are intentionally narrow: packed layouts, batch-1 shapes, selected pooling/resize cases, and DFL-oriented softmax.
- `permute` remains unsupported.
- The max-pool indices result is a CPU placeholder behavior that must be preserved.
- PrivateUse1 factory registration, device naming, and public tensor helpers are caller-visible compatibility surfaces.

### ARCHITECTURE

- `implementation.py` is primarily a dependency hub.
- `operation_context.py` must remain a small assembler and must not absorb shader generation or operator implementations.
- The strongest coupling is through direct global names, private cache fields, and dispatch references.
- Lazy callbacks are useful transitional seams but should be removed progressively rather than replicated.

### GM45-SPECIFIC

- GLSL 1.20, immediate-mode fullscreen rendering, RGBA32F packing, and shared FBO behavior are embedded in many shader/render paths.
- Conv2D remains correctly isolated in `convolution.py` with the validated 256-tile and `per_tile` synchronization baseline.
- No decomposition stage should alter `glFinish()` placement, tile geometry, consolidation, or cache lifetime.

## Recommended final ownership

```text
backend.py
  public backend façade

dispatch.py
  ATen routing only

operation_context.py
  small shared service façade only

ops/*
  operator-specific validation, shader generation, cache lookup, uniforms, execution

convolution.py
  dedicated tiled Conv2D subsystem

tensor.py
  MatrixManTensor and texture ownership

metadata.py
  logical views, strides, and storage offsets

resources.py
  texture/resource manipulation and parameter/scratch caches

runtime.py
  OpenGL context and context-owned lifetime

kernels.py
  generic program-cache mechanics

render.py
  generic FBO/fullscreen render mechanics

profiling.py
  profiling instrumentation and counters
```
