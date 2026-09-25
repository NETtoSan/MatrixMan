"""Hardware-independent tests for OpenGL parameter-cache policy."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import numpy as np
import torch

from drivers.matrixman.backends.opengl import resources


def main() -> int:
    runtime = resources.runtime
    original_runtime_required = runtime.runtime_required
    original_is_active = runtime.is_active
    original_upload = resources.upload_raw_packed_array
    original_enabled = resources.profiling.enabled
    original_counters = resources.profiling.counters
    fake_runtime = SimpleNamespace(parameter_cache={}, parameter_cache_current={})
    uploaded = []
    deleted = []

    class Owner:
        def __init__(self, texture: int):
            self.texture = texture

    def fake_upload(_array, _kind, **_metadata):
        owner = Owner(len(uploaded) + 1)
        uploaded.append(owner)
        return owner

    def fake_delete(_count, pointer):
        deleted.append(int(pointer._obj.value))

    try:
        runtime.runtime_required = lambda: fake_runtime
        runtime.is_active = lambda: True
        resources.upload_raw_packed_array = fake_upload
        resources.gm.glDeleteTextures = fake_delete
        resources.profiling.enabled = True
        resources.profiling.counters.clear()

        first = torch.ones((4,), dtype=torch.float32)
        same = resources.cached_parameter_texture(first, "weight")
        again = resources.cached_parameter_texture(first, "weight")
        assert same is again and len(uploaded) == 1

        distinct = torch.ones((4,), dtype=torch.float32)
        assert resources.cached_parameter_texture(distinct, "weight") is not same
        assert len(uploaded) == 2

        linear_weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        transposed_a = linear_weight.t()
        transposed_b = linear_weight.t()
        cached_transposed = resources.cached_parameter_texture(transposed_a, "gemm_mat2")
        assert resources.cached_parameter_texture(transposed_b, "gemm_mat2") is cached_transposed
        assert len(uploaded) == 3
        assert tuple(transposed_a.shape) == (3, 4)
        assert tuple(transposed_a.stride()) == (1, 3)

        first.add_(1.0)
        changed = resources.cached_parameter_texture(first, "weight")
        assert changed is not same and len(uploaded) == 4
        assert same.texture == 0 and deleted

        del distinct
        gc.collect()
        assert resources.persistent_parameter_resources() == 2
        resources.profiling.record_parameter_upload(
            np.zeros((1,), dtype=np.float32),
            "bias",
            operation="Conv2D",
            parameter_role="bias",
            cacheable=False,
        )
        groups = resources.profiling.parameter_group_snapshot()
        assert any(
            group["operation"] == "Conv2D"
            and group["parameter_role"] == "bias"
            and group["tensor_shape"] == [1]
            and not group["cacheable"]
            and group["upload_count"] == 1
            for group in groups
        )
        print("OpenGL parameter cache tests: PASS")
        print(f"  uploads={len(uploaded)} hits={int(resources.profiling.counters['parameter_cache_hits'])} invalidations={int(resources.profiling.counters['parameter_cache_invalidations'])}")
        return 0
    finally:
        runtime.runtime_required = original_runtime_required
        runtime.is_active = original_is_active
        resources.upload_raw_packed_array = original_upload
        resources.profiling.enabled = original_enabled
        resources.profiling.counters = original_counters


if __name__ == "__main__":
    raise SystemExit(main())
