"""Hardware-independent tests for the OpenGL activation texture pool."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import torch

from drivers.matrixman.backends.opengl import gpumatrix as gm, resources, runtime
from drivers.matrixman.backends.opengl.storage import StorageLayout
from drivers.matrixman.tensor import MatrixManTensor


def _fake_runtime():
    return SimpleNamespace(
        activation_texture_pool={},
        activation_texture_pool_order=[],
        activation_texture_pool_bytes=0,
        retired_activation_textures=[],
        retired_activation_bytes=0,
        completion_generation=0,
        activation_texture_pool_peak_count=0,
        activation_texture_pool_peak_bytes=0,
    )


def main() -> int:
    original_runtime_required = runtime.runtime_required
    original_is_active = runtime.is_active
    original_create = resources.create_rgba32f_texture
    original_delete = resources.gm.glDeleteTextures
    original_sync_finish = resources.profiling.sync_finish
    original_enabled = resources.profiling.enabled
    original_counters = resources.profiling.counters
    original_max_textures = runtime._MAX_ACTIVATION_TEXTURES
    original_activation_policy = resources.config.activationPool
    fake = _fake_runtime()
    next_texture = [100]
    deleted = []
    sync_reasons = []

    def fake_create(_width, _height, _data=None):
        texture = next_texture[0]
        next_texture[0] += 1
        return texture

    def fake_delete(_count, pointer):
        deleted.append(int(pointer._obj.value))

    try:
        runtime.runtime_required = lambda: fake
        runtime.is_active = lambda: True
        runtime._MAX_ACTIVATION_TEXTURES = 256
        resources.create_rgba32f_texture = fake_create
        resources.gm.glDeleteTextures = fake_delete
        resources.profiling.enabled = True
        resources.profiling.counters.clear()
        resources.config.activationPool = "safe"

        def fake_sync_finish(reason):
            sync_reasons.append(reason)
            fake.completion_generation += 1
            resources.promote_retired_activation_textures(fake)

        resources.profiling.sync_finish = fake_sync_finish

        shape = (1, 4, 4, 4)
        first = resources.acquire_activation_texture(shape)
        first_texture = first.texture
        resources.release_activation_texture(first)
        reused = resources.acquire_activation_texture(shape)
        assert reused.texture == first_texture
        assert "activation_reuse" in sync_reasons
        resources.release_activation_texture(reused)

        different_shape = resources.acquire_activation_texture((1, 4, 4, 5))
        assert different_shape.texture != first_texture
        resources.release_activation_texture(different_shape)

        key = resources._activation_pool_key(fake, 4, 4)
        assert key != resources._activation_pool_key(fake, 4, 4, target=gm.GL_TEXTURE_2D + 1)
        assert key != resources._activation_pool_key(fake, 4, 4, internal_format=gm.GL_RGBA32F + 1)
        assert key != resources._activation_pool_key(fake, 4, 4, dtype="float16")
        assert key != resources._activation_pool_key(fake, 4, 4, layout_kind="matrix2d_red")
        other_context = _fake_runtime()
        other_key = resources._activation_pool_key(other_context, 4, 4)
        assert key != other_key
        runtime.runtime_required = lambda: other_context
        other_owner = resources.acquire_activation_texture(shape)
        assert other_owner.texture != first_texture
        resources.release_activation_texture(other_owner)
        runtime.runtime_required = lambda: fake

        live = resources.acquire_activation_texture(shape)
        live_texture = live.texture
        view_a = MatrixManTensor._from_owner(live, shape)
        view_b = MatrixManTensor._from_owner(live, shape)
        concurrent = resources.acquire_activation_texture(shape)
        assert concurrent.texture != live_texture
        resources.release_activation_texture(concurrent)
        del view_a
        gc.collect()
        assert all(texture != live_texture for _key, texture in fake.activation_texture_pool_order)
        del view_b
        del live
        gc.collect()
        assert any(texture == live_texture for _key, texture in fake.activation_texture_pool_order)

        # A persistent owner has no pool key and must be deleted, never pooled.
        persistent = resources.owner_from_texture(999, StorageLayout("packed_rgba", 1, 1, 1))
        del persistent
        gc.collect()
        assert 999 in deleted

        resources.clear_activation_pool(fake)
        runtime._MAX_ACTIVATION_TEXTURES = 1
        # With a one-texture limit, returning a second incompatible
        # resource evicts it rather than growing the pool without bound.
        eviction_owner = resources.acquire_activation_texture((1, 2, 2, 2))
        resources.release_activation_texture(eviction_owner)
        eviction_owner = resources.acquire_activation_texture((1, 3, 3, 3))
        resources.release_activation_texture(eviction_owner)
        assert len(fake.activation_texture_pool_order) == 1
        assert int(resources.profiling.counters["activation_pool_evictions"]) == 1

        # Deferred mode never waits merely to recycle a retired activation.
        resources.clear_activation_pool(fake)
        resources.config.activationPool = "deferred"
        deferred_sync_count = int(resources.profiling.counters["activation_pool_reuse_sync_calls"])
        fake.completion_generation = 4
        deferred = resources.acquire_activation_texture(shape)
        deferred_texture = deferred.texture
        resources.release_activation_texture(deferred)
        assert fake.retired_activation_textures[0][1] == deferred_texture
        before_completion = resources.acquire_activation_texture(shape)
        assert before_completion.texture != deferred_texture
        assert int(resources.profiling.counters["activation_pool_reuse_sync_calls"]) == deferred_sync_count

        # A later completion promotes all textures retired at generation 4.
        fake.completion_generation = 5
        resources.promote_retired_activation_textures(fake)
        assert not fake.retired_activation_textures
        reused_after_completion = resources.acquire_activation_texture(shape)
        assert reused_after_completion.texture == deferred_texture
        resources.release_activation_texture(reused_after_completion)

        # A completion before retirement does not make the new retirement safe.
        fake.completion_generation = 6
        prior_completion = resources.acquire_activation_texture(shape)
        assert prior_completion.texture == deferred_texture
        resources.release_activation_texture(prior_completion)
        resources.promote_retired_activation_textures(fake)
        assert fake.retired_activation_textures
        fake.completion_generation = 7
        resources.promote_retired_activation_textures(fake)
        assert not fake.retired_activation_textures
        final_reuse = resources.acquire_activation_texture(shape)
        assert final_reuse.texture == deferred_texture
        resources.release_activation_texture(final_reuse)
        assert int(resources.profiling.counters["activation_pool_retired"]) >= 3
        assert int(resources.profiling.counters["activation_pool_promoted"]) >= 3

        # Exact pool keys remain mandatory in deferred mode as well.
        incompatible = resources.acquire_activation_texture((1, 4, 4, 5))
        assert incompatible.texture != deferred_texture
        resources.release_activation_texture(incompatible)

        before_clear = set(deleted)
        fake.completion_generation = 8
        resources.clear_activation_pool(fake)
        assert not fake.activation_texture_pool_order
        assert len(deleted) > len(before_clear)
        print("OpenGL activation pool tests: PASS")
        print(
            f"  reuses={int(resources.profiling.counters['activation_pool_reuses'])} "
            f"evictions={int(resources.profiling.counters['activation_pool_evictions'])}"
        )
        return 0
    finally:
        runtime.runtime_required = original_runtime_required
        runtime.is_active = original_is_active
        runtime._MAX_ACTIVATION_TEXTURES = original_max_textures
        resources.config.activationPool = original_activation_policy
        resources.create_rgba32f_texture = original_create
        resources.gm.glDeleteTextures = original_delete
        resources.profiling.sync_finish = original_sync_finish
        resources.profiling.enabled = original_enabled
        resources.profiling.counters = original_counters


if __name__ == "__main__":
    raise SystemExit(main())
