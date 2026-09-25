"""Hardware-independent tests for safe and deferred OpenGL scratch pooling."""

from __future__ import annotations

from types import SimpleNamespace

from drivers.matrixman.backends.opengl import gpumatrix as gm, resources, runtime
from drivers.matrixman.backends.opengl.storage import StorageLayout


def _fake_runtime():
    return SimpleNamespace(
        scratch_texture_pool={},
        scratch_texture_records={},
        retired_scratch_textures=[],
        retired_scratch_bytes=0,
        retired_deferred_scratch_textures=[],
        retired_deferred_scratch_bytes=0,
        activation_texture_pool={},
        activation_texture_pool_order=[],
        activation_texture_pool_bytes=0,
        activation_texture_pool_peak_count=0,
        activation_texture_pool_peak_bytes=0,
        retired_activation_textures=[],
        retired_activation_bytes=0,
        completion_generation=0,
    )


def _owner(texture: int, width: int, height: int):
    return SimpleNamespace(
        texture=texture,
        layout=StorageLayout("packed_rgba", width, height, width * height * 4),
    )


def main() -> int:
    original_runtime_required = runtime.runtime_required
    original_is_active = runtime.is_active
    original_create = resources.create_rgba32f_texture
    original_delete = resources.gm.glDeleteTextures
    original_sync_finish = resources.profiling.sync_finish
    original_enabled = resources.profiling.enabled
    original_counters = resources.profiling.counters
    original_max = runtime._MAX_SCRATCH_TEXTURES
    original_scratch_pool = resources.config.scratchPool
    fake = _fake_runtime()
    next_texture = [200]
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
        runtime._MAX_SCRATCH_TEXTURES = 1
        resources.create_rgba32f_texture = fake_create
        resources.gm.glDeleteTextures = fake_delete
        resources.profiling.enabled = True
        resources.profiling.counters.clear()

        def fake_sync_finish(reason):
            sync_reasons.append(reason)
            fake.completion_generation += 1
            resources.promote_retired_activation_textures(fake)
            resources.promote_retired_scratch_textures(fake)

        resources.profiling.sync_finish = fake_sync_finish

        # Safe mode retains the current scratch_reuse synchronization.
        resources.config.scratchPool = "safe"
        safe_a = resources.acquire_scratch_texture(4, 4)
        resources.release_scratch_texture(_owner(safe_a, 4, 4))
        safe_b = resources.acquire_scratch_texture(4, 4)
        assert safe_b == safe_a
        assert "scratch_reuse" in sync_reasons
        resources.release_scratch_texture(_owner(safe_b, 4, 4))

        # Deferred mode retires A and allocates B before a later completion.
        resources.config.scratchPool = "deferred"
        fake.completion_generation = 10
        deferred_a = resources.acquire_scratch_texture(4, 4)
        resources.release_scratch_texture(_owner(deferred_a, 4, 4))
        deferred_b = resources.acquire_scratch_texture(4, 4)
        assert deferred_b != deferred_a
        assert sync_reasons.count("scratch_reuse") == 1

        # The completion advances the shared generation and promotes A.
        fake.completion_generation = 11
        resources.promote_retired_scratch_textures(fake)
        promoted = resources.acquire_scratch_texture(4, 4)
        assert promoted == deferred_a
        resources.release_scratch_texture(_owner(promoted, 4, 4))

        # A completion before retirement does not make a new retirement safe.
        fake.completion_generation = 12
        same_generation = resources.acquire_scratch_texture(4, 4)
        assert same_generation == deferred_a
        resources.release_scratch_texture(_owner(same_generation, 4, 4))
        resources.promote_retired_scratch_textures(fake)
        assert fake.retired_deferred_scratch_textures
        fake.completion_generation = 13
        resources.promote_retired_scratch_textures(fake)
        assert not fake.retired_deferred_scratch_textures

        # Different generations and incompatible keys remain independent.
        first = resources.acquire_scratch_texture(4, 4)
        resources.release_scratch_texture(_owner(first, 4, 4))
        fake.completion_generation = 14
        second = resources.acquire_scratch_texture(8, 4)
        assert second != first
        resources.release_scratch_texture(_owner(second, 8, 4))
        resources.promote_retired_scratch_textures(fake)
        assert first in fake.scratch_texture_pool[(4, 4)]
        assert any(item[1] == second for item in fake.retired_deferred_scratch_textures)
        fake.completion_generation = 15
        resources.promote_retired_scratch_textures(fake)
        assert second in fake.scratch_texture_pool[(8, 4)]
        assert int(resources.profiling.counters["scratch_pool_promoted"]) >= 3

        # Shared completion state promotes activation and scratch together.
        fake.completion_generation = 15
        fake.retired_activation_textures.append((
            ("opengl", id(fake), gm.GL_TEXTURE_2D, gm.GL_RGBA32F, "float32", "packed_rgba", 4, 4),
            999, 14, 256,
        ))
        fake.retired_activation_bytes = 256
        fake.retired_deferred_scratch_textures.append(((4, 4), 998, 14, 256))
        fake.retired_deferred_scratch_bytes = 256
        resources.promote_retired_activation_textures(fake)
        resources.promote_retired_scratch_textures(fake)
        assert 999 in fake.activation_texture_pool_order[-1]
        assert 998 in fake.scratch_texture_pool[(4, 4)]

        # Safe eviction still waits before deleting a pooled texture.
        resources.config.scratchPool = "safe"
        runtime._MAX_SCRATCH_TEXTURES = 1
        eviction = resources.acquire_scratch_texture(2, 2)
        resources.release_scratch_texture(_owner(eviction, 2, 2))
        other = resources.acquire_scratch_texture(3, 3)
        resources.release_scratch_texture(_owner(other, 3, 3))
        assert "scratch_eviction" in sync_reasons
        assert deleted

        assert int(resources.profiling.counters["scratch_pool_peak_retired_bytes"]) >= 256
        print("OpenGL scratch pool tests: PASS")
        print(
            f"  safe_reuses={int(resources.profiling.counters['scratch_pool_safe_reuses'])} "
            f"retired={int(resources.profiling.counters['scratch_pool_retired'])} "
            f"promoted={int(resources.profiling.counters['scratch_pool_promoted'])}"
        )
        return 0
    finally:
        runtime.runtime_required = original_runtime_required
        runtime.is_active = original_is_active
        runtime._MAX_SCRATCH_TEXTURES = original_max
        resources.create_rgba32f_texture = original_create
        resources.gm.glDeleteTextures = original_delete
        resources.profiling.sync_finish = original_sync_finish
        resources.profiling.enabled = original_enabled
        resources.profiling.counters = original_counters
        resources.config.scratchPool = original_scratch_pool


if __name__ == "__main__":
    raise SystemExit(main())
