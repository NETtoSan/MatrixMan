"""Device-scoped OpenGL synchronization policy.

This module describes synchronization decisions only.  The initial policy
implementation deliberately maps every mode to the existing safe behavior;
future device validation can populate the auto branch without changing the
call sites again.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...config import config


_CATEGORIES = (
    "per_tile_sync",
    "pre_consolidation_sync",
    "post_consolidation_sync",
    "scratch_reuse_sync",
    "activation_reuse_sync",
    "scratch_eviction_sync",
    "activation_eviction_sync",
)


@dataclass(frozen=True)
class SyncPolicy:
    requested: str
    resolved: str
    device_key: str
    decisions: dict[str, str]

    def decision(self, category: str) -> str:
        return self.decisions.get(category, "finish")

    def should_sync(self, category: str) -> bool:
        return self.decision(category) == "finish"


_current = SyncPolicy(
    requested="safe",
    resolved="safe",
    device_key="uninitialized",
    decisions={category: "finish" for category in _CATEGORIES},
)


def _text(value) -> str:
    return str(value or "unknown").strip()


def _device_key(info: dict[str, str]) -> str:
    fields = ("vendor", "renderer", "opengl", "glsl")
    return "opengl-sync-v1|" + "|".join(f"{field}={_text(info.get(field))}" for field in fields)


def _safe_decisions() -> dict[str, str]:
    decisions = {category: "finish" for category in _CATEGORIES}
    # Preserve the existing tileSync/skip-pre-consolidation behavior exactly.
    if config.tileSync == "per_tile" and config.skipPreConsolidationSync:
        decisions["pre_consolidation_sync"] = "skipped_after_per_tile"
    elif config.tileSync == "none" or config.tileSync == "flush":
        decisions["pre_consolidation_sync"] = "skipped_by_tile_sync"
    return decisions


def resolve(device_info: dict[str, str]) -> SyncPolicy:
    """Resolve once per OpenGL context; auto has no validated data yet."""
    global _current
    requested = config.syncPolicy
    # No device has a validated relaxed decision table yet.  Explicit relaxed
    # remains visible as the selected policy, but uses the safe table until a
    # future probe validates weaker choices.  Auto falls back to safe.
    decisions = _safe_decisions()
    if requested == "relaxed":
        decisions["post_consolidation_sync"] = "deferred"
    _current = SyncPolicy(
        requested=requested,
        resolved=requested if requested in {"safe", "relaxed"} else "safe",
        device_key=_device_key(device_info),
        decisions=decisions,
    )
    return _current


def current() -> SyncPolicy:
    return _current


def reset() -> None:
    global _current
    _current = SyncPolicy(
        requested="safe", resolved="safe", device_key="uninitialized",
        decisions={category: "finish" for category in _CATEGORIES},
    )


def diagnostics() -> dict[str, object]:
    policy = current()
    return {
        "requested": policy.requested,
        "resolved": policy.resolved,
        "device_key": policy.device_key,
        "decisions": dict(policy.decisions),
    }
