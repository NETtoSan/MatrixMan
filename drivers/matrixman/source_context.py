"""Context-local source-module metadata for diagnostic operation tracing."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceModuleInfo:
    qualified_name: str
    module_path: str
    preferred: bool = False
    fused: bool = False


_STACK: ContextVar[tuple[SourceModuleInfo, ...]] = ContextVar(
    "matrixman_source_module_stack", default=()
)


def push(info: SourceModuleInfo) -> None:
    _STACK.set(_STACK.get() + (info,))


def pop() -> None:
    stack = _STACK.get()
    if stack:
        _STACK.set(stack[:-1])


def clear() -> None:
    _STACK.set(())


def current() -> SourceModuleInfo | None:
    stack = _STACK.get()
    if not stack:
        return None
    preferred = [item for item in stack if item.preferred]
    return preferred[-1] if preferred else stack[-1]
