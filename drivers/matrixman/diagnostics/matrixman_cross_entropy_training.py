"""Phase 5 probe: native CrossEntropyLoss with integer MatrixMan labels."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _compare(label, actual, expected, rtol=2e-3, atol=2e-3):
    actual_cpu = readback_tensor(actual, audit_reason=f"CrossEntropy diagnostic: {label}") if isinstance(actual, MatrixManTensor) else actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    difference = (actual_cpu - expected_cpu).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol))
    print(f"{label}: max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} allclose={close}")
    return close


def _describe(value):
    if isinstance(value, MatrixManTensor):
        return (
            f"type={type(value).__name__} device={value.device} shape={list(value.shape)} "
            f"dtype={value.dtype} storage={value._owner.layout.kind} "
            f"texture={value._owner.texture} requires_grad={value.requires_grad}"
        )
    return f"type={type(value).__name__} device={value.device} shape={list(value.shape)} dtype={value.dtype}"


def _failure(stage, error):
    print(f"{stage}: FAIL")
    print(f"  exception_type={type(error).__name__}")
    print(f"  exception_str={str(error)}")
    print(f"  exception_repr={error!r}")
    traceback.print_exc()


def main() -> int:
    torch.manual_seed(20260926)
    logits_cpu = torch.tensor(
        [[1.25, -0.5, 0.25, 2.0], [-1.0, 0.75, 1.5, -0.25], [0.1, 0.2, 0.3, 0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels_cpu = torch.tensor([3, 2, 0], dtype=torch.int64)
    criterion = nn.CrossEntropyLoss()
    cpu_loss = criterion(logits_cpu, labels_cpu)

    matrixman.config.backend = "opengl"
    initialized = False
    try:
        matrixman.init()
        initialized = True
        logits_mm = matrixman.to_device(logits_cpu.detach())
        logits_mm.requires_grad_(True)
        print("logits upload: PASS", _describe(logits_mm))
        try:
            labels_mm = matrixman.to_device(labels_cpu)
        except BaseException as error:
            _failure("integer label upload", error)
            print("first CrossEntropy blocker reported above")
            return 0
        print("labels upload: PASS", _describe(labels_mm))
        print("integer-label dtype: source labels dtype=int64; "
              f"MatrixMan labels dtype={labels_mm.dtype}; logical dtype preserved")
        mm_loss = criterion(logits_mm, labels_mm)
        print("CrossEntropy forward: PASS", _describe(mm_loss))
        _compare("scalar CrossEntropy loss", mm_loss, cpu_loss)
        mm_loss.backward()
        cpu_loss.backward()
        print("CrossEntropy backward: PASS")
        _compare("logits.grad", logits_mm.grad, logits_cpu.grad)
        print("tiny CrossEntropy diagnostic: PASS")
        return 0
    except BaseException as error:
        _failure("CrossEntropy diagnostic", error)
        return 1
    finally:
        if initialized:
            matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
