"""Static and optional live checks for packed NLL loss and int64 labels."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.matrixman.backends.opengl.ops import nll_loss


def _static_checks():
    params = (3, 4, 0, (4, 1), 8, 8, 0, 1, 8, 8, 1, 1, -100, "loss")
    source = nll_loss._shader_source(params).decode("ascii")
    assert "target_at" in source
    assert "selected_loss" in source
    assert "-read_packed(input_tex" in source
    assert "valid_count" in source
    backward = nll_loss._backward_shader_source(
        (3, 4, 0, 1, 8, 8, 0, 8, 8, 0, 1, 1, 8, 1, -100)
    ).decode("ascii")
    assert "target_at" in backward
    assert "class_index != target_index" in backward
    assert "scale / total_weight" in backward
    print("OpenGL NLL/int64 static checks: PASS")


def _live_checks():
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.init()
    try:
        logits_cpu = torch.tensor([[1.25, -0.5, 0.25, 2.0], [-1.0, 0.75, 1.5, -0.25], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        labels_cpu = torch.tensor([3, 2, 0], dtype=torch.int64)
        logits = matrixman.to_device(logits_cpu)
        labels = matrixman.to_device(labels_cpu)
        assert labels.dtype == torch.int64
        assert torch.equal(labels.cpu(), labels_cpu)
        for reduction in ("none", "sum", "mean"):
            actual = F.nll_loss(logits, labels, reduction=reduction)
            expected = F.nll_loss(logits_cpu, labels_cpu, reduction=reduction)
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
        print("OpenGL NLL/int64 live checks: PASS")
    finally:
        matrixman.shutdown()


def main():
    _static_checks()
    if "--gpu" in sys.argv:
        _live_checks()


if __name__ == "__main__":
    main()
