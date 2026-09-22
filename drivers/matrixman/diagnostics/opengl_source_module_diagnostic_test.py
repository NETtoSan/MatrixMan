"""Hardware-independent checks for source-module diagnostic context."""

from __future__ import annotations

import torch
from torch import nn

from drivers.matrixman import preparation, source_context


class _Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.seen = None

    def forward(self, value):
        self.seen = source_context.current()
        return self.conv(value)


def main() -> int:
    first = _Probe().eval()
    second = _Probe().eval()
    model = nn.Sequential(first, second).eval()
    preparation._install_source_module_hooks(model)
    model(torch.randn(1, 1, 2, 2))

    assert first.seen is not None
    assert second.seen is not None
    assert first.seen.module_path == "0"
    assert second.seen.module_path == "1"
    assert first.seen.qualified_name.endswith("._Probe")
    assert second.seen.qualified_name.endswith("._Probe")
    assert source_context.current() is None

    print("OpenGL source-module diagnostic tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
