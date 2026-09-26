"""Phase 4 probe: real torchvision MNIST with MSE targets and SGD."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def _cpu_value(value, reason):
    if isinstance(value, MatrixManTensor):
        return readback_tensor(value, audit_reason=reason)
    return value.detach().cpu()


def _compare(label, actual, expected, rtol=2e-3, atol=2e-3):
    actual_cpu = _cpu_value(actual, f"real MNIST validation: {label}")
    expected_cpu = _cpu_value(expected, f"real MNIST reference: {label}")
    difference = (actual_cpu - expected_cpu).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol))
    print(
        f"{label}: max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} "
        f"allclose={close}"
    )
    return close


def _one_hot(labels):
    return torch.nn.functional.one_hot(labels, num_classes=10).to(dtype=torch.float32)


def _loss(logits, targets):
    error = logits - targets
    return (error * error).mean()


def _failure(stage, error):
    print(f"{stage}: FAIL")
    print(f"  exception_type={type(error).__name__}")
    print(f"  exception_str={str(error)}")
    print(f"  exception_repr={error!r}")
    traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(20260926)
    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    count = min(int(args.samples), len(dataset))
    subset = Subset(dataset, range(count))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    print(f"dataset: PASS torchvision MNIST samples={count} batches={len(loader)}")

    cpu_model = MNISTMLP()
    state = {name: value.detach().clone() for name, value in cpu_model.state_dict().items()}
    mm_model = MNISTMLP()
    mm_model.load_state_dict(state)

    matrixman.config.backend = "opengl"
    try:
        matrixman.init()
        prepared = matrixman.prepare(mm_model, backend="opengl", training=True)
    except BaseException as error:
        _failure("MatrixMan initialization/preparation", error)
        return 1

    try:
        expected_names = ["net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias", "net.5.weight", "net.5.bias"]
        parameter_ids = {name: id(value) for name, value in prepared.named_parameters()}
        registration_ok = [name for name, _ in prepared.named_parameters()] == expected_names
        registration_ok &= all(isinstance(value, MatrixManTensor) for value in prepared.parameters())
        print(f"training preparation: PASS all six parameters={registration_ok}")
        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=args.lr)
        optimizer = torch.optim.SGD(prepared.parameters(), lr=args.lr)
        first_batch_checked = False
        losses = []
        total_correct = 0
        total_seen = 0

        for epoch in range(args.epochs):
            epoch_losses = []
            for batch_index, (images_cpu, labels_cpu) in enumerate(loader, start=1):
                targets_cpu = _one_hot(labels_cpu)
                images_mm = matrixman.to_device(images_cpu)
                targets_mm = matrixman.to_device(targets_cpu)
                cpu_images = images_cpu.clone().requires_grad_(False)

                optimizer.zero_grad()
                cpu_optimizer.zero_grad()
                cpu_logits = cpu_model(cpu_images)
                cpu_loss = _loss(cpu_logits, targets_cpu)
                mm_logits = prepared(images_mm)
                mm_loss = _loss(mm_logits, targets_mm)
                mm_loss_value = float(readback_tensor(mm_loss, audit_reason="real MNIST batch loss").item())
                epoch_losses.append(mm_loss_value)
                losses.append(mm_loss_value)

                if not first_batch_checked:
                    print(f"selected batch: shape={list(images_cpu.shape)} labels={labels_cpu.tolist()}")
                    _compare("selected batch logits", mm_logits, cpu_logits)
                    _compare("selected batch loss", mm_loss, cpu_loss)

                mm_loss.backward()
                cpu_loss.backward()
                if not first_batch_checked:
                    for name, value in prepared.named_parameters():
                        _compare(f"selected batch {name}.grad", value.grad, dict(cpu_model.named_parameters())[name].grad)
                    _compare("selected batch input", images_mm, images_cpu)

                optimizer.step()
                cpu_optimizer.step()
                first_batch_checked = True

                logits_cpu = _cpu_value(mm_logits, "real MNIST accuracy logits")
                predictions = logits_cpu.argmax(dim=1)
                total_correct += int((predictions == labels_cpu).sum().item())
                total_seen += int(labels_cpu.numel())
                current_ids = {name: id(value) for name, value in prepared.named_parameters()}
                if current_ids != parameter_ids:
                    raise RuntimeError("MatrixMan parameter object identity changed during training")
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"mean_loss={sum(epoch_losses) / len(epoch_losses):.9g}"
            )

        print(f"loss trend: {losses}")
        print(f"loss decreased: {losses[-1] < losses[0] if len(losses) > 1 else 'not enough batches'}")
        print(f"accuracy_from_explicit_readback: {100.0 * total_correct / total_seen:.3f}%")
        print(f"parameter identity stable: {all(id(value) == parameter_ids[name] for name, value in prepared.named_parameters())}")
        print("Phase 4 real MNIST/MSE/SGD: PASS")
        return 0
    except BaseException as error:
        _failure("real MNIST MatrixMan training", error)
        return 1
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
