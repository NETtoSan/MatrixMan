"""Phase 5: real MNIST, integer labels, CrossEntropyLoss, and SGD."""

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
from drivers.matrixman.diagnostics.matrixman_mnist_realdata_training import MNISTMLP
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def _cpu_value(value, reason):
    if isinstance(value, MatrixManTensor):
        return readback_tensor(value, audit_reason=reason)
    return value.detach().cpu()


def _compare(label, actual, expected, rtol=2e-3, atol=2e-3):
    actual_cpu = _cpu_value(actual, f"CrossEntropy MNIST: {label} actual")
    expected_cpu = _cpu_value(expected, f"CrossEntropy MNIST: {label} expected")
    difference = (actual_cpu - expected_cpu).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    mean_abs = float(difference.mean().item()) if difference.numel() else 0.0
    close = bool(torch.allclose(actual_cpu, expected_cpu, rtol=rtol, atol=atol))
    print(f"{label}: max_abs={max_abs:.9g} mean_abs={mean_abs:.9g} allclose={close}")
    return close


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

    torch.manual_seed(20260927)
    dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    sample_count = min(int(args.samples), len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"dataset load: PASS samples={sample_count} batches={len(loader)}")

    cpu_model = MNISTMLP()
    state = {name: value.detach().clone() for name, value in cpu_model.state_dict().items()}
    mm_model = MNISTMLP()
    mm_model.load_state_dict(state)
    criterion = nn.CrossEntropyLoss()

    matrixman.config.backend = "opengl"
    initialized = False
    try:
        matrixman.init()
        initialized = True
        prepared = matrixman.prepare(mm_model, backend="opengl", training=True)
        expected_names = [
            "net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias",
            "net.5.weight", "net.5.bias",
        ]
        parameters = dict(prepared.named_parameters())
        parameter_ids = {name: id(value) for name, value in parameters.items()}
        registration_ok = [name for name, _ in prepared.named_parameters()] == expected_names
        registration_ok &= len(parameters) == 6 and all(isinstance(value, MatrixManTensor) for value in parameters.values())
        print(f"training preparation: PASS all six parameters registered={registration_ok}")

        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=args.lr)
        optimizer = torch.optim.SGD(prepared.parameters(), lr=args.lr)
        losses = []
        correct = 0
        seen = 0
        selected_checked = False

        for epoch in range(args.epochs):
            epoch_losses = []
            for batch_index, (images_cpu, labels_cpu) in enumerate(loader, start=1):
                images_mm = matrixman.to_device(images_cpu)
                labels_mm = matrixman.to_device(labels_cpu)
                if labels_mm.dtype != torch.int64:
                    raise RuntimeError(f"MatrixMan labels lost int64 dtype: {labels_mm.dtype}")
                if not selected_checked:
                    print(f"selected labels: source_dtype={labels_cpu.dtype} matrixman_dtype={labels_mm.dtype}")

                cpu_images = images_cpu.clone().requires_grad_(not selected_checked)
                mm_images = images_mm
                if not selected_checked:
                    mm_images.requires_grad_(True)
                cpu_optimizer.zero_grad()
                optimizer.zero_grad()

                cpu_logits = cpu_model(cpu_images)
                cpu_loss = criterion(cpu_logits, labels_cpu)
                mm_logits = prepared(mm_images)
                mm_loss = criterion(mm_logits, labels_mm)
                loss_value = float(readback_tensor(mm_loss, audit_reason="CrossEntropy batch loss").item())
                losses.append(loss_value)
                epoch_losses.append(loss_value)

                mm_loss.backward()
                cpu_loss.backward()
                if not selected_checked:
                    _compare("selected batch logits", mm_logits, cpu_logits)
                    _compare("selected batch CrossEntropy loss", mm_loss, cpu_loss)
                    _compare("selected batch input gradient", mm_images.grad, cpu_images.grad)
                    for name in expected_names:
                        _compare(f"selected batch {name}.grad", parameters[name].grad, dict(cpu_model.named_parameters())[name].grad)

                optimizer.step()
                cpu_optimizer.step()
                selected_checked = True

                logits_cpu = _cpu_value(mm_logits, "CrossEntropy accuracy logits")
                correct += int((logits_cpu.argmax(dim=1) == labels_cpu).sum().item())
                seen += int(labels_cpu.numel())
                current_ids = {name: id(value) for name, value in prepared.named_parameters()}
                if current_ids != parameter_ids:
                    raise RuntimeError("MatrixMan parameter identity changed during training")
            print(f"epoch={epoch + 1}/{args.epochs} mean_loss={sum(epoch_losses) / len(epoch_losses):.9g}")

        print(f"loss trend: {losses}")
        print(f"loss decreased: {losses[-1] < losses[0] if len(losses) > 1 else 'not enough batches'}")
        print(f"accuracy_from_explicit_logits_readback: {100.0 * correct / seen:.3f}%")
        print(f"parameter identity stable: {all(id(value) == parameter_ids[name] for name, value in prepared.named_parameters())}")
        print("Phase 5 real MNIST/CrossEntropy/SGD: PASS")
        return 0
    except BaseException as error:
        _failure("real MNIST CrossEntropy training", error)
        return 1
    finally:
        if initialized:
            matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
