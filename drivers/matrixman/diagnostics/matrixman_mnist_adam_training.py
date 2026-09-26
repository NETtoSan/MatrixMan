"""Phase 7: real MNIST, integer labels, CrossEntropyLoss, and Adam."""

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


PARAMETER_NAMES = [
    "net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias",
    "net.5.weight", "net.5.bias",
]


def _cpu_value(value, reason):
    if isinstance(value, MatrixManTensor):
        return readback_tensor(value, audit_reason=reason)
    return value.detach().cpu()


def _compare(label, actual, expected, rtol=2e-3, atol=2e-3):
    actual_cpu = _cpu_value(actual, f"Adam MNIST: {label} actual")
    expected_cpu = _cpu_value(expected, f"Adam MNIST: {label} expected")
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


def _state_parameter(optimizer, parameter, key):
    state = optimizer.state.get(parameter, {})
    return state.get(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
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
        parameters = dict(prepared.named_parameters())
        registration_ok = [name for name, _ in prepared.named_parameters()] == PARAMETER_NAMES
        registration_ok &= len(parameters) == 6 and all(
            isinstance(value, MatrixManTensor) for value in parameters.values()
        )
        print(f"training preparation: PASS all six parameters registered={registration_ok}")

        cpu_optimizer = torch.optim.Adam(cpu_model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(prepared.parameters(), lr=args.lr)
        losses = []
        epoch_losses = []
        correct = 0
        seen = 0
        selected_checked = False
        all_close = bool(registration_ok)

        for epoch in range(args.epochs):
            epoch_values = []
            for images_cpu, labels_cpu in loader:
                images_mm = matrixman.to_device(images_cpu)
                labels_mm = matrixman.to_device(labels_cpu)
                if labels_mm.dtype != torch.int64:
                    raise RuntimeError(f"MatrixMan labels lost int64 dtype: {labels_mm.dtype}")
                if not selected_checked:
                    print(
                        f"selected labels: source_dtype={labels_cpu.dtype} "
                        f"matrixman_dtype={labels_mm.dtype}"
                    )

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
                loss_value = float(readback_tensor(mm_loss, audit_reason="Adam MNIST batch loss").item())
                losses.append(loss_value)
                epoch_values.append(loss_value)

                mm_loss.backward()
                cpu_loss.backward()
                if not selected_checked:
                    all_close &= _compare("selected batch logits", mm_logits, cpu_logits)
                    all_close &= _compare("selected batch CrossEntropy loss", mm_loss, cpu_loss)
                    all_close &= _compare("selected batch input gradient", mm_images.grad, cpu_images.grad)
                    cpu_parameters = dict(cpu_model.named_parameters())
                    for name in PARAMETER_NAMES:
                        all_close &= _compare(
                            f"selected batch {name}.grad",
                            parameters[name].grad,
                            cpu_parameters[name].grad,
                        )

                optimizer.step()
                cpu_optimizer.step()
                if not selected_checked:
                    cpu_parameters = dict(cpu_model.named_parameters())
                    for name in PARAMETER_NAMES:
                        all_close &= _compare(
                            f"selected batch {name} after Adam",
                            parameters[name],
                            cpu_parameters[name],
                        )
                        mm_state = optimizer.state[parameters[name]]
                        cpu_state = cpu_optimizer.state[cpu_parameters[name]]
                        for key in ("exp_avg", "exp_avg_sq"):
                            all_close &= _compare(
                                f"selected batch {name} state[{key}]",
                                mm_state[key],
                                cpu_state[key],
                            )
                        print(
                            f"state placement {name}: exp_avg={mm_state['exp_avg'].device} "
                            f"exp_avg_sq={mm_state['exp_avg_sq'].device} "
                            f"step={mm_state['step'].device}"
                        )
                selected_checked = True

                logits_cpu = _cpu_value(mm_logits, "Adam MNIST accuracy logits")
                correct += int((logits_cpu.argmax(dim=1) == labels_cpu).sum().item())
                seen += int(labels_cpu.numel())
                current_ids = {name: id(value) for name, value in prepared.named_parameters()}
                expected_ids = {name: id(value) for name, value in parameters.items()}
                if current_ids != expected_ids:
                    raise RuntimeError("MatrixMan parameter identity changed during Adam training")
            mean_loss = sum(epoch_values) / len(epoch_values)
            epoch_losses.append(mean_loss)
            print(f"epoch={epoch + 1}/{args.epochs} mean_loss={mean_loss:.9g}")

        print(f"loss trend: {losses}")
        print(f"epoch mean losses: {epoch_losses}")
        print(f"loss decreased: {losses[-1] < losses[0] if len(losses) > 1 else 'not enough batches'}")
        print(f"accuracy_from_explicit_logits_readback: {100.0 * correct / seen:.3f}%")
        stable = all(id(value) == expected_ids[name] for name, value in prepared.named_parameters())
        print(f"parameter identity stable: {stable}")
        if not all_close or not stable:
            print("Phase 7 real MNIST/CrossEntropy/Adam: FAIL (validation mismatch)")
            return 1
        print("Phase 7 real MNIST/CrossEntropy/Adam: PASS")
        return 0
    except BaseException as error:
        _failure("real MNIST CrossEntropy Adam training", error)
        return 1
    finally:
        if initialized:
            matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
