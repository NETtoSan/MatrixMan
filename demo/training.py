import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from drivers import matrixman
from drivers.matrixman.tensor import MatrixManTensor, readback_tensor


def render_bouncing_ball(width, step):
    """Return a small left-right-left progress animation for terminal output."""
    width = max(1, int(width))
    if width == 1:
        return "[o]"
    cycle = list(range(width)) + list(range(width - 2, 0, -1))
    position = cycle[int(step) % len(cycle)]
    return "[" + " " * position + "o" + " " * (width - position - 1) + "]"


_last_status_length = 0


def show_train_status(epoch, total_epochs, batch, total_batches, loss, accuracy=None):
    """Render one live training line when stdout is an interactive terminal."""
    global _last_status_length
    if not sys.stdout.isatty():
        return

    line = (
        f"Epoch {epoch}/{total_epochs} {render_bouncing_ball(20, batch - 1)} "
        f"batch {batch}/{total_batches} loss={loss:.4f}"
    )
    if accuracy is not None:
        line += f" acc={accuracy:.2f}%"
    padding = max(0, _last_status_length - len(line))
    sys.stdout.write("\r" + line + " " * padding)
    sys.stdout.flush()
    _last_status_length = len(line)


class MNISTMLP(nn.Module):
    def __init__(self, image_shape=(1, 28, 28), num_classes=10):
        super().__init__()

        channels, height, width = (int(value) for value in image_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * height * width, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MNISTCNN(nn.Module):
    """Deeper MNIST classifier with spatial dimensions derived at runtime."""

    def __init__(self, image_shape=(1, 28, 28), num_classes=10):
        super().__init__()

        channels, height, width = (int(value) for value in image_shape)
        if channels <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"image_shape must contain positive dimensions, got {image_shape}")

        self.image_shape = (channels, height, width)
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


parser = argparse.ArgumentParser(description="Train the MatrixMan MNIST model")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--train-samples", type=int, default=None)
parser.add_argument("--test-samples", type=int, default=None)
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset if args.train_samples is None else torch.utils.data.Subset(
        train_dataset, range(min(args.train_samples, len(train_dataset)))
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset if args.test_samples is None else torch.utils.data.Subset(
        test_dataset, range(min(args.test_samples, len(test_dataset)))
    ),
    batch_size=256,
    shuffle=False,
)

image_shape = tuple(int(value) for value in train_dataset[0][0].shape)
# model = MNISTCNN(image_shape=image_shape, num_classes=10)

model = MNISTMLP()

matrixman.config.backend = "opengl"
matrixman.init()
model = matrixman.prepare(model, backend="opengl", training=True)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
)

epochs = args.epochs


def _cpu_value(value, reason):
    if isinstance(value, MatrixManTensor):
        return readback_tensor(value, audit_reason=reason)
    return value.detach().cpu()


def _cpu_state_dict(module):
    """Read GPU-authoritative parameters for an explicit CPU serialization boundary."""
    return {
        name: _cpu_value(value, f"training serialization: {name}").clone()
        for name, value in module.state_dict().items()
    }

try:
    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        total_batches = len(train_loader)
        for batch_index, (images_cpu, labels_cpu) in enumerate(train_loader, start=1):
            images = matrixman.to_device(images_cpu)
            labels = matrixman.to_device(labels_cpu)
            if labels.dtype != torch.int64:
                raise RuntimeError(f"MatrixMan labels lost int64 dtype: {labels.dtype}")

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_cpu = _cpu_value(loss, "training loss metric")
            logits_cpu = _cpu_value(outputs, "training logits metric")
            running_loss += float(loss_cpu.item())
            running_correct += int((logits_cpu.argmax(dim=1) == labels_cpu).sum().item())
            running_total += int(labels_cpu.numel())
            show_train_status(
                epoch + 1,
                epochs,
                batch_index,
                total_batches,
                running_loss / batch_index,
                100.0 * running_correct / running_total if running_total else None,
            )

        if sys.stdout.isatty():
            print()

        avg_loss = running_loss / len(train_loader)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images_cpu, labels_cpu in test_loader:
                images = matrixman.to_device(images_cpu)
                labels = matrixman.to_device(labels_cpu)
                if labels.dtype != torch.int64:
                    raise RuntimeError(f"MatrixMan labels lost int64 dtype: {labels.dtype}")

                outputs = model(images)

                predictions = _cpu_value(outputs, "evaluation logits metric").argmax(dim=1)

                total += int(labels_cpu.numel())
                correct += int((predictions == labels_cpu).sum().item())

        accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"loss={avg_loss:.4f} "
            f"accuracy={accuracy:.2f}%"
        )

    torch.save(_cpu_state_dict(model), "./demo/models/mnist_cnn.pth")
finally:
    matrixman.shutdown()

print("Saved mnist_cnn.pth")
