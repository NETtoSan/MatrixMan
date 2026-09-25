import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_dataset,
    batch_size=128,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
)

model = MNISTMLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
)

epochs = 5

for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    total_batches = len(train_loader)
    for batch_index, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        running_total += labels.size(0)
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
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = 100.0 * correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"loss={avg_loss:.4f} "
        f"accuracy={accuracy:.2f}%"
    )

torch.save(
    model.state_dict(),
    "mnist_mlp.pth",
)

print("Saved mnist_mlp.pth")
