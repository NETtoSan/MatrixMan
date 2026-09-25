import torch
import torch.nn as nn
from torchvision import datasets, transforms

from drivers import matrixman


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


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_PATH = "mnist_mlp.pth"
SAMPLE_INDEX = 1

# Keep correctness-first defaults.
#matrixman.profile = "summary"
matrixman.trace = True

matrixman.config.sync_policy = "safe"
matrixman.config.activation_pool = "safe"
matrixman.config.scratch_pool = "safe"


# ------------------------------------------------------------
# Load trained weights
# ------------------------------------------------------------

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
)

cpu_model = MNISTMLP().eval()
cpu_model.load_state_dict(state_dict)

mm_model = MNISTMLP().eval()
mm_model.load_state_dict(state_dict)


# ------------------------------------------------------------
# Load one MNIST test image
# ------------------------------------------------------------

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

image, label = test_dataset[SAMPLE_INDEX]

# Add batch dimension:
#
# [1, 28, 28]
# ->
# [1, 1, 28, 28]

x_cpu = image.unsqueeze(0)


# ------------------------------------------------------------
# CPU reference
# ------------------------------------------------------------

with torch.no_grad():
    cpu_logits = cpu_model(x_cpu)

cpu_prediction = cpu_logits.argmax(dim=1).item()


# ------------------------------------------------------------
# MatrixMan inference
# ------------------------------------------------------------

matrixman.prepare(mm_model)

x_mm = matrixman.to_device(x_cpu)

with torch.no_grad():
    mm_logits = mm_model(x_mm)

print("\nMatrixMan tensor:")
print(mm_logits)

# Explicit final readback.
mm_logits_cpu = mm_logits.cpu()

mm_prediction = mm_logits_cpu.argmax(dim=1).item()


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

diff = (cpu_logits - mm_logits_cpu).abs()

print("\n=== MNIST inference ===")
print("dataset label :", label)
print("CPU prediction:", cpu_prediction)
print("MM prediction :", mm_prediction)

print("\nCPU logits:")
print(cpu_logits)

print("\nMatrixMan logits:")
print(mm_logits_cpu)

print("\n=== Numerical comparison ===")
print("max_abs :", diff.max().item())
print("mean_abs:", diff.mean().item())

if cpu_prediction == mm_prediction:
    print("\nCPU / MatrixMan prediction: MATCH")
else:
    print("\nCPU / MatrixMan prediction: MISMATCH")
