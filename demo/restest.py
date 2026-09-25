from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import torch
import torchvision.models as models
from drivers import matrixman

matrixman.prefer("opengl")
matrixman.profile = False #"summary"
matrixman.trace = False

matrixman.config.sync_policy = "safe"
matrixman.config.activation_pool = "deferred"
matrixman.config.scratch_pool = "deferred"

model = models.resnet18(weights=None).eval()
matrixman.prepare(model)

x = torch.randn(1, 3, 224, 224)
x = matrixman.to_device(x)

with torch.no_grad():
    y = model(x)

print(y.shape)
print(y.cpu())
