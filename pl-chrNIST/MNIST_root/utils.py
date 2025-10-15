import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

MNIST_MEAN_STD: Tuple[float, float] = (0.1307, 0.3081)

# Get avaibale resource
def get_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def accuracy(correct, total):
    return 100.0 * correct / max(1, total)

def save_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)

def load_checkpoint(model: nn.Module, path: str | Path, map_location: Optional[torch.device] = None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    return model

def resolve_path(p: str | None, inputdir: Path) -> Path | None:
    if p is None:
        return None
    pth = Path(p)
    return (inputdir / pth) if not pth.is_absolute() else pth

# Taken from online forum, standard conversion.
def preprocess_image(path: Path) -> torch.Tensor:
    mean, std = MNIST_MEAN_STD
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    img = Image.open(path).convert("L")
    return tfm(img).unsqueeze(0)  # [1,1,28,28]

def _eval(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()                                # Set to eval mode
    crit = torch.nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():                       # No model updates
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                   # Softmax input logit
            loss = crit(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc