import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

# HELPER MODULE

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
