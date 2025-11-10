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


def print_dir_tree(root: Path, label: str) -> None:
    """
    Print a full recursive listing of `root`.
    Shows directories with trailing '/' and files with byte size.
    """
    print(f"\n[{label}] root={root}")
    if not root.exists():
        print("  (does not exist)")
        return
    # print the root itself
    print("  [D] .")
    try:
        for p in sorted(root.rglob("*"), key=lambda x: str(x).lower()):
            rel = p.relative_to(root)
            if p.is_dir():
                print(f"  [D] {rel}/")
            else:
                try:
                    size = p.stat().st_size
                except Exception as e:
                    size = f"stat-error:{e}"
                print(f"  [F] {rel}  ({size} bytes)")
    except Exception as e:
        print(f"  [error] failed walking {root}: {e}")

def prep_dataset_root(outputdir: Path) -> None:
    """
    Ensure torchvision's `root="data"` resolves to a writable path.
    We move CWD to outputdir and pre-create ./data under it.
    """
    try:
        (outputdir / "data").mkdir(parents=True, exist_ok=True)
        os.chdir(outputdir)
        print(f"[setup] CWD set to {Path.cwd()} ; dataset root will be ./data")
    except Exception as e:
        print(f"[setup][warn] could not switch CWD to outputdir: {e}")

# For plugin running on miniChRIS (cannot print to terminal in during setup time)
def _is_info_probe() -> bool:
    return os.getenv("CHRIS_PLUGIN_INFO") == "1" or "--json" in sys.argv