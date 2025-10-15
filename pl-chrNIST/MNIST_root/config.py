from dataclasses import dataclass
from pathlib import Path

# Default train config values
@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 1337            # must be an int
    num_workers: int = 2
    out_dir: str = "outputs"
    device: str = "auto"        # "cpu", "cuda", "auto"
    amp: bool = True            # mixed precision
    weight_decay: float = 0.0   # 0 by default, found some documentation saying this is a hyperparameter

@dataclass
class EvalConfig:
    weights: str
    batch_size: int = 256
    num_workers: int = 2
    device: str = "auto"        