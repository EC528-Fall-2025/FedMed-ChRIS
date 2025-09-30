from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from utils import get_device, set_seed, accuracy, save_checkpoint

# Values in the config are set by calling main.py with CL arguments (specified by ChRIS documentation)
class Trainer:
    def __init__(self, model: nn.Module, lr: float, weight_decay: float,
                 device: torch.device, amp: bool, out_dir: str):
        self.device = device
        self.model  = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = GradScaler(enabled=amp)
        self.amp = amp
        self.best_acc = 0.0
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_loader, val_loader, epochs: int) -> Dict[str, list]:
        history = {"train_loss": [], "validation_loss": [], "validation_accuracy": []}
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader, epoch, epochs)
            val_loss, val_acc = self.evaluate(val_loader)
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(val_loss)
            history["validation_accuracy"].append(val_acc)

            # Only save the best ckpt based on the val.
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                save_checkpoint(self.model, self.out_dir / "best.ckpt")
        return history

    def _train_one_epoch(self, loader, epoch: int, epochs: int) -> float:
        self.model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return running / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with autocast(enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return total_loss / total, accuracy(correct, total)

# High-level training API
def train_model(config, loaders, model_cls):
    set_seed(config.seed)
    device = get_device(config.device)
    model = model_cls()
    trainer = Trainer(model, config.lr, config.weight_decay, device, config.amp, config.out_dir)
    hist = trainer.train(*loaders, epochs=config.epochs)
    return hist, trainer.out_dir / "best.ckpt"
