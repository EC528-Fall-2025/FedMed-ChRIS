#!/usr/bin/env python

import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from chris_plugin import chris_plugin

# Reuse chrNIST code
from MNIST_root.models import SimpleCNN
from MNIST_root.utils import get_device, set_seed, MNIST_MEAN_STD


__version__ = '1.0.0'

DISPLAY_TITLE = r"""
MNIST Flower Client
"""

# ------------------------------------ CLI parser -------------------------------------

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Flower MNIST client that trains SimpleCNN locally.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cid", type=int, default=0, help="client id (0-indexed)")
    parser.add_argument("--total-clients", type=int, default=1, help="number of logical clients")
    parser.add_argument("--server-host", type=str, default="fedmed-fl-server", help="server host")
    parser.add_argument("--server-port", type=int, default=9091, help="server port")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="optimizer learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=13, help="random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="compute device")
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="client_metrics.json",
        help="filename (inside /outgoing) to store the latest training metrics",
    )
    parser.add_argument("-V", "--version", action="version", version=f"mnist-fl-client {__version__}", help = "Plugin Version")
    parser.add_argument("--data-root", type=str, default="MNIST_root/data",
    help=(
        "Directory where MNIST is stored (pre-downloaded). "
        "No download is attempted; files must already exist here."), 
    )
    return parser

# Global
parser = build_parser()

# ------------------------------------ Data Load and Partition ------------------------------------

@dataclass(frozen=True)
class ClientLoaders:
    trainloader: DataLoader
    testloader: DataLoader


def _mnist_partition_loaders(
    client_id: int,
    total_clients: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    data_root: Path,
) -> ClientLoaders:
    """Create train/test loaders for a single logical client.

    Training data is partitioned across `total_clients` evenly.
    Test data is shared across all clients.
    Assumes MNIST already exists in `data_root` (MNIST_root/data).
    """
    if client_id < 0 or client_id >= total_clients:
        raise ValueError(f"client id must be within [0, {total_clients - 1}]")

    set_seed(seed)

    mean, std = MNIST_MEAN_STD
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )

    # IMPORTANT download=False since the data is already present in MNIST_root/data
    train_ds = datasets.MNIST(root=str(data_root), train=True, download=False, transform=tfm)
    test_ds = datasets.MNIST(root=str(data_root), train=False, download=False, transform=tfm)

    indices = np.arange(len(train_ds))
    partitions = np.array_split(indices, total_clients)
    client_indices = partitions[client_id]
    train_subset = Subset(train_ds, client_indices)

    trainloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ClientLoaders(trainloader=trainloader, testloader=testloader)

# ------------------------------------ Local Training & Eval ------------------------------------

def _train_one_round(
    model: torch.nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train `model` for `epochs` over `trainloader` and return avg loss per batch."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

    return running_loss / max(1, num_batches)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on `testloader`. return (avg_loss, accuracy)."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in testloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(1, len(testloader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy

# ------------------------------------ Flower to PyTorch Parameter Conversion Helpers ------------------------------------

def _get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# ------------------------------------ Flower NumPyClient ------------------------------------

class MnistClient(fl.client.NumPyClient):
    """Wrap SimpleCNN so Flower can orchestrate local training."""

    def __init__(
        self,
        model: torch.nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: int,
        lr: float,
        device: torch.device,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self._history: list[Dict[str, float]] = []

    def latest_metrics(self) -> Dict[str, float]:
        return self._history[-1] if self._history else {}

    # Flower NumPyClient interface

    def get_parameters(self, config):  # type: ignore[override]
        return _get_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore[override]
        _set_parameters(self.model, parameters)
        train_loss = _train_one_round(
            self.model,
            self.trainloader,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
        )
        val_loss, val_acc = _evaluate(self.model, self.testloader, self.device)
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
        }
        self._history.append(metrics)
        num_examples = len(self.trainloader.dataset)
        return _get_parameters(self.model), num_examples, metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        _set_parameters(self.model, parameters)
        loss, acc = _evaluate(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}

# ------------------------------------ Orchestration helper ------------------------------------

def run_client(
    server_address: str,
    client_id: int,
    total_clients: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    device_str: str,
    data_root: Path,
) -> Dict[str, float]:
    
    device = get_device(device_str)

    loaders = _mnist_partition_loaders(
        client_id=client_id,
        total_clients=total_clients,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        data_root=data_root,
    )

    model = SimpleCNN().to(device)

    client = MnistClient(
        model=model,
        trainloader=loaders.trainloader,
        testloader=loaders.testloader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
    )

    print(
        f"[mnist-fl-client:{client_id}] connecting to {server_address} "
        f"with {len(loaders.trainloader.dataset)} train samples",
        flush=True,
    )

    fl.client.start_client(server_address=server_address, client=client.to_client())

    return client.latest_metrics()


@chris_plugin(
    parser=parser,
    title='MNIST Flower Client',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='2Gi',    # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    # Network-based plugin, no file inputs yet
    del inputdir

    if options.total_clients <= 0:
        raise ValueError("total-clients must be >= 1")
    if options.cid < 0 or options.cid >= options.total_clients:
        raise ValueError("cid must be in [0, total-clients-1]")

    # Where to cache MNIST inside the container
    data_root = Path(options.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # NETWORKING
    address = f"{options.server_host}:{options.server_port}"

    # Launch
    metrics = run_client(
        server_address=address,
        client_id=options.cid,
        total_clients=options.total_clients,
        learning_rate=options.learning_rate,
        epochs=options.epochs,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        seed=options.seed,
        device_str=options.device,
        data_root=data_root,
    )

    summary_path = outputdir / options.metrics_file
    summary_path.write_text(json.dumps(metrics, indent=2))
    print(f"[mnist-fl-client:{options.cid}] wrote metrics to {summary_path}", flush=True)


if __name__ == "__main__":
    main()