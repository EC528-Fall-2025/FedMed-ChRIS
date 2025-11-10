#!/usr/bin/env python
"""ChRIS-compatible Flower client that trains a local model."""

import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import flwr as fl
import numpy as np
from sklearn.datasets import make_classification

from chris_plugin import chris_plugin

__version__ = "0.1.0"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Flower NumPyClient wrapper that runs an isolated trainer.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cid", type=int, default=0, help="client id (0-indexed)")
    parser.add_argument("--total-clients", type=int, default=1, help="number of logical clients")
    parser.add_argument("--server-host", type=str, default="fedmed-fl-server", help="server host")
    parser.add_argument("--server-port", type=int, default=9091, help="server port")
    parser.add_argument("--learning-rate", type=float, default=0.2, help="gradient descent step size")
    parser.add_argument("--epochs", type=int, default=10, help="local epochs per round")
    parser.add_argument("--seed", type=int, default=13, help="seed for synthetic data")
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="client_metrics.json",
        help="filename (inside /outgoing) to store the latest training metrics",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="fedmed-fl-client 0.1.0",
    )
    return parser


parser = build_parser()


@dataclass(frozen=True)
class ClientDataset:
    x_train: np.ndarray
    y_train: np.ndarray


def _build_synthetic_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    features, labels = make_classification(
        n_samples=400,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=seed,
    )
    return features.astype(np.float32), labels.astype(np.float32).reshape(-1, 1)


def load_client_dataset(client_id: int, total_clients: int, seed: int) -> ClientDataset:
    if client_id < 0 or client_id >= total_clients:
        raise ValueError(f"client id must be within [0, {total_clients - 1}]")
    features, labels = _build_synthetic_dataset(seed)
    partitions = np.array_split(np.arange(len(features)), total_clients)
    indices = partitions[client_id]
    return ClientDataset(x_train=features[indices], y_train=labels[indices])


class LocalLogisticRegression:
    def __init__(self, n_features: int, learning_rate: float, epochs: int) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros((n_features, 1), dtype=np.float32)
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.predict_logits(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.float32)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        probs = self.predict_proba(x)
        loss = float(np.mean(-y * np.log(probs + 1e-8) - (1 - y) * np.log(1 - probs + 1e-8)))
        accuracy = float(np.mean(self.predict(x) == y))
        return loss, accuracy

    def fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        for _ in range(self.epochs):
            preds = self.predict_proba(x)
            error = preds - y
            grad_w = (x.T @ error) / len(x)
            grad_b = float(np.mean(error))
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self.evaluate(x, y)

    def get_parameters(self) -> list[np.ndarray]:
        return [self.weights.copy(), np.array([self.bias], dtype=np.float32)]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        self.weights = parameters[0].copy()
        self.bias = float(parameters[1][0])


class FedMedClient(fl.client.NumPyClient):
    """Wrap the local trainer so that Flower can coordinate our runs."""

    def __init__(self, dataset: ClientDataset, trainer: LocalLogisticRegression) -> None:
        self.dataset = dataset
        self.trainer = trainer
        self.history: list[Dict[str, float]] = []

    def latest_metrics(self) -> Dict[str, float]:
        return self.history[-1] if self.history else {}

    # Flower NumPyClient interface -------------------------------------------------
    def get_parameters(self, config: Dict | None = None):  # type: ignore[override]
        return self.trainer.get_parameters()

    def fit(self, parameters, config):  # type: ignore[override]
        self.trainer.set_parameters(parameters)
        loss, accuracy = self.trainer.fit(self.dataset.x_train, self.dataset.y_train)
        metrics = {"loss": loss, "accuracy": accuracy}
        self.history.append(metrics)
        print(
            f"[fedmed-fl-client] finished local training "
            f"(loss={loss:.4f}, accuracy={accuracy:.3f})",
            flush=True,
        )
        return self.trainer.get_parameters(), len(self.dataset.x_train), metrics

    def evaluate(self, parameters, config):  # type: ignore[override]
        self.trainer.set_parameters(parameters)
        loss, accuracy = self.trainer.evaluate(self.dataset.x_train, self.dataset.y_train)
        return loss, len(self.dataset.x_train), {"accuracy": accuracy}


def run_client(
    server_address: str,
    client_id: int,
    total_clients: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> Dict[str, float]:
    dataset = load_client_dataset(client_id, total_clients, seed)
    trainer = LocalLogisticRegression(
        n_features=dataset.x_train.shape[1],
        learning_rate=learning_rate,
        epochs=epochs,
    )
    client = FedMedClient(dataset, trainer)
    print(
        f"[fedmed-fl-client:{client_id}] connecting to {server_address} "
        f"with {len(dataset.x_train)} samples",
        flush=True,
    )
    fl.client.start_client(server_address=server_address, client=client.to_client())
    return client.latest_metrics()


@chris_plugin(
    parser=parser,
    title="FedMed Flower Client",
    category="Federated Learning",
    min_memory_limit="200Mi",
    min_cpu_limit="500m",
)
def main(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    del inputdir  # network-based plugin; no file inputs yet
    if options.total_clients <= 0:
        raise ValueError("total-clients must be >= 1")
    if options.cid >= options.total_clients:
        raise ValueError("cid must be less than total-clients")

    address = f"{options.server_host}:{options.server_port}"
    metrics = run_client(
        address,
        client_id=options.cid,
        total_clients=options.total_clients,
        learning_rate=options.learning_rate,
        epochs=options.epochs,
        seed=options.seed,
    )
    summary_path = outputdir / options.metrics_file
    summary_path.write_text(json.dumps(metrics, indent=2))
    print(f"[fedmed-fl-client:{options.cid}] wrote metrics to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
