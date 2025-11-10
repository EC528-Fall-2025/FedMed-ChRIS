"""Shared FedMed Flower App utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from flwr.app import Array, ArrayRecord

FEATURES = 4
SAMPLES = 400
TRUE_WEIGHTS = np.array([[0.8], [-1.1], [1.4], [0.3]], dtype=np.float32)
TRUE_BIAS = -0.25


@dataclass(frozen=True)
class ClientDataset:
    """Simple container for a client's dataset."""

    x_train: np.ndarray
    y_train: np.ndarray


@dataclass(frozen=True)
class ModelParameters:
    """Serializable representation of model parameters."""

    weights: np.ndarray
    bias: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _build_synthetic_dataset(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic binary classification dataset."""
    rng = np.random.default_rng(seed)
    features = rng.normal(loc=0.0, scale=1.0, size=(SAMPLES, FEATURES)).astype(np.float32)
    logits = features @ TRUE_WEIGHTS + TRUE_BIAS
    logits += rng.normal(loc=0.0, scale=0.35, size=logits.shape).astype(np.float32)
    probabilities = _sigmoid(logits)
    labels = (rng.random(size=probabilities.shape) < probabilities).astype(np.float32)
    return features, labels


def load_client_dataset(client_id: int, total_clients: int, seed: int) -> ClientDataset:
    """Load a deterministic dataset partition."""
    if total_clients <= 0:
        raise ValueError("total_clients must be positive")
    if client_id < 0 or client_id >= total_clients:
        raise ValueError(f"client_id must be within [0, {total_clients - 1}]")

    features, labels = _build_synthetic_dataset(seed)
    partitions = np.array_split(np.arange(len(features)), total_clients)
    indices = partitions[client_id]
    return ClientDataset(x_train=features[indices], y_train=labels[indices])


class LocalLogisticRegression:
    """Tiny logistic regression trainer."""

    def __init__(self, n_features: int, learning_rate: float, epochs: int) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros((n_features, 1), dtype=np.float32)
        self.bias = 0.0

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logits(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.float32)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        probs = self.predict_proba(x)
        eps = 1e-8
        loss = float(np.mean(-y * np.log(probs + eps) - (1 - y) * np.log(1 - probs + eps)))
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

    def get_parameters(self) -> ModelParameters:
        return ModelParameters(weights=self.weights.copy(), bias=float(self.bias))

    def set_parameters(self, parameters: ModelParameters) -> None:
        self.weights = parameters.weights.copy()
        self.bias = float(parameters.bias)


def encode_parameters(parameters: ModelParameters) -> ArrayRecord:
    """Convert model parameters into an ArrayRecord."""
    record = ArrayRecord()
    record["weights"] = Array.from_numpy_ndarray(parameters.weights.astype(np.float32))
    record["bias"] = Array.from_numpy_ndarray(
        np.array([[parameters.bias]], dtype=np.float32)
    )
    return record


def decode_parameters(record: ArrayRecord) -> ModelParameters:
    """Convert an ArrayRecord into model parameters."""
    weights = record["weights"].numpy().astype(np.float32)
    bias = float(record["bias"].numpy().reshape(-1)[0])
    return ModelParameters(weights=weights, bias=bias)
