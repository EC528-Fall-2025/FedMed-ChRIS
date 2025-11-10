"""FedMed Flower ServerApp."""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .core import (
    FEATURES,
    LocalLogisticRegression,
    decode_parameters,
    encode_parameters,
    load_client_dataset,
    ModelParameters,
)

SERVER_SUMMARY_TOKEN = "[fedmed-superlink-app] SUMMARY "

app = ServerApp()


def _initial_parameters() -> ModelParameters:
    return ModelParameters(
        weights=np.zeros((FEATURES, 1), dtype=np.float32),
        bias=0.0,
    )


def _central_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    *,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> MetricRecord:
    params = decode_parameters(arrays)
    dataset = load_client_dataset(client_id=0, total_clients=1, seed=seed)
    trainer = LocalLogisticRegression(
        n_features=dataset.x_train.shape[1],
        learning_rate=learning_rate,
        epochs=epochs,
    )
    trainer.set_parameters(params)
    loss, accuracy = trainer.evaluate(dataset.x_train, dataset.y_train)
    return MetricRecord(
        {
            "server_round": server_round,
            "loss": loss,
            "accuracy": accuracy,
            "num_examples": len(dataset.x_train),
        }
    )


def _metrics_to_dict(metrics: Dict[int, MetricRecord]) -> Dict[int, Dict[str, Any]]:
    return {round_id: dict(record) for round_id, record in metrics.items()}


def _emit_summary(result, context: Context) -> Dict[str, Any]:  # type: ignore[override]
    summary = {
        "run_id": context.run_id,
        "rounds": context.run_config.get("num-server-rounds"),
        "total_clients": context.run_config.get("total-clients"),
        "train_metrics": _metrics_to_dict(result.train_metrics_clientapp),
        "evaluate_metrics": _metrics_to_dict(result.evaluate_metrics_clientapp),
        "server_metrics": _metrics_to_dict(result.evaluate_metrics_serverapp),
    }
    print(f"{SERVER_SUMMARY_TOKEN}{json.dumps(summary)}", flush=True)
    return summary


@app.main()
def main(grid: Grid, context: Context) -> None:
    """FedMed ServerApp entrypoint."""

    run_config = context.run_config
    total_clients = int(run_config.get("total-clients", 1))
    num_rounds = int(run_config.get("num-server-rounds", 1))
    local_epochs = int(run_config.get("local-epochs", 5))
    learning_rate = float(run_config.get("learning-rate", 0.2))
    data_seed = int(run_config.get("data-seed", 13))
    fraction_evaluate = float(run_config.get("fraction-evaluate", 1.0))

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=total_clients,
        min_evaluate_nodes=max(1, int(total_clients * fraction_evaluate)),
        min_available_nodes=total_clients,
    )

    params = encode_parameters(_initial_parameters())
    train_config = ConfigRecord(
        {
            "local-epochs": local_epochs,
            "learning-rate": learning_rate,
            "total-clients": total_clients,
            "data-seed": data_seed,
        }
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=params,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arrays: _central_evaluate(
            server_round,
            arrays,
            learning_rate=learning_rate,
            epochs=local_epochs,
            seed=data_seed,
        ),
    )

    _emit_summary(result, context)
