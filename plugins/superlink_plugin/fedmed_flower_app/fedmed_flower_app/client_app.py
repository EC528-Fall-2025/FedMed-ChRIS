"""FedMed Flower ClientApp."""

from __future__ import annotations

import json
from typing import Any, Dict

from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .core import (
    LocalLogisticRegression,
    decode_parameters,
    encode_parameters,
    load_client_dataset,
)

CLIENT_SUMMARY_TOKEN = "[fedmed-supernode-app] SUMMARY "

app = ClientApp()


def _get_node_config(context: Context) -> Dict[str, Any]:
    """Resolve node configuration with sensible defaults."""
    run_config = context.run_config
    node_config = context.node_config or {}
    return {
        "partition-id": int(node_config.get("partition-id", 0)),
        "num-partitions": int(
            node_config.get("num-partitions", run_config.get("total-clients", 1))
        ),
        "data-seed": int(node_config.get("data-seed", run_config.get("data-seed", 13))),
    }


def _log_metrics(kind: str, partition_id: int, metrics: Dict[str, Any]) -> None:
    payload = {"kind": kind, "partition_id": partition_id, "metrics": metrics}
    print(f"{CLIENT_SUMMARY_TOKEN}{json.dumps(payload)}", flush=True)


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Handle train instructions coming from the ServerApp."""
    node_config = _get_node_config(context)
    partition_id = node_config["partition-id"]
    num_partitions = node_config["num-partitions"]
    data_seed = node_config["data-seed"]

    config = msg.content.get("config", {})
    learning_rate = float(
        config.get("learning-rate", context.run_config.get("learning-rate", 0.2))
    )
    epochs = int(config.get("local-epochs", context.run_config.get("local-epochs", 10)))

    dataset = load_client_dataset(partition_id, num_partitions, data_seed)
    trainer = LocalLogisticRegression(
        n_features=dataset.x_train.shape[1],
        learning_rate=learning_rate,
        epochs=epochs,
    )
    trainer.set_parameters(decode_parameters(msg.content["arrays"]))

    loss, accuracy = trainer.fit(dataset.x_train, dataset.y_train)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "num-examples": len(dataset.x_train),
    }
    _log_metrics("train", partition_id, metrics)

    reply = RecordDict(
        {
            "arrays": encode_parameters(trainer.get_parameters()),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Handle evaluate instructions coming from the ServerApp."""
    node_config = _get_node_config(context)
    partition_id = node_config["partition-id"]
    num_partitions = node_config["num-partitions"]
    data_seed = node_config["data-seed"]

    dataset = load_client_dataset(partition_id, num_partitions, data_seed)
    trainer = LocalLogisticRegression(
        n_features=dataset.x_train.shape[1],
        learning_rate=context.run_config.get("learning-rate", 0.2),
        epochs=context.run_config.get("local-epochs", 10),
    )
    trainer.set_parameters(decode_parameters(msg.content["arrays"]))
    loss, accuracy = trainer.evaluate(dataset.x_train, dataset.y_train)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "num-examples": len(dataset.x_train),
    }
    _log_metrics("evaluate", partition_id, metrics)
    reply = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=reply, reply_to=msg)
