#!/usr/bin/env python
"""ChRIS-compatible Flower server plugin. Designed by Ryan Smith"""

import json
import signal
import subprocess
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import flwr as fl

from chris_plugin import chris_plugin

__version__ = "0.1.0"

DISPLAY_TITLE = r"""
MNIST Flower Server
"""

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9091
DEFAULT_ROUNDS = 1
DEFAULT_EXPECTED_CLIENTS = 1
DEFAULT_SUMMARY = "server_summary.json"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Flower-based coordinator that waits for external trainers.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="listening port")
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help="number of FL rounds"
    )
    parser.add_argument(
        "--expected-clients",
        type=int,
        default=DEFAULT_EXPECTED_CLIENTS,
        help="how many clients must connect before each round can start",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=DEFAULT_SUMMARY,
        help="filename (inside /outgoing) to store the training summary",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="fedmed-fl-server 0.1.0",
    )
    return parser


parser = build_parser()


def _serialize_round_metrics(records: Iterable[Tuple[int, float]]) -> list[dict[str, float]]:
    return [{"round": int(rnd), "value": float(val)} for rnd, val in records]


def serialize_history(history: Any | None) -> Dict[str, Any]:
    """Convert Flower's History object into JSON-serialisable primitives."""
    if history is None:
        return {}

    summary: Dict[str, Any] = {}
    if history.losses_distributed:
        summary["losses_distributed"] = _serialize_round_metrics(history.losses_distributed)
    if history.losses_centralized:
        summary["losses_centralized"] = _serialize_round_metrics(history.losses_centralized)
    if history.metrics_distributed:
        summary["metrics_distributed"] = {
            name: _serialize_round_metrics(values) for name, values in history.metrics_distributed.items()
        }
    if history.metrics_centralized:
        summary["metrics_centralized"] = {
            name: _serialize_round_metrics(values) for name, values in history.metrics_centralized.items()
        }
    return summary


def _discover_ipv4_addresses() -> list[str]:
    """Attempt to list IPv4 addresses that clients could need to target."""
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True).strip()
        ips = sorted({token for token in output.split() if token})
        return ips
    except Exception:
        return []


def run_server(address: str, rounds: int, expected_clients: int) -> Dict[str, Any]:
    """Start the Flower server and return the training history."""
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=expected_clients,
        min_fit_clients=expected_clients,
        min_evaluate_clients=expected_clients,
    )
    print(
        f"[fedmed-fl-server] waiting for {expected_clients} client(s) at {address} "
        f"for {rounds} FL round(s)",
        flush=True,
    )
    reachable_ips = _discover_ipv4_addresses()
    if reachable_ips:
        print(
            "[fedmed-fl-server] reachable IPv4 addresses: "
            + ", ".join(reachable_ips),
            flush=True,
        )
    else:
        print(
            "[fedmed-fl-server] unable to auto-detect host IPs; "
            "clients must be pointed at the compute-node address manually.",
            flush=True,
        )

    history = fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
    print("[fedmed-fl-server] training finished", flush=True)
    return serialize_history(history)


def handle_signals() -> None:
    def _handle(signum, frame):  # type: ignore[override]
        print(f"\n[fedmed-fl-server] received signal {signum}, shutting down...", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


@chris_plugin(
    parser=parser,
    title="MNIST Flower Server",
    category="Federated Learning",
    min_memory_limit="200Mi",
    min_cpu_limit="500m",
)
def main(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    del inputdir  # network-based plugin no file inputs 
    handle_signals()
    address = f"{options.host}:{options.port}"
    summary = run_server(address, rounds=options.rounds, expected_clients=options.expected_clients)
    summary_path = outputdir / options.summary_file
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[fedmed-fl-server] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()