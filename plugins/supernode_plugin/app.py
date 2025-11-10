#!/usr/bin/env python
"""Flower SuperNode-backed FedMed plugin."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List

from chris_plugin import chris_plugin

__version__ = "0.2.0"

SUMMARY_TOKEN = "[fedmed-supernode-app] SUMMARY "
DEFAULT_SUPERLINK_PORT = 9092
DEFAULT_CLIENTAPP_PORT = 9094
DEFAULT_STATE_DIR = Path("/tmp/fedmed-flwr-node")
DEFAULT_METRICS_FILE = "client_metrics.json"
IMAGE_TAG = f"docker.io/fedmed/fl-supernode:{__version__}"
REPO_URL = "https://github.com/EC528-Fall-2025/FedMed-ChRIS"

Process = subprocess.Popen
CHILDREN: List[Process] = []


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run the FedMed Flower SuperNode inside ChRIS.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cid", type=int, default=0, help="client id (partition id)")
    parser.add_argument("--total-clients", type=int, default=1, help="logical clients")
    parser.add_argument("--superlink-host", default="fedmed-fl-superlink", help="SuperLink host/IP")
    parser.add_argument("--superlink-port", type=int, default=DEFAULT_SUPERLINK_PORT, help="SuperLink Fleet API port")
    parser.add_argument("--clientapp-host", default="0.0.0.0", help="ClientAppIo bind host")
    parser.add_argument("--clientapp-port", type=int, default=DEFAULT_CLIENTAPP_PORT, help="ClientAppIo bind port")
    parser.add_argument("--data-seed", type=int, default=13, help="seed for synthetic data partitioning")
    parser.add_argument("--metrics-file", default=DEFAULT_METRICS_FILE, help="filename to store metrics")
    parser.add_argument("--state-dir", type=str, default=str(DEFAULT_STATE_DIR), help="directory used as FLWR_HOME")
    parser.add_argument("--keep-state", action="store_true", help="keep Flower cache instead of deleting it")
    parser.add_argument(
        "--transport",
        choices=["grpc-rere", "grpc-adapter", "rest"],
        default="grpc-rere",
        help="transport used to connect to the SuperLink",
    )
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"fedmed-fl-supernode {__version__}",
    )
    return parser


parser = build_parser()


def _register_child(proc: Process) -> None:
    CHILDREN.append(proc)


def _terminate_process(proc: Process, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _cleanup_children() -> None:
    for proc in reversed(CHILDREN):
        try:
            _terminate_process(proc)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[fedmed-fl-supernode] failed to terminate {proc.args}: {exc}", flush=True)
    CHILDREN.clear()


def handle_signals() -> None:
    def _handle(signum, _frame):  # type: ignore[override]
        print(f"\n[fedmed-fl-supernode] received signal {signum}, shutting down...", flush=True)
        _cleanup_children()
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def _transport_flag(transport: str) -> List[str]:
    if transport == "grpc-rere":
        return ["--grpc-rere"]
    if transport == "grpc-adapter":
        return ["--grpc-adapter"]
    return ["--rest"]


def _stream_lines(stream, prefix: str, hook=None) -> None:  # type: ignore[override]
    if stream is None:
        return
    for raw in iter(stream.readline, ""):
        line = raw.rstrip()
        print(f"[{prefix}] {line}", flush=True)
        if hook:
            hook(line)
    stream.close()


def _run_supernode(options: Namespace, env: dict[str, str]) -> Dict[str, Any]:
    superlink = f"{options.superlink_host}:{options.superlink_port}"
    clientapp_addr = f"{options.clientapp_host}:{options.clientapp_port}"
    node_config = (
        f"partition-id={options.cid} "
        f"num-partitions={options.total_clients} "
        f"data-seed={options.data_seed}"
    )
    cmd: List[str] = [
        "flower-supernode",
        "--insecure",
        *_transport_flag(options.transport),
        f"--superlink={superlink}",
        f"--clientappio-api-address={clientapp_addr}",
        "--node-config",
        node_config,
    ]
    print(f"[fedmed-fl-supernode] starting SuperNode: {' '.join(cmd)}", flush=True)

    metrics: Dict[str, Any] | None = None

    def _capture(line: str) -> None:
        nonlocal metrics
        if SUMMARY_TOKEN in line:
            payload = line.split(SUMMARY_TOKEN, maxsplit=1)[1].strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                print(f"[fedmed-fl-supernode] failed to parse metrics: {exc}", flush=True)
                return
            if parsed.get("kind") == "train":
                metrics = parsed

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    _register_child(proc)
    stdout_thread = threading.Thread(
        target=_stream_lines, args=(proc.stdout, "supernode", _capture), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_stream_lines, args=(proc.stderr, "supernode"), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    exit_code = proc.wait()
    stdout_thread.join()
    stderr_thread.join()

    if exit_code != 0:
        raise RuntimeError(f"flower-supernode exited with {exit_code}")
    if metrics is None:
        metrics = {
            "kind": "train",
            "partition_id": options.cid,
            "metrics": {},
            "message": "No metrics emitted by the ClientApp.",
        }
    return metrics


@chris_plugin(
    parser=parser,
    title="FedMed Flower SuperNode",
    category="Federated Learning",
    min_memory_limit="200Mi",
    min_cpu_limit="500m",
)
def _plugin_main(options: Namespace, inputdir: Path, outputdir: Path) -> None:
    del inputdir
    handle_signals()

    if getattr(options, "json", False):
        emit_plugin_json()
        return

    if options.total_clients <= 0:
        raise ValueError("total-clients must be >= 1")
    if options.cid < 0 or options.cid >= options.total_clients:
        raise ValueError("cid must be within [0, total-clients)")

    env = os.environ.copy()
    state_root = Path(options.state_dir).expanduser()
    flwr_home = (state_root / f"cid-{options.cid}")
    flwr_home.mkdir(parents=True, exist_ok=True)
    env["FLWR_HOME"] = str(flwr_home)

    summary = None
    try:
        summary = _run_supernode(options, env)
    finally:
        _cleanup_children()

    metrics_path = outputdir / options.metrics_file
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"[fedmed-fl-supernode:{options.cid}] wrote metrics to {metrics_path}", flush=True)

    if not options.keep_state:
        shutil.rmtree(flwr_home, ignore_errors=True)
        print(f"[fedmed-fl-supernode:{options.cid}] cleaned {flwr_home}", flush=True)


def main(*args, **kwargs):
    if not args and not kwargs and "--json" in sys.argv:
        emit_plugin_json()
        return
    return _plugin_main(*args, **kwargs)


def emit_plugin_json() -> None:
    from chris_plugin.tool import chris_plugin_info

    argv = [
        "chris_plugin_info",
        "--dock-image",
        IMAGE_TAG,
        "--name",
        "fl-supernode",
        "--public-repo",
        REPO_URL,
    ]
    prev = sys.argv
    try:
        sys.argv = argv
        chris_plugin_info.main()
    finally:
        sys.argv = prev


if __name__ == "__main__":
    main()
