#!/usr/bin/env bash
# Run the FedMed SuperNode locally with published ports and helper paths.
#
# Usage:
#   SUPERLINK_HOST=10.0.0.6 CID=0 TOTAL_CLIENTS=3 bash scripts/run_supernode.sh

set -euo pipefail

# Override via env vars if needed
IMAGE="${IMAGE:-docker.io/fedmed/pl-supernode:0.1.0}"
# Set SUPERLINK_HOST to the SuperLink’s routable IP/DNS (required)
SUPERLINK_HOST="${SUPERLINK_HOST:-10.239.56.236}"
CID="${CID:-0}"
TOTAL_CLIENTS="${TOTAL_CLIENTS:-3}"
DATA_SEED="${DATA_SEED:-13}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/supernode_run_${CID}"
IN_DIR="${BASE_DIR}/incoming"
OUT_DIR="${BASE_DIR}/outgoing"

mkdir -p "${IN_DIR}" "${OUT_DIR}"

if ! docker info >/dev/null 2>&1; then
  echo "[supernode] docker is not available; please start Docker and retry." >&2
  exit 1
fi

if [[ -z "${SUPERLINK_HOST}" ]]; then
  echo "[supernode] SUPERLINK_HOST is required (set it to the SuperLink's IP/DNS)." >&2
  exit 1
fi

echo "[supernode] using image: ${IMAGE}"
echo "[supernode] SuperLink target: ${SUPERLINK_HOST}:9092"
echo "[supernode] incoming: ${IN_DIR}"
echo "[supernode] outgoing: ${OUT_DIR}"
echo
echo "[supernode] launching container..."

docker run --rm --name "fedmed-supernode-${CID}" \
  -p 9094:9094 \
  -v "${IN_DIR}":/incoming:ro \
  -v "${OUT_DIR}":/outgoing:rw \
  "${IMAGE}" \
    fedmed-pl-supernode \
      --cid ${CID} \
      --total-clients ${TOTAL_CLIENTS} \
      --superlink-host ${SUPERLINK_HOST} \
      --data-seed ${DATA_SEED} \
      /incoming /outgoing
