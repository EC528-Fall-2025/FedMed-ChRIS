#!/usr/bin/env bash
# Run the FedMed SuperLink locally with published ports and helper paths.

set -euo pipefail

IMAGE="${IMAGE:-docker.io/fedmed/pl-superlink:0.2.0}"
ROUNDS=3
TOTAL_CLIENTS=3
LOCAL_EPOCHS=10
LEARNING_RATE=0.001
DATA_SEED=13
FRACTION_EVALUATE=1.0
SUMMARY_FILE="server_summary.json"
HOST_PORT_FLEET=9092
HOST_PORT_CONTROL=9093
HOST_PORT_SERVERAPP=9091

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/superlink_run"
IN_DIR="${BASE_DIR}/incoming"
OUT_DIR="${BASE_DIR}/outgoing"

mkdir -p "${IN_DIR}" "${OUT_DIR}"

resolve_host_ip() {
  local ip
  ip=$(hostname -I 2>/dev/null | awk 'NF {print $1; exit}') || true
  if [[ -n "${ip:-}" ]]; then
    echo "${ip}"
    return
  fi

  if command -v ipconfig >/dev/null 2>&1; then
    for iface in en0 en1 en2; do
      ip=$(ipconfig getifaddr "${iface}" 2>/dev/null || true)
      if [[ -n "${ip}" ]]; then
        echo "${ip}"
        return
      fi
    done
  fi

  if command -v ip >/dev/null 2>&1; then
    ip=$(ip route get 1 2>/dev/null | awk 'NR==1 {for(i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}') || true
    if [[ -n "${ip}" ]]; then
      echo "${ip}"
      return
    fi
  fi

  echo "localhost"
}

HOST_IP="${ADVERTISE_HOST:-$(resolve_host_ip)}"

if ! docker info >/dev/null 2>&1; then
  echo "[superlink] docker is not available; please start Docker and retry." >&2
  exit 1
fi

echo "[superlink] using image: ${IMAGE}"
echo "[superlink] host IP: ${HOST_IP}"
echo "[superlink] incoming: ${IN_DIR}"
echo "[superlink] outgoing: ${OUT_DIR}"
echo "[superlink] exposing ports: ${HOST_PORT_SERVERAPP},${HOST_PORT_FLEET},${HOST_PORT_CONTROL}"
echo "[superlink] connect SuperNodes to ${HOST_IP}:${HOST_PORT_FLEET}"
echo
echo "[superlink] launching container..."

docker run --rm --name fedmed-superlink \
  -p ${HOST_PORT_SERVERAPP}:9091 \
  -p ${HOST_PORT_FLEET}:9092 \
  -p ${HOST_PORT_CONTROL}:9093 \
  -v "${IN_DIR}":/incoming:ro \
  -v "${OUT_DIR}":/outgoing:rw \
  "${IMAGE}" \
    fedmed-pl-superlink \
      --host 0.0.0.0 \
      --fleet-port 9092 \
      --control-port 9093 \
      --serverapp-port 9091 \
      --rounds ${ROUNDS} \
      --total-clients ${TOTAL_CLIENTS} \
      --local-epochs ${LOCAL_EPOCHS} \
      --learning-rate ${LEARNING_RATE} \
      --data-seed ${DATA_SEED} \
      --fraction-evaluate ${FRACTION_EVALUATE} \
      --summary-file ${SUMMARY_FILE} \
      /incoming /outgoing
