#!/usr/bin/env bash
# Convenience wrapper so everything can be run from the repository root.
# Usage:
#   ./minichris          # bring the stack up (runs minichris.sh)
#   ./minichris down     # tear the stack down (runs unmake.sh)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_DIR="${REPO_ROOT}/minichris"

if [[ ! -d "${STACK_DIR}" ]]; then
  echo "error: expected miniChRIS assets under ${STACK_DIR}" >&2
  exit 1
fi

cd "${STACK_DIR}"

if [[ "${1:-}" == "down" ]]; then
  shift
  ./unmake.sh "$@"
else
  ./minichris.sh "$@"
fi
