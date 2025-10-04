#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-localhost/pl-example:1.0.0}"

echo "→ Refreshing image: $TAG"

# stop/remove any containers created from this image
if ids=$(docker ps -aq --filter "ancestor=$TAG"); [[ -n "${ids}" ]]; then
  echo "  removing containers: $ids"
  docker rm -fv $ids
fi

# remove old image if present
if docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "  removing old image"
  docker rmi -f "$TAG"
fi

# rebuild with same tag (no cache avoids stale layers)
echo "  building fresh image"
docker build --no-cache -t "$TAG" .

echo "done:"
docker image inspect "$TAG" --format '  {{.RepoTags}}  ID={{.Id}}'