#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-localhost/pl-chrnist:1.0.0}" # rename tag here for clarity

# remove any containers using the image, then remove the image
if ids=$(docker ps -aq --filter "ancestor=$TAG"); [[ -n "${ids}" ]]; then
  echo "  removing containers: $ids"
  docker rm -fv $ids
fi
if docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "  removing old image"
  docker rmi -f "$TAG"
fi

# rebuild with same tag (no cache avoids stale layers)
docker build --no-cache -t "$TAG" .

docker image inspect "$TAG" --format '  {{.RepoTags}}  ID={{.Id}}'