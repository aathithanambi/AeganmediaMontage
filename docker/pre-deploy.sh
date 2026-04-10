#!/usr/bin/env bash
# pre-deploy.sh — Runs on the remote server BEFORE deploying new containers.
# Stops and removes old AeganMediaMontage containers and prunes dangling images.
set -euo pipefail

CONTAINERS=("aeganmediamontage-web" "aeganmediamontage-worker")

echo "=== Pre-deploy cleanup ==="

for cname in "${CONTAINERS[@]}"; do
  if docker ps -a --format '{{.Names}}' | grep -qx "$cname"; then
    echo "Stopping container: $cname"
    docker stop "$cname" --time 15 2>/dev/null || true
    echo "Removing container: $cname"
    docker rm -f "$cname" 2>/dev/null || true
  else
    echo "Container $cname does not exist, skipping."
  fi
done

echo "Removing old aeganmediamontage-web images (keeping none — new image loaded after this)..."
docker images --filter "reference=aeganmediamontage-web" --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | while read -r img; do
  echo "  Removing image: $img"
  docker rmi "$img" 2>/dev/null || true
done

echo "Pruning dangling (unused/untagged) images..."
docker image prune -f 2>/dev/null || true

echo "=== Pre-deploy cleanup complete ==="
