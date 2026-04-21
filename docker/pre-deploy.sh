#!/usr/bin/env bash
# pre-deploy.sh — Runs on the remote server BEFORE deploying new containers.
# Stops and removes old AeganMediaMontage containers and prunes dangling images.
#
# IMPORTANT: This script NEVER deletes user data!
# - aeganmedia-data/videos/ — User videos (only deleted by auto-cleanup based on VIDEO_RETENTION_HOURS)
# - aeganmedia-data/projects/ — Project files (persistent)
# - aeganmedia-data/.env — Environment config (persistent)
#
# Only Docker containers and images are cleaned up during deployment.
set -euo pipefail

CONTAINERS=("aeganmediamontage-web" "aeganmediamontage-worker")

echo "=== Pre-deploy cleanup (containers only, data preserved) ==="

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
