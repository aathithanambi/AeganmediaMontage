#!/usr/bin/env sh
set -eu

# Use VIDEOS_ROOT from env or default
VIDEOS_DIR="${VIDEOS_ROOT:-/var/aeganmediamontage/videos}"
PROJECTS_DIR="/app/projects"

mkdir -p "$PROJECTS_DIR" "$VIDEOS_DIR" /tmp

if [ ! -w "$PROJECTS_DIR" ]; then
  echo "ERROR: $PROJECTS_DIR is not writable for current user."
  echo "Check volume permissions."
  exit 1
fi

if [ ! -w "$VIDEOS_DIR" ]; then
  echo "ERROR: $VIDEOS_DIR is not writable for current user."
  echo "Check volume permissions."
  exit 1
fi

echo "Writable paths OK: $PROJECTS_DIR, $VIDEOS_DIR"

if [ "${INIT_DB_ON_BOOT:-false}" = "true" ]; then
  echo "INIT_DB_ON_BOOT=true -> running bootstrap"
  python -m webapp.bootstrap
fi

# If arguments are passed (e.g. "python -m webapp.worker"), run those instead of uvicorn
if [ $# -gt 0 ]; then
  echo "Running custom command: $@"
  exec "$@"
fi

exec uvicorn webapp.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-51004}"

