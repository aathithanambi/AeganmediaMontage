#!/usr/bin/env sh
set -eu

mkdir -p /app/projects /var/aeganmediamontage/videos /tmp

if [ ! -w /app/projects ] || [ ! -w /var/aeganmediamontage/videos ]; then
  echo "Required writable paths are not writable for current user."
  echo "Check volume permissions for /app/projects and /var/aeganmediamontage/videos."
  exit 1
fi

if [ "${INIT_DB_ON_BOOT:-false}" = "true" ]; then
  echo "INIT_DB_ON_BOOT=true -> running bootstrap"
  python -m webapp.bootstrap
fi

exec uvicorn webapp.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-41006}"

