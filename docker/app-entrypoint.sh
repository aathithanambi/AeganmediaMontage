#!/usr/bin/env sh
set -eu

if [ "${INIT_DB_ON_BOOT:-false}" = "true" ]; then
  echo "INIT_DB_ON_BOOT=true -> running bootstrap"
  python -m webapp.bootstrap
fi

exec uvicorn webapp.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-41005}"

