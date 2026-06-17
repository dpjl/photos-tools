#!/usr/bin/env bash
# Lance le comparateur de photos en mode web (API FastAPI + SPA Svelte buildée).
#
#   ./run_web.sh [PORT]
#
# Pré-requis :
#   - dépendances Python : .venv/bin/python -m pip install fastapi "uvicorn[standard]" python-multipart pillow
#   - build du front (une fois, ou après modif) :  (cd web/frontend && npm install && npm run build)
#
# Le serveur écoute sur 0.0.0.0 → accessible depuis un mobile sur le même réseau.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8000}"
PY="$HERE/.venv/bin/python"

if [ ! -d "$HERE/web/frontend/dist" ]; then
  echo "⚠  Front non buildé. Lancez : (cd web/frontend && npm install && npm run build)" >&2
fi

cd "$HERE"
exec "$PY" -m uvicorn web.server.main:app --host 0.0.0.0 --port "$PORT"
