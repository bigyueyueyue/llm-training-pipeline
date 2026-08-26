#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-dev}"

uv venv --python 3.12 .venv
source .venv/bin/activate

case "$MODE" in
  dev)   uv pip install -r requirements-dev.txt ;;
  train) uv pip install -r requirements-train.txt ;;
  *) echo "usage: $0 [dev|train]" >&2; exit 1 ;;
esac
