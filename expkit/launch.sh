#!/bin/bash
# Run an expkit command from the repository root, in its venv if it has one.
#
#   expkit/launch.sh run --config experiments/foo.yaml --cpu-workers 8
#   sbatch -p main -N 1 -n 32 expkit/launch.sh run --config experiments/foo.yaml --cpu-workers 32
#
# `python -m expkit submit` writes these sbatch lines for you (one per campaign).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY="${PYTHON:-python3}"
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m expkit "$@"
