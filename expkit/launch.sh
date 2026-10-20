#!/bin/bash
# Run an expkit command from the repository root, in its venv if it has one.
#
#   expkit/launch.sh run --config experiments/foo.yaml --cpu-workers 8
#   sbatch -p main -N 1 -n 32 expkit/launch.sh run --config experiments/foo.yaml --cpu-workers 32
#
# `python -m expkit submit` writes these sbatch lines for you (one per campaign).
#
# EXPKIT_MPS=1 starts a private CUDA MPS daemon for the job when it holds a GPU
# (and nvidia-cuda-mps-control exists), so the worker processes sharing a card
# run their small kernels side by side instead of time-slicing whole contexts.
# The daemon's pipe and log directories live under $TMPDIR, so two jobs on one
# node never share a daemon, and it is shut down when the command exits.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY="${PYTHON:-python3}"
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ "${EXPKIT_MPS:-0}" = 1 ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] \
        && command -v nvidia-cuda-mps-control > /dev/null; then
    MPS_DIR="$(mktemp -d "${TMPDIR:-/tmp}/expkit-mps-${SLURM_JOB_ID:-$$}.XXXX")"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if nvidia-cuda-mps-control -d; then
        echo "expkit: CUDA MPS daemon started for GPU(s) $CUDA_VISIBLE_DEVICES ($MPS_DIR)"
        trap 'echo quit | nvidia-cuda-mps-control; rm -rf "$MPS_DIR"' EXIT
        status=0
        "$PY" -m expkit "$@" || status=$?
        exit "$status"
    fi
    echo "expkit: MPS daemon did not start; running without it" >&2
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
exec "$PY" -m expkit "$@"
