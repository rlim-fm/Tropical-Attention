# Shared by tune_all.sh, approve_all.sh and run_all.sh; sourced, not run.
#
#   CONFIG       experiment YAML                          (configs/tropical.yaml)
#   CAMPAIGNS    comma list of campaigns                  (every campaign)
#   PROFILE      one profile instead, e.g. dry_run        (LOCAL=1 only)
#   LOCAL=1      run here, one campaign after another, instead of submitting to Slurm
#   GPUS         GPUs per Slurm job / used locally        (1; locally 0 without nvidia-smi)
#   PER_GPU      concurrent runs per GPU                  (4)
#   CORES        CPU cores per Slurm job                  (8)
#   CPU_WORKERS  concurrent CPU runs when GPUS=0          (4)
# Slurm itself: EXPKIT_PARTITION, EXPKIT_GB_PER_CORE, EXPKIT_JOB_PREFIX, EXPKIT_SBATCH_ARGS.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG="${CONFIG:-configs/tropical.yaml}"
GPUS="${GPUS:-1}"
PER_GPU="${PER_GPU:-4}"
CORES="${CORES:-8}"
CPU_WORKERS="${CPU_WORKERS:-4}"
EXPKIT="$REPO_ROOT/expkit/launch.sh"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then PY="$REPO_ROOT/.venv/bin/python"; else PY="${PYTHON:-python3}"; fi

# The campaigns (or the one profile) to act on, one per line.
targets() {
    if [ -n "${PROFILE:-}" ]; then
        echo "$PROFILE"
    elif [ -n "${CAMPAIGNS:-}" ]; then
        tr ',' '\n' <<< "$CAMPAIGNS"
    else
        PYTHONPATH="$REPO_ROOT" "$PY" -c "from expkit.config import campaigns; print('\n'.join(campaigns('$CONFIG')))"
    fi
}

# Device flags for a local run: every visible GPU slot, or CPU workers.
local_devices() {
    if [ "$GPUS" -gt 0 ] && command -v nvidia-smi > /dev/null; then
        echo "--gpus $(seq -s, 0 $((GPUS - 1))) --per-gpu $PER_GPU"
    else
        echo "--cpu-workers $CPU_WORKERS"
    fi
}

# Device flags for one Slurm job.
slurm_devices() {
    if [ "$GPUS" -gt 0 ]; then echo "--gpus $GPUS --per-gpu $PER_GPU --cores $CORES"; else echo "--cores $CORES"; fi
}

campaign_flag() {
    if [ -n "${CAMPAIGNS:-}" ]; then echo "--campaigns $CAMPAIGNS"; fi
}
