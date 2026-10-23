# Sourced by run_all.sh, not run.
#
#   CONFIG       experiment YAML                          (configs/tropical.yaml)
#   CAMPAIGNS    comma list of campaigns                  (every campaign)
#   PROFILE      one profile instead, e.g. dry_run        (LOCAL=1 only)
#   LOCAL=1      run here, one campaign after another, instead of submitting to Slurm
#   GPUS         GPUs per Slurm job / used locally        (1; locally 0 without nvidia-smi)
#   PER_GPU      concurrent workers per GPU               (12: a campaign's 12 cells at once)
#   GROUP_SEEDS  1: each worker trains all seeds of a cell at once (train:run_group)  (1)
#   GPU_TYPE     the card a GPU job asks for              (pro6000)
#   CPUS_PER_GPU cores a GPU job asks for, per GPU        (40)
#   CPU_CORES    cores of a campaign with slurm.device: cpu, and of a GPUS=0 job (32)
#   CPU_WORKERS  concurrent CPU runs locally when GPUS=0  (4)
#   EXPKIT_MPS   1: run a CUDA MPS daemon inside each GPU job (expkit/launch.sh)  (1)
#
# The defaults fit the Bowdoin HPC: GPU jobs go to the `mixed` queue, whose PRO6000
# nodes (192 cores, 760 GB, 3 cards) take your full mixed allowance of 2 cards,
# 80 cores and 500 GB as two jobs of 1 card + 40 cores + 120 GB. (The `gpu` queue
# caps you at 4 cores in total, too few for many workers per card.) Campaigns
# marked `slurm: {device: cpu}` in the config, and the report jobs, go to `main`.
# Slurm itself: EXPKIT_PARTITION (mixed), EXPKIT_CPU_PARTITION (main),
# EXPKIT_GB_PER_CORE (3), EXPKIT_JOB_PREFIX, EXPKIT_SBATCH_ARGS.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG="${CONFIG:-configs/tropical.yaml}"
GPUS="${GPUS:-1}"
PER_GPU="${PER_GPU:-12}"
GROUP_SEEDS="${GROUP_SEEDS:-1}"
GPU_TYPE="${GPU_TYPE:-pro6000}"
CPUS_PER_GPU="${CPUS_PER_GPU:-40}"
CPU_CORES="${CPU_CORES:-32}"
CPU_WORKERS="${CPU_WORKERS:-4}"
export EXPKIT_PARTITION="${EXPKIT_PARTITION:-mixed}"
export EXPKIT_CPU_PARTITION="${EXPKIT_CPU_PARTITION:-main}"
export EXPKIT_GB_PER_CORE="${EXPKIT_GB_PER_CORE:-3}"
export EXPKIT_MPS="${EXPKIT_MPS:-1}"         # sbatch passes the environment on to the job
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

group_flag() {
    if [ "$GROUP_SEEDS" = 1 ]; then echo "--group-seeds"; fi
}

# Device flags for a local run: every visible GPU slot, or CPU workers.
local_devices() {
    if [ "$GPUS" -gt 0 ] && command -v nvidia-smi > /dev/null; then
        echo "--gpus $(seq -s, 0 $((GPUS - 1))) --per-gpu $PER_GPU $(group_flag)"
    else
        echo "--cpu-workers $CPU_WORKERS $(group_flag)"
    fi
}

# Device flags for the Slurm jobs.
slurm_devices() {
    if [ "$GPUS" -gt 0 ]; then
        echo "--gpus $GPUS --per-gpu $PER_GPU --gpu-type $GPU_TYPE --cpus-per-gpu $CPUS_PER_GPU" \
             "--cpu-cores $CPU_CORES $(group_flag)"
    else
        echo "--cores $CPU_CORES $(group_flag)"
    fi
}

campaign_flag() {
    if [ -n "${CAMPAIGNS:-}" ]; then echo "--campaigns $CAMPAIGNS"; fi
}
