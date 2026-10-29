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
#   GPU_Q_PER_GPU  workers per card in the `gpu` queue's jobs, which get 1-2 cores  (4)
#   EXPKIT_LANES the job shapes campaigns are spread over (expkit/slurm.py: parse_lanes)
#   CPU_CORES    cores of a CPU-lane job, and of a GPUS=0 job (32)
#   CPU_JOBS     CPU-lane jobs at once in `main`           (4: 128 of its 150 cores)
#   CPU_JOB_WORKERS  concurrent workers in a CPU-lane job  (12: a campaign's 12 cells)
#   CPU_SPEED    a CPU-lane job's speed relative to a GPU job (0.3; calibrate from the logs)
#   CPU_WORKERS  concurrent CPU runs locally when GPUS=0  (4)
#   EXPKIT_MPS   1: run a CUDA MPS daemon inside each GPU job (expkit/launch.sh)  (1)
#
# The defaults fit the Bowdoin HPC, whose `mixed` and `gpu` allowances stack. GPU
# campaigns are spread over three lanes, in proportion to the jobs each runs at once:
#   mixed  2 jobs of 1 PRO6000 + 40 cores + 120 GB, PER_GPU workers   (2 cards, 80 cores)
#   gpu    2 jobs of 1 PRO6000 + 1 core + 13 GB, GPU_Q_PER_GPU workers
#   gpu    1 job  of 1 A100    + 2 cores + 14 GB, GPU_Q_PER_GPU workers (3 cards, 4 cores, 40 GB)
#   main   4 jobs of 32 cores, CPU_JOB_WORKERS workers, speed CPU_SPEED  (128 cores)
# A lane is partition:gpu_type:slots:cpus_per_gpu:mem_gb:per_gpu[:speed] (gpu_type
# `cpu`: a CPU job); an empty mem_gb is cpus x EXPKIT_GB_PER_CORE. A campaign goes to
# the job slot that would finish it soonest: `slurm: {device: any}` in the config
# may take any lane, `gpu` only the GPU lanes, `cpu` only `main`. Speed is a job's
# throughput relative to a `mixed` job; set it from the runs' times in slurm/*.out
# (CPU_SPEED, or a :speed on a gpu-queue lane that proves slower). The grand-report
# job goes to `main` too.
# Slurm itself: EXPKIT_PARTITION (mixed), EXPKIT_CPU_PARTITION (main),
# EXPKIT_GB_PER_CORE (3), EXPKIT_JOB_PREFIX, EXPKIT_SBATCH_ARGS.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG="${CONFIG:-configs/tropical.yaml}"
GPUS="${GPUS:-1}"
PER_GPU="${PER_GPU:-48}"
GROUP_SEEDS="${GROUP_SEEDS:-1}"
GPU_TYPE="${GPU_TYPE:-pro6000}"
CPUS_PER_GPU="${CPUS_PER_GPU:-40}"
GPU_Q_PER_GPU="${GPU_Q_PER_GPU:-4}"
CPU_CORES="${CPU_CORES:-32}"
CPU_JOBS="${CPU_JOBS:-4}"
CPU_JOB_WORKERS="${CPU_JOB_WORKERS:-12}"
CPU_SPEED="${CPU_SPEED:-0.3}"
CPU_WORKERS="${CPU_WORKERS:-4}"
export EXPKIT_PARTITION="${EXPKIT_PARTITION:-mixed}"
export EXPKIT_CPU_PARTITION="${EXPKIT_CPU_PARTITION:-main}"
export EXPKIT_GB_PER_CORE="${EXPKIT_GB_PER_CORE:-3}"
export EXPKIT_LANES="${EXPKIT_LANES-$EXPKIT_PARTITION:$GPU_TYPE:2:$CPUS_PER_GPU::$PER_GPU gpu:pro6000:2:1:13:$GPU_Q_PER_GPU gpu:a100:1:2:14:$GPU_Q_PER_GPU $EXPKIT_CPU_PARTITION:cpu:$CPU_JOBS:$CPU_CORES::$CPU_JOB_WORKERS:$CPU_SPEED}"
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
