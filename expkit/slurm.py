"""Submitting campaigns to Slurm: job names, the queue, and sbatch.

``expkit submit`` runs on the login node and submits one job per campaign,
named ``<prefix>-<experiment>-<campaign>`` with ``--dependency=singleton``.
Two jobs on one campaign would both train the same missing runs. A name already
in ``squeue`` is not submitted again, and a campaign with nothing missing gets
no job. Each job runs ``expkit/launch.sh run ...``. Every job's stdout goes to
``slurm/%x-%j.out`` in the repository.

Settings come from flags or the environment:

    EXPKIT_PARTITION     -p value, e.g. 'main,mixed' (Slurm starts the job in
                         whichever listed queue has room first); unset: omitted
    EXPKIT_CPU_PARTITION -p value for the campaigns sent to the CPU while the
                         others take GPUs; unset: omitted (the default queue)
    EXPKIT_GPU_TYPE      --gres=gpu:<type>:N, e.g. 'pro6000'; unset: any GPU
    EXPKIT_GB_PER_CORE   --mem = ceil(cores x this); unset: omitted
    EXPKIT_JOB_PREFIX    job-name prefix (default 'xk')
    EXPKIT_SBATCH_ARGS   anything else, appended verbatim (e.g. '--time=12:00:00')
"""
import math
import os
import shlex
import subprocess

from expkit.record import PACKAGE, REPO


def say(*args):
    print(*args, flush=True)


def job_name(experiment, campaign, prefix=None):
    prefix = prefix or os.environ.get('EXPKIT_JOB_PREFIX', 'xk')
    return f'{prefix}-{experiment}-{campaign}'.replace('/', '-').replace('=', '-')


def queued(name, runner=subprocess.run):
    """Your pending or running job ids under this name; empty off the cluster."""
    try:
        out = runner(['squeue', '-h', '-u', os.environ.get('USER', ''), '-n', name, '-o', '%i'],
                     capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    return out.stdout.split()


def memory_gb(cores, gb_per_core=None):
    gb = gb_per_core if gb_per_core is not None else os.environ.get('EXPKIT_GB_PER_CORE')
    return math.ceil(float(gb) * cores) if gb else None


def launcher():
    return os.path.relpath(os.path.join(PACKAGE, 'launch.sh'), REPO)


def gres(gpus, gpu_type=None):
    """``--gres`` value: ``gpu:N``, or ``gpu:<type>:N`` to pin the card."""
    gpu_type = gpu_type or os.environ.get('EXPKIT_GPU_TYPE')
    return f'gpu:{gpu_type}:{gpus}' if gpu_type else f'gpu:{gpus}'


def submit(name, cores, args, *, dependency='singleton', partition=None, gb_per_core=None,
           gpus=None, gpu_type=None, cpus_per_gpu=None, parsable=True, kill_on_invalid=False,
           extra=None, plan=False, runner=subprocess.run, log=say):
    """Submit ``launch.sh <args>`` as one job. Returns the job id (a placeholder
    under ``plan``), or None on failure. ``kill_on_invalid`` cancels a job
    whose dependency can never be met (its ``afterok`` failed) rather than
    leaving it pending forever.

    With ``gpus`` and ``cpus_per_gpu``, the job asks for ``--cpus-per-gpu``
    (the form the mixed queue takes) instead of ``-n cores``, and ``cores``
    becomes ``gpus x cpus_per_gpu`` for the memory request."""
    partition = partition or os.environ.get('EXPKIT_PARTITION')
    per_gpu = bool(gpus and cpus_per_gpu)
    if per_gpu:
        cores = int(gpus) * int(cpus_per_gpu)
    mem = memory_gb(cores, gb_per_core)
    extra = list(extra or []) + shlex.split(os.environ.get('EXPKIT_SBATCH_ARGS', ''))
    cmd = ['sbatch', *(['--parsable'] if parsable else []), f'--job-name={name}',
           '--output=slurm/%x-%j.out', f'--dependency={dependency}',
           *(['--kill-on-invalid-dep=yes'] if kill_on_invalid else []),
           *(['-p', partition] if partition else []), '-N', '1',
           *([f'--cpus-per-gpu={cpus_per_gpu}'] if per_gpu else ['-n', str(cores)]),
           *([f'--mem={mem}G'] if mem else []), *([f'--gres={gres(gpus, gpu_type)}'] if gpus else []),
           *extra, launcher(), *args]
    log('$ ' + ' '.join(shlex.quote(c) for c in cmd))
    if plan:
        return f'<{name}>'
    os.makedirs(os.path.join(REPO, 'slurm'), exist_ok=True)
    out = runner(cmd, cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        log(f'  sbatch failed: {(out.stderr or "").strip()}')
        return None
    return (out.stdout or '').strip().split(';')[0] or True
