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
    EXPKIT_LANES         the job lanes (see ``parse_lanes``); unset: one GPU lane
                         from the partition / GPU-type / per-GPU flags
    EXPKIT_JOB_PREFIX    job-name prefix (default 'xk')
    EXPKIT_SBATCH_ARGS   anything else, appended verbatim (e.g. '--time=12:00:00')
"""
import math
import os
import shlex
import subprocess
from collections import namedtuple

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


Lane = namedtuple('Lane', 'partition gpu_type slots cpus_per_gpu mem_gb per_gpu speed')


def parse_lanes(text):
    """Job lanes from ``partition:gpu_type:slots:cpus_per_gpu:mem_gb:per_gpu[:speed]``,
    whitespace-separated, e.g. ``'mixed:pro6000:2:40:120:12 main:cpu:4:32::12:0.3'``.

    A lane is one shape of job in one queue. ``slots`` is how many of its jobs
    your limits let run at once and ``speed`` (default 1) how fast one of them
    trains relative to the others; campaigns go to whichever job slot would
    finish them soonest (``assign_lane``). A ``gpu_type`` of ``cpu`` makes a CPU
    lane: no card, ``cpus_per_gpu`` cores and ``per_gpu`` workers per job. An
    empty ``gpu_type`` or ``mem_gb`` is omitted from the request (``mem_gb``
    then falls back to ``EXPKIT_GB_PER_CORE``). Jobs beyond a lane's ``slots``
    simply pend.
    """
    lanes = []
    for spec in (text or '').split():
        fields = spec.split(':')
        if len(fields) not in (6, 7):
            raise SystemExit(f'lane {spec!r}: want partition:gpu_type:slots:cpus_per_gpu:'
                             'mem_gb:per_gpu[:speed]')
        part, gpu_type, slots, cpus, mem, per_gpu = fields[:6]
        lanes.append(Lane(part or None, gpu_type or None, int(slots), int(cpus),
                          int(mem) if mem else None, int(per_gpu),
                          float(fields[6]) if len(fields) == 7 else 1.0))
    return lanes


def is_cpu(lane):
    return lane.gpu_type == 'cpu'


def assign_lane(lanes, filled, load, allowed=None):
    """The lane among ``allowed`` (default: all) with the job slot that would finish
    ``load`` soonest; adds it to that slot of ``filled`` (per lane, per slot).

    Slots, not whole lanes, because one campaign runs as one job: a long one
    finishes no sooner on a slow lane for that lane having other slots free.
    """
    allowed = range(len(lanes)) if allowed is None else allowed
    i, j = min(((i, j) for i in allowed for j in range(len(filled[i]))),
               key=lambda ij: filled[ij[0]][ij[1]] + load / lanes[ij[0]].speed)
    filled[i][j] += load / lanes[i].speed
    return lanes[i]


def submit(name, cores, args, *, dependency='singleton', partition=None, gb_per_core=None,
           mem_gb=None, gpus=None, gpu_type=None, cpus_per_gpu=None, parsable=True,
           kill_on_invalid=False,
           extra=None, plan=False, runner=subprocess.run, log=say):
    """Submit ``launch.sh <args>`` as one job. Returns the job id (a placeholder
    under ``plan``), or None on failure. ``kill_on_invalid`` cancels a job
    whose dependency can never be met (its ``afterok`` failed) rather than
    leaving it pending forever.

    With ``gpus`` and ``cpus_per_gpu``, the job asks for ``--cpus-per-gpu``
    (the form the mixed queue takes) instead of ``-n cores``, and ``cores``
    becomes ``gpus x cpus_per_gpu`` for the memory request. ``mem_gb``
    overrides that request."""
    partition = partition or os.environ.get('EXPKIT_PARTITION')
    per_gpu = bool(gpus and cpus_per_gpu)
    if per_gpu:
        cores = int(gpus) * int(cpus_per_gpu)
    mem = mem_gb or memory_gb(cores, gb_per_core)
    extra = list(extra or []) + shlex.split(os.environ.get('EXPKIT_SBATCH_ARGS', ''))
    cmd = ['sbatch', *(['--parsable'] if parsable else []), f'--job-name={name}',
           f'--chdir={REPO}', '--output=slurm/%x-%j.out', f'--dependency={dependency}',
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
