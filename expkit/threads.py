"""How many CPUs this process may actually use, and how to divide them.

``os.cpu_count()`` reports the machine's cores, not this job's allocation.
Under ``sbatch --cpus-per-task=8`` on a 128-core node it still says 128, so a
thread count derived from it oversubscribes the cgroup by 16x and the workers'
BLAS threads fight each other for the cores Slurm actually granted. Ask the
scheduler first, fall back to the affinity mask, and only then to the machine.

Used by the worker pool (``expkit/executor.py``) to split the allocation
across concurrent runs, and by single-process entry points (``expkit replay``),
which would otherwise leave torch at its default of one thread per machine core.

It also owns the other half of the worker environment: ``export_device_env``
hides the GPUs from a CPU-only launch. Torch can initialize a CUDA context even
for CPU parameters, and a busy GPU then kills every worker but the first.
"""
import os

_SLURM_VARS = ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE')


def allocated_cpus() -> int:
    """CPUs this process may use: the Slurm grant, else the affinity mask, else the machine."""
    for var in _SLURM_VARS:
        value = (os.environ.get(var) or '').strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    affinity = getattr(os, 'sched_getaffinity', None)   # Linux only; absent on macOS
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def threads_per_worker(n_workers: int = 1) -> int:
    """The allocation split evenly across ``n_workers`` concurrent processes."""
    return max(1, allocated_cpus() // max(1, int(n_workers)))


def export_thread_env(threads: int) -> None:
    """Pin the BLAS thread counts for processes spawned after this call.

    OpenBLAS and MKL read these once at import, so setting them here only
    affects children -- which is the point: a spawned worker imports numpy and
    torch fresh, and without this each one would start a pool sized to the
    whole machine.
    """
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(var, str(max(1, int(threads))))


def configure_threads(n_workers: int = 1) -> int:
    """Set torch's intra-op width to this process's share; returns the count."""
    import torch
    threads = threads_per_worker(n_workers)
    torch.set_num_threads(threads)
    return threads


def export_device_env(devices) -> bool:
    """Hide the GPUs when every fit is headed for the CPU; returns whether it did.

    A CPU fit still touches CUDA. Adam's ``step`` runs a graph-capture health
    check that asks ``torch.accelerator.current_stream()`` whenever a CUDA
    build sees a device, whatever the parameters sit on, and that call
    initializes a context. On a GPU in exclusive-process mode -- or one another
    job already holds -- the second process onwards dies with
    ``cudaErrorDevicesUnavailable``, so ``--workers 8`` on the CPU failed on a
    node whose GPU was busy while ``--workers 1`` went through.

    Emptying ``CUDA_VISIBLE_DEVICES`` makes the device count zero, the health
    check falls through and no context is created. It is set, not defaulted:
    Slurm exports the variable for any job with a ``--gres``, and a run that
    asked for the CPU means it. Call it before torch first looks at CUDA --
    the count is read once -- and the processes spawned afterwards inherit it.
    """
    devices = [str(d) for d in ([devices] if isinstance(devices, str) else devices)]
    if not devices or any(not d.startswith('cpu') for d in devices):
        return False
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return True
