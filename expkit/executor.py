"""Run a list of specs in parallel worker processes, with retries.

When a worker process starts, it takes one device slot from a queue. A slot
list like ``['cuda:0'] * 3 + ['cuda:1'] * 3`` therefore runs three concurrent
runs per GPU, and ``['cpu'] * 4`` runs four on the CPU. Workers are spawned,
not forked, so CUDA initializes cleanly in each.

A failed run has its traceback appended to the store's ``registry.jsonl`` and
is resubmitted up to ``retries`` more times. It never stops the rest of the
grid. A worker that dies outright (killed by the OOM killer, say) breaks the
process pool, and every worker goes down with it. Only the run the dead worker
was running is charged an attempt, with the signal that killed it. The others
were collateral and go back in the queue at the attempt they were on. Charging
them all would let three crashes exhaust the retries of every run in the grid.

The parent process is the store's only writer: workers *return* the record
and curves, and ``execute`` appends them.
"""
import importlib
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from expkit.api import RunContext, RunResult
from expkit.record import RunRecord, RunSpec, code_version, library_versions
from expkit.seeding import DEFAULT_STREAMS, derive_seeds, enable_determinism
from expkit.store import Store, format_exception
from expkit.threads import export_device_env, export_thread_env, threads_per_worker

_DEVICE = None
_STARTED = None


def load_entry(entry):
    """``'package.module:function'`` -> the function."""
    module, sep, name = entry.partition(':')
    if not sep:
        raise ValueError(f'entry must be module:function, got {entry!r}')
    return getattr(importlib.import_module(module), name)


def _init_worker(queue, threads, started):
    global _DEVICE, _STARTED
    _DEVICE = queue.get()
    _STARTED = started
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(max(1, int(threads)))
    if _DEVICE.startswith('cuda'):
        torch.cuda.set_device(torch.device(_DEVICE))


def _call(worker, key, spec_dict, store_root):
    # A SimpleQueue writes before put() returns, so the parent can still read
    # which run a worker held after the worker is killed mid-run.
    _STARTED.put((os.getpid(), key))
    return worker(spec_dict, store_root, _DEVICE)


def _exit_reason(code):
    if code is not None and code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f'signal {-code}'
        return f'{name} (likely the OOM killer)' if code == -signal.SIGKILL else name
    return f'exit code {code}'


def _drain(started, held):
    """Move the queued ``(pid, key)`` notes into ``held``; a worker's latest run wins.

    This is called after every completion, not only when a worker dies. The
    queue is a pipe: once its 64 KiB fill (~1,700 runs), every worker blocks in
    put() before its next run, and the grid stalls with no error.
    """
    while not started.empty():
        pid, key = started.get()
        held[pid] = key


def _casualties(pool, started, held):
    """``{key: reason}`` for the runs whose worker died on its own.

    Once one worker dies, the pool SIGTERMs the survivors, so a SIGTERM is
    collateral and any other non-zero exit is a death. Read this before
    shutdown(), which drops the process table.

    The futures fail before the pool's manager thread has joined the workers,
    so a dead worker can still read ``exitcode is None`` at that point. Wait
    for the manager thread (it exits once every worker is joined) first.
    """
    manager = getattr(pool, '_executor_manager_thread', None)
    if manager is not None:
        manager.join(timeout=60)
    _drain(started, held)
    out = {}
    for pid, proc in list((getattr(pool, '_processes', None) or {}).items()):
        code = proc.exitcode
        if code in (None, 0, -signal.SIGTERM) or pid not in held:
            continue
        out[held[pid]] = f'worker process {pid} died: {_exit_reason(code)}'
    return out


def train_spec(spec, device='cpu', hooks=None, artifact_dir=None):
    """Run one spec through its entry: ``(seeds, RunResult, wall seconds)``.

    This is the one path a run takes, both at grid time (``run_one``) and on
    replay (``replay``). The result is a function of the spec alone.
    """
    seeds = derive_seeds(spec.global_seed, spec.seed_index,
                         tuple(spec.seed_streams or DEFAULT_STREAMS))
    enable_determinism()
    ctx = RunContext(base=spec.base, cell=dict(spec.cell), seeds=seeds,
                     seed_index=spec.seed_index, device=str(device),
                     experiment=spec.experiment, hooks=dict(hooks or {}),
                     artifact_dir=artifact_dir)
    start = time.time()
    result = RunResult.coerce(load_entry(spec.entry)(ctx))
    return seeds, result, time.time() - start


def run_one(spec_dict, store_root, device='cpu'):
    """Train and score one spec. Returns its record and curves for the parent to store."""
    spec = RunSpec.from_dict(spec_dict)
    artifact_dir = Store(store_root).artifact_dir(spec.key)
    seeds, result, wall = train_spec(spec, device, artifact_dir=artifact_dir)
    sha, dirty = code_version()
    record = RunRecord(
        key=spec.key, experiment=spec.experiment, entry=spec.entry, cell=spec.cell,
        seed_index=spec.seed_index, seeds=seeds, global_seed=spec.global_seed,
        frozen_hash=spec.frozen_hash, base=spec.base,
        metrics={k: _jsonable(v) for k, v in result.metrics.items()},
        steps=result.steps, wall_seconds=wall, diverged=bool(result.diverged),
        device=str(device), versions=library_versions(), git_sha=sha, git_dirty=dirty,
        extra=result.extra, seed_streams=spec.seed_streams)
    curves = {k: np.asarray(v) for k, v in result.curves.items()}
    return {'key': spec.key, 'metrics': record.metrics, 'wall_seconds': wall,
            'device': str(device), 'record': record, 'curves': curves}


def replay(record, device='cpu', hooks=None, base=None):
    """Retrain a stored run (a ``RunRecord``) through the same path: its ``RunResult``.

    ``base`` overrides the record's own, for a setting that leaves the run's
    computation alone (logging more often, say).
    """
    spec = record.spec()
    if base is not None:
        spec.base = base
    return train_spec(spec, device, hooks=hooks)[1]


def _jsonable(value):
    if hasattr(value, 'item'):
        value = value.item()
    return value


def device_slots(gpus=None, per_gpu=1, cpu_workers=1):
    if gpus:
        return [f'cuda:{g}' for g in gpus for _ in range(int(per_gpu))]
    return ['cpu'] * max(1, int(cpu_workers))


def export_worker_env(slots):
    """Export the environment that spawned workers inherit, for ``slots`` running
    at once. Returns the thread count each worker should take.

    Call it once, for everything that runs concurrently, before any pool starts.
    A worker imports numpy (which reads the BLAS caps) before its initializer
    runs, so the environment cannot be set from inside the worker.
    """
    # Use the allocation, not the machine: os.cpu_count() ignores the cgroup.
    threads = threads_per_worker(len(slots))
    export_thread_env(threads)
    export_device_env(slots)            # CPU slots inherit no GPU at all
    return threads


def execute(specs, store_root, slots, *, retries=2, worker=run_one, log=print, threads=None):
    """Run every spec; returns ``{'ok': [keys], 'failed': [keys]}``.

    If ``threads`` is given, the caller has already exported the worker
    environment (``export_worker_env``) and it is left alone. Without it, this
    pool assumes it is the only one running and exports its own.
    """
    specs = list(specs)
    by_key = {s.key: s for s in specs}
    store = Store(store_root)
    if threads is None:
        threads = export_worker_env(slots)
    queue_items = [(s, 1) for s in specs]
    ok, failed = [], []
    total = len(specs)

    while queue_items:
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        for slot in slots:
            queue.put(slot)
        notes, held = ctx.SimpleQueue(), {}
        pool = ProcessPoolExecutor(max_workers=len(slots), mp_context=ctx,
                                   initializer=_init_worker, initargs=(queue, threads, notes))
        started, inflight = {}, {}
        for spec, attempt in queue_items:
            inflight[pool.submit(_call, worker, spec.key, spec.to_dict(), store_root)] = (spec, attempt)
            started[spec.key] = time.time()
        queue_items = []
        broken = False
        try:
            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                _drain(notes, held)
                for future in done:
                    spec, attempt = inflight.pop(future)
                    try:
                        out = future.result()
                    except BrokenProcessPool:
                        inflight[future] = (spec, attempt)   # handled with the rest below
                        raise
                    except Exception as exc:  # noqa: BLE001 -- any failure is recorded
                        store.log_attempt(spec, 'failed', attempt, error=format_exception(exc),
                                          wall_seconds=time.time() - started[spec.key])
                        if attempt <= retries:
                            log(f'  ✗ {spec.key} {spec.label}: {type(exc).__name__}: {exc} '
                                f'-- retry {attempt}/{retries}')
                            new = pool.submit(_call, worker, spec.key, spec.to_dict(), store_root)
                            inflight[new] = (spec, attempt + 1)
                            started[spec.key] = time.time()
                        else:
                            log(f'  ✗ {spec.key} failed after {attempt} attempts')
                            failed.append(spec.key)
                        continue
                    store.append_run(out['record'], out['curves'])
                    store.log_attempt(spec, 'ok', attempt, wall_seconds=out.get('wall_seconds'),
                                      device=out.get('device'))
                    ok.append(spec.key)
                    headline = ', '.join(f'{k}={v:.4g}' for k, v in out['metrics'].items()
                                         if isinstance(v, (int, float)))
                    log(f'  ✓ [{len(ok) + len(failed)}/{total}] {spec.key} {spec.label} '
                        f'({out.get("wall_seconds", 0):.1f}s, {out.get("device")}) {headline}')
        except BrokenProcessPool as exc:
            broken = True
            running = {spec.key for spec, _ in inflight.values()}
            dead = {k: v for k, v in _casualties(pool, notes, held).items() if k in running}
            for future, (spec, attempt) in inflight.items():
                if dead and spec.key not in dead:
                    queue_items.append((spec, attempt))     # collateral: not its fault
                    continue
                # With no culprit identified, every in-flight run is charged,
                # so each rebuild still spends an attempt and the loop ends.
                reason = dead.get(spec.key) or f'worker process died: {exc}'
                store.log_attempt(spec, 'failed', attempt, error=reason,
                                  wall_seconds=time.time() - started[spec.key])
                if dead:
                    log(f'  ✗ {spec.key} {spec.label}: {reason}'
                        f'{f" -- retry {attempt}/{retries}" if attempt <= retries else ""}')
                if attempt <= retries:
                    queue_items.append((spec, attempt + 1))
                else:
                    failed.append(spec.key)
            log(f'  ! a worker died; rebuilding the pool for {len(queue_items)} runs')
        finally:
            pool.shutdown(wait=not broken, cancel_futures=broken)

    assert set(ok) | set(failed) <= set(by_key)
    return {'ok': ok, 'failed': failed}
