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

**Seed groups** (``group_seeds=True``, ``expkit run --group-seeds``): when the
entry module also defines ``<name>_group`` (``train:run`` -> ``train:run_group``),
the pending seeds of each cell go to one worker together, which calls it with
their ``RunContext`` list and gets one ``RunResult`` per seed back, or the
exception a seed failed with. A small model then trains S seeds per kernel
launch instead of one. Each seed is still stored as its own run (with
``extra['seed_group']`` and its share of the group's wall time), and only the
seeds still missing are grouped, so a re-run fills the gaps. A slot holds one
group, so ``--per-gpu`` counts groups. A seed that fails, and every seed of a
group that raises or whose worker dies, is retried on its own, so one bad
seed never sinks its siblings twice. ``replay`` retrains one seed alone, and
batched kernels round differently, so a grouped run replays only to float
tolerance (``expkit replay --tolerance 1e-5``), not bit for bit.
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
from expkit.config import canonical_json
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


def _call(worker, ident, payload, store_root):
    # A SimpleQueue writes before put() returns, so the parent can still read
    # which run (or seed group) a worker held after the worker is killed mid-run.
    _STARTED.put((os.getpid(), ident))
    return worker(payload, store_root, _DEVICE)


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
    return _output(spec, seeds, result, wall, device)


def group_entry(entry):
    """``'module:name_group'`` if the entry's module defines one, else None."""
    module, sep, name = entry.partition(':')
    try:
        fn = getattr(importlib.import_module(module), f'{name}_group', None)
    except ImportError:
        return None
    return f'{module}:{name}_group' if callable(fn) else None


def seed_groups(specs):
    """The specs bucketed by everything but the seed: one list per cell, in order."""
    buckets = {}
    for s in specs:
        key = (s.entry, canonical_json(s.cell), s.frozen_hash, canonical_json(s.base),
               s.global_seed, tuple(s.seed_streams or ()))
        buckets.setdefault(key, []).append(s)
    return list(buckets.values())


def run_group(spec_dicts, store_root, device='cpu'):
    """Train the seeds of one cell together through the entry's ``<name>_group``.

    Returns one item per spec: ``run_one``'s output, or ``{'key', 'error'}``
    for a seed the group entry reported as failed.
    """
    specs = [RunSpec.from_dict(d) for d in spec_dicts]
    store = Store(store_root)
    all_seeds = [derive_seeds(s.global_seed, s.seed_index, tuple(s.seed_streams or DEFAULT_STREAMS))
                 for s in specs]
    enable_determinism()
    ctxs = [RunContext(base=s.base, cell=dict(s.cell), seeds=seeds, seed_index=s.seed_index,
                       device=str(device), experiment=s.experiment,
                       artifact_dir=store.artifact_dir(s.key))
            for s, seeds in zip(specs, all_seeds)]
    entry = group_entry(specs[0].entry)
    if entry is None:
        raise ValueError(f'{specs[0].entry} has no _group variant')
    start = time.time()
    results = list(load_entry(entry)(ctxs))
    wall = time.time() - start
    if len(results) != len(specs):
        raise ValueError(f'{entry} returned {len(results)} results for {len(specs)} seeds')
    out = []
    for spec, seeds, result in zip(specs, all_seeds, results):
        if isinstance(result, BaseException):
            out.append({'key': spec.key, 'error': format_exception(result)})
            continue
        result = RunResult.coerce(result)
        result.extra = {**(result.extra or {}), 'seed_group': len(specs)}
        out.append(_output(spec, seeds, result, wall / len(specs), device))
    return out


def _output(spec, seeds, result, wall, device):
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


def execute(specs, store_root, slots, *, retries=2, worker=run_one, log=print, threads=None,
            group_seeds=False, group_worker=run_group):
    """Run every spec; returns ``{'ok': [keys], 'failed': [keys]}``.

    If ``threads`` is given, the caller has already exported the worker
    environment (``export_worker_env``) and it is left alone. Without it, this
    pool assumes it is the only one running and exports its own.

    ``group_seeds`` sends each cell's pending seeds to one worker together
    when the entry has a ``_group`` variant (see the module docstring).
    """
    specs = list(specs)
    by_key = {s.key: s for s in specs}
    store = Store(store_root)
    if threads is None:
        threads = export_worker_env(slots)
    # A queue item: (specs, attempt, grouped). A group shares one attempt count.
    queue_items = []
    if group_seeds:
        for bucket in seed_groups(specs):
            grouped = len(bucket) > 1 and group_entry(bucket[0].entry) is not None
            if len(bucket) > 1 and not grouped:
                log(f'  ! {bucket[0].entry} has no _group variant; running its seeds one by one')
            queue_items += [(tuple(bucket), 1, True)] if grouped else [((s,), 1, False) for s in bucket]
    else:
        queue_items = [((s,), 1, False) for s in specs]
    ok, failed = [], []
    total = len(specs)

    def ident(item):
        members, _, grouped = item
        return 'group:' + '+'.join(s.key for s in members) if grouped else members[0].key

    def stored(spec, attempt, out):
        store.append_run(out['record'], out['curves'])
        store.log_attempt(spec, 'ok', attempt, wall_seconds=out.get('wall_seconds'),
                          device=out.get('device'))
        ok.append(spec.key)
        headline = ', '.join(f'{k}={v:.4g}' for k, v in out['metrics'].items()
                             if isinstance(v, (int, float)))
        group = (out['record'].extra or {}).get('seed_group')
        log(f'  ✓ [{len(ok) + len(failed)}/{total}] {spec.key} {spec.label} '
            f'({out.get("wall_seconds", 0):.1f}s, {out.get("device")}'
            f'{f", group of {group}" if group else ""}) {headline}')

    while queue_items:
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        for slot in slots:
            queue.put(slot)
        notes, held = ctx.SimpleQueue(), {}
        pool = ProcessPoolExecutor(max_workers=len(slots), mp_context=ctx,
                                   initializer=_init_worker, initargs=(queue, threads, notes))
        started, inflight = {}, {}

        def submit(item):
            members, _, grouped = item
            if grouped:
                future = pool.submit(_call, group_worker, ident(item),
                                     [s.to_dict() for s in members], store_root)
            else:
                future = pool.submit(_call, worker, ident(item), members[0].to_dict(), store_root)
            inflight[future] = item
            for s in members:
                started[s.key] = time.time()

        def charge(spec, attempt, error, reason=None, quiet=False):
            """Log a failed attempt; queue the spec alone for the next, or give up."""
            store.log_attempt(spec, 'failed', attempt, error=error,
                              wall_seconds=time.time() - started[spec.key])
            if attempt <= retries:
                if not quiet:
                    log(f'  ✗ {spec.key} {spec.label}: {reason or error} -- retry {attempt}/{retries}')
                return ((spec,), attempt + 1, False)
            log(f'  ✗ {spec.key} failed after {attempt} attempts')
            failed.append(spec.key)
            return None

        for item in queue_items:
            submit(item)
        queue_items = []
        broken = False
        try:
            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                _drain(notes, held)
                for future in done:
                    item = inflight.pop(future)
                    members, attempt, grouped = item
                    try:
                        out = future.result()
                    except BrokenProcessPool:
                        inflight[future] = item   # handled with the rest below
                        raise
                    except Exception as exc:  # noqa: BLE001 -- any failure is recorded
                        for spec in members:
                            retry = charge(spec, attempt, format_exception(exc),
                                           reason=f'{type(exc).__name__}: {exc}')
                            if retry:
                                submit(retry)
                        continue
                    for spec, one in zip(members, out if grouped else [out]):
                        if 'error' in one:
                            retry = charge(spec, attempt, one['error'],
                                           reason=one['error'].strip().splitlines()[-1])
                            if retry:
                                submit(retry)
                        else:
                            stored(spec, attempt, one)
        except BrokenProcessPool as exc:
            broken = True
            running = {ident(item) for item in inflight.values()}
            dead = {k: v for k, v in _casualties(pool, notes, held).items() if k in running}
            for future, item in inflight.items():
                if dead and ident(item) not in dead:
                    queue_items.append(item)     # collateral: not its fault
                    continue
                # With no culprit identified, every in-flight run is charged,
                # so each rebuild still spends an attempt and the loop ends.
                reason = dead.get(ident(item)) or f'worker process died: {exc}'
                members, attempt, _ = item
                for spec in members:
                    retry = charge(spec, attempt, reason, quiet=not dead)
                    if retry:
                        queue_items.append(retry)
            log(f'  ! a worker died; rebuilding the pool for {len(queue_items)} tasks')
        finally:
            pool.shutdown(wait=not broken, cancel_futures=broken)

    assert set(ok) | set(failed) <= set(by_key)
    return {'ok': ok, 'failed': failed}
