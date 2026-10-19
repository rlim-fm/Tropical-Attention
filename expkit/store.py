"""The results store: append-only JSONL, one file per kind of record.

Layout of one store directory (a campaign's ``results/``)::

    runs.jsonl        one RunRecord per completed run: cell, seeds, metrics, code
                      version -- everything but the base config
    configs.jsonl     one line per frozen hash: the base config its runs share
    curves.jsonl      one line per run: every curve, as lists (the bulk of the bytes)
    registry.jsonl    one line per attempt: ok / failed, with the traceback
    artifacts/<key>/  files a run chose to write (RunContext.artifact_dir)

A grid holds thousands of runs. One small file per run clogs git and the
filesystem. One append-only file per kind diffs as appended lines.
Suggested git policy: track ``runs``, ``configs`` and ``registry``, which are
what the statistics read, and ignore ``curves.jsonl`` and ``artifacts/`` once
they grow. Float repr round-trips exactly, so the curves in the store are
exactly the ones the run produced, which is what replay compares against.

A run counts as complete when its key has a line in ``runs.jsonl``. Its curves
line is written first, so a crash in between leaves an orphan curve, never a
record without one. A re-run (``--force``) appends again, and the last line for
a key wins. A line torn by a crash mid-write is skipped by readers and cut off
by the next append. **One writer, no lock:** the orchestrating process is the
only one that appends. An ``flock`` on an NFS store can block forever when the
lock service stalls, which hangs every job with no error.
"""
import json
import math
import os
import traceback

import numpy as np

from expkit.record import RunRecord, now


def read_jsonl(path):
    """Every complete line of ``path`` as a dict; a torn last line is skipped."""
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        text = fh.read()
    if text and not text.endswith('\n'):
        text = text[:text.rfind('\n') + 1]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def append_jsonl(path, entries):
    """Append one line per entry, and fsync. Single writer: no lock."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = ''.join(json.dumps(e, separators=(',', ':')) + '\n' for e in entries).encode()
    with open(path, 'a+b') as fh:
        _drop_torn_tail(fh)
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def rewrite_jsonl(path, entries):
    """Replace ``path`` with ``entries``, atomically (a tmp file, then a rename)."""
    tmp = f'{path}.tmp-{os.getpid()}'
    with open(tmp, 'w') as fh:
        fh.writelines(json.dumps(e, separators=(',', ':')) + '\n' for e in entries)
    os.replace(tmp, path)


def _drop_torn_tail(fh):
    size = fh.seek(0, os.SEEK_END)
    if size == 0:
        return
    fh.seek(size - 1)
    if fh.read(1) == b'\n':
        return
    fh.seek(0)
    fh.truncate(fh.read().rfind(b'\n') + 1)


def format_exception(exc):
    return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class Store:
    FILES = ('runs.jsonl', 'configs.jsonl', 'curves.jsonl', 'registry.jsonl')

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.runs_path = os.path.join(self.root, 'runs.jsonl')
        self.configs_path = os.path.join(self.root, 'configs.jsonl')
        self.curves_path = os.path.join(self.root, 'curves.jsonl')
        self.log_path = os.path.join(self.root, 'registry.jsonl')
        self.artifacts_dir = os.path.join(self.root, 'artifacts')
        self._cache = {}

    def artifact_dir(self, key):
        return os.path.join(self.artifacts_dir, key)

    def _read(self, path):
        """``read_jsonl``, re-read only when the file has changed since the last call."""
        try:
            st = os.stat(path)
            stamp = (st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            return []
        hit = self._cache.get(path)
        if hit is None or hit[0] != stamp:
            hit = self._cache[path] = (stamp, read_jsonl(path))
        return hit[1]

    def runs(self):
        """``{key: line}`` of ``runs.jsonl``; a later line for a key wins."""
        return {line['key']: line for line in self._read(self.runs_path)}

    def configs(self):
        return {line['frozen_hash']: line['base'] for line in self._read(self.configs_path)}

    def is_complete(self, key):
        return key in self.runs()

    def pending(self, specs, force=False):
        runs = self.runs()
        return list(specs) if force else [s for s in specs if s.key not in runs]

    def append_run(self, record, curves):
        """Persist a finished run: its config (once per hash), its curves, then its record."""
        if record.frozen_hash not in self.configs():
            append_jsonl(self.configs_path, [{'frozen_hash': record.frozen_hash,
                                              'base': record.base}])
        append_jsonl(self.curves_path, [{'key': record.key, **_listed(curves)}])
        append_jsonl(self.runs_path, [record.to_line()])

    def attempts(self):
        return self._read(self.log_path)

    def log_attempt(self, spec, status, attempt, error=None, wall_seconds=None, device=None):
        append_jsonl(self.log_path, [{
            'key': spec.key, 'status': status, 'attempt': attempt, 'at': now(),
            'cell': spec.cell, 'seed_index': spec.seed_index, 'device': device,
            'wall_seconds': wall_seconds, 'error': error}])

    def failures(self):
        """The latest attempt per key that never succeeded."""
        latest = {}
        for a in self.attempts():
            latest[a['key']] = a
        runs = self.runs()
        return [a for k, a in latest.items() if a['status'] == 'failed' and k not in runs]

    def load_records(self, frozen_hash=None):
        configs = self.configs()
        return [RunRecord.from_line(line, configs.get(line['frozen_hash']))
                for _, line in sorted(self.runs().items())
                if frozen_hash is None or line['frozen_hash'] == frozen_hash]

    def load_curves(self, keys=None):
        """``{key: {name: array}}`` from ``curves.jsonl``, for ``keys`` or every run."""
        wanted = None if keys is None else set(keys)
        out = {}
        for line in self._read(self.curves_path):
            if wanted is None or line['key'] in wanted:
                out[line['key']] = {n: np.asarray(v) for n, v in line.items() if n != 'key'}
        return out

    def load_frame(self, frozen_hash=None):
        """One row per completed run: its factor levels, seed, status and every metric.

        Factor columns carry the factors' own names; a metric with the same name
        as a factor is refused rather than silently overwriting it.
        """
        import pandas as pd

        rows = []
        for _, r in sorted(self.runs().items()):
            if frozen_hash is not None and r['frozen_hash'] != frozen_hash:
                continue
            row = {'key': r['key'], **r['cell'], 'seed_index': r['seed_index'],
                   'diverged': r['diverged'], 'steps': r.get('steps'),
                   'wall_seconds': r.get('wall_seconds'), 'frozen_hash': r['frozen_hash']}
            clash = set(r['metrics']) & set(row)
            if clash:
                raise ValueError(f'metric name(s) {sorted(clash)} collide with a column')
            row.update({k: _float(v) for k, v in r['metrics'].items()})
            rows.append(row)
        return pd.DataFrame(rows)


def _listed(curves):
    return {name: np.asarray(v).tolist() for name, v in (curves or {}).items()}


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
