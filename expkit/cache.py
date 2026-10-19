"""A keyed cache of finished computations (tuning fits, probes, anything deterministic).

The run store skips a run whose key already has a line. This is the same idea
for work that is not a grid run, for example the many candidate fits of a
hyperparameter search. A result is keyed by everything that determines it,
and by nothing else: never the device, the worker count or a display label.
Then widening a search refits only what is new, and a job killed mid-search
resumes where it stopped.

    cache = ResultCache('campaigns/toy/default/cache/search')
    identity = {'base': base, 'epochs': 50, 'seed_index': s, 'lr': lr}
    result = cache.get(identity)
    if result is None:
        result = fit(...)
        cache.put(identity, result)       # result: a JSON-safe dict

The key cannot see your code. After a change that alters results without
changing their inputs, bump ``version`` (or construct with ``fresh=True``,
which ignores what is on disk but still records and reuses new results).
Like the run store, the process that calls ``put`` must be the only writer.
"""
import os

from expkit.config import config_hash, plain
from expkit.record import now
from expkit.store import append_jsonl, read_jsonl


class ResultCache:
    def __init__(self, root, version=1, fresh=False):
        self.root = os.path.abspath(root)
        self.path = os.path.join(self.root, 'results.jsonl')
        self.version = version
        self._lines = {} if fresh else None

    def key(self, identity):
        return config_hash({'v': self.version, **identity}, length=16)

    def _index(self):
        if self._lines is None:
            self._lines = {line['key']: line for line in read_jsonl(self.path)}
        return self._lines

    def get(self, identity):
        line = self._index().get(self.key(identity))
        return None if line is None else line['result']

    def put(self, identity, result):
        line = {'key': self.key(identity), 'created': now(), 'result': plain(result)}
        append_jsonl(self.path, [line])
        self._index()[line['key']] = line
        return result

    def __len__(self):
        return len(self._index())
