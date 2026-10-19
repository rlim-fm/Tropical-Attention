"""What a run is (``RunSpec``) and what it produced (``RunRecord``).

``run_key`` hashes the experiment name, the frozen base config's hash, the
cell and the seed index, and nothing else. Re-running an identical cell lands
on the same key. Changing the frozen configuration moves every key.
"""
import datetime
import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Optional

from expkit.config import canonical_json, cell_label, cell_matches, same_level

PACKAGE = os.path.dirname(os.path.abspath(__file__))
# The repository expkit was pasted into: the directory holding the package.
REPO = os.environ.get('EXPKIT_REPO') or os.path.dirname(PACKAGE)


def run_key(experiment, frozen_hash, cell, seed_index) -> str:
    identity = {'experiment': experiment, 'frozen': frozen_hash, 'cell': cell,
                'seed': int(seed_index)}
    return hashlib.sha1(canonical_json(identity).encode()).hexdigest()[:12]


def parse_selector(text):
    """``'lr=0.1,seed=1'`` -> ``{'lr': '0.1', 'seed': '1'}``."""
    out = {}
    for part in filter(None, (text or '').split(',')):
        name, _, value = part.partition('=')
        if not value:
            raise SystemExit(f'--only expects NAME=VALUE pairs, got {part!r}')
        out[name.strip()] = value.strip()
    return out


def selector_matches(selector, cell, seed_index) -> bool:
    """True if every entry agrees with the run; ``seed`` names the seed index."""
    rest = {k: v for k, v in selector.items() if k != 'seed'}
    unknown = set(rest) - set(cell)
    if unknown:
        raise ValueError(f'unknown selector(s) {sorted(unknown)}; factors: {sorted(cell)}, seed')
    if 'seed' in selector and not same_level(selector['seed'], seed_index):
        return False
    return cell_matches(cell, rest)


@dataclass
class RunSpec:
    experiment: str
    entry: str
    frozen_hash: str
    base: dict
    cell: dict
    seed_index: int
    global_seed: int
    seed_streams: Optional[list] = None

    @property
    def key(self):
        return run_key(self.experiment, self.frozen_hash, self.cell, self.seed_index)

    @property
    def label(self):
        return f'{cell_label(self.cell)}  seed {self.seed_index}'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def matches(self, selector) -> bool:
        return selector_matches(selector, self.cell, self.seed_index)


def code_version(repo=REPO):
    """(git sha, dirty) of the code a run is trained with."""
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=repo, capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return 'unknown', True


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def library_versions():
    """The versions a run's numbers depend on (only those that are installed)."""
    out = {}
    for name in ('torch', 'numpy', 'jax', 'sklearn'):
        try:
            module = __import__(name)
        except ImportError:
            continue
        out[name] = getattr(module, '__version__', 'unknown')
    return out


@dataclass
class RunRecord:
    key: str
    experiment: str
    entry: str
    cell: dict
    seed_index: int
    seeds: dict
    global_seed: int
    frozen_hash: str
    base: dict
    metrics: dict
    steps: Optional[int]
    wall_seconds: float
    diverged: bool
    device: str
    versions: dict
    git_sha: str
    git_dirty: bool
    created: str = field(default_factory=now)
    extra: dict = field(default_factory=dict)
    seed_streams: Optional[list] = None

    def to_line(self):
        """The record as one ``runs.jsonl`` line, with ``base`` left out.

        Every run under one frozen hash shares its base config, so the store
        keeps it once, in ``configs.jsonl``. :meth:`from_line` puts it back.
        """
        d = asdict(self)
        del d['base']
        return d

    @classmethod
    def from_line(cls, d, base):
        return cls(**d, base=base)

    def spec(self):
        return RunSpec(experiment=self.experiment, entry=self.entry,
                       frozen_hash=self.frozen_hash, base=self.base, cell=self.cell,
                       seed_index=self.seed_index, global_seed=self.global_seed,
                       seed_streams=self.seed_streams)
