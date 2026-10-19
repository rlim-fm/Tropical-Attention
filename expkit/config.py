"""An experiment's configuration: its base settings, its factor grid, its campaigns.

An experiment is one YAML file::

    experiment: toy                 # names the store; default: the file's stem
    entry: expkit.examples.toy:run  # module:function called once per run
    global_seed: 0
    seeds: 5                        # seed indices 0 .. seeds-1, paired across cells
    base: {...}                     # every setting the runs share; hashed
    optional: {model.dropout: 0.0}  # dotted path -> default: pruned from the hash at it
    factors:                        # the full factorial of these levels
      lr: [0.001, 0.01]
      batch_size: [32, 128]
    exclude: [{lr: 0.01, batch_size: 32}]
    control: {lr: 0.001}            # the reference level the report compares against
    metrics: {mse: {direction: min, bounds: [0, null]}}
    headline: mse
    replay_curves: [train_loss]     # what `expkit replay` must reproduce exactly
    require_frozen: false           # true: refuse to run without an approved frozen file
    axes:                           # campaigns: one level of each axis, crossed
      noise:  {clean: {}, noisy: {base: {data: {noise: 0.1}}}}
      method: {plain: {}, wd: {base: {optim: {weight_decay: 1e-4}}}}
    profiles:                       # other named overrides
      dry_run: {seeds: 2, base: {epochs: 2}}

A profile is either a ``profiles`` key or a campaign name, one level per axis
joined by ``/`` (``noisy/wd``), whose blocks are merged in axis order. With no
profile the first level of every axis applies. Every override block is deep-merged
over the whole document, so it may set ``base``, ``factors``, ``seeds`` or
anything else.

**The hash is the identity of every stored run.** ``frozen_hash`` hashes the
canonical JSON of ``base`` with each ``optional`` setting removed while it is at
its default. So a knob added later does not move the hash, and the runs already
stored stay valid. Setting it away from its default does move the hash, which is
right: those runs are different ones.
"""
import copy
import hashlib
import itertools
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

DEFAULT_CAMPAIGN = 'default'
CAMPAIGN_SEP = '/'
RESERVED_KEYS = ('profiles', 'axes')


class Loader(yaml.SafeLoader):
    """SafeLoader that reads ``1e-3`` as a float.

    PyYAML follows YAML 1.1, whose float pattern requires a dot, so ``lr: 1e-3``
    would load as the *string* '1e-3' (and hash differently from 0.001).
    """


Loader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                   |\.[0-9_]+(?:[eE][-+][0-9]+)?
                   |[-+]?\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$''', re.X),
    list('-+0123456789.'))


def yaml_load(stream):
    return yaml.load(stream, Loader=Loader)


# --- canonical form and hashing ------------------------------------------------

def plain(obj):
    """YAML/JSON-safe copy: tuples become lists and infinities the string 'inf'."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, float) and math.isinf(obj):
        return 'inf' if obj > 0 else '-inf'
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        return plain(obj.item())          # numpy scalars
    return obj


def canonical_json(obj) -> str:
    return json.dumps(plain(obj), sort_keys=True, separators=(',', ':'))


def config_hash(obj, length=12) -> str:
    return hashlib.sha1(canonical_json(obj).encode()).hexdigest()[:length]


def deep_merge(a, b):
    """``b`` merged over ``a``: dicts recursively, everything else replaced."""
    out = copy.deepcopy(a) if a is not None else {}
    for key, value in (b or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_path(d, path, default=None):
    for part in path.split('.'):
        if not isinstance(d, dict) or part not in d:
            return default
        d = d[part]
    return d


def set_path(d, path, value):
    parts = path.split('.')
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def pop_path(d, path):
    parts = path.split('.')
    for part in parts[:-1]:
        if not isinstance(d, dict) or part not in d:
            return
        d = d[part]
    if isinstance(d, dict):
        d.pop(parts[-1], None)


def flatten(d, prefix=''):
    """``{'a': {'b': 1}}`` -> ``{'a.b': 1}``."""
    out = {}
    for key, value in d.items():
        if isinstance(value, dict) and value:
            out.update(flatten(value, f'{prefix}{key}.'))
        else:
            out[f'{prefix}{key}'] = value
    return out


def with_overrides(base, overrides):
    """A copy of ``base`` with ``{'optim.lr': 0.01}``-style or nested overrides applied."""
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and '.' not in key:
            out[key] = deep_merge(out.get(key, {}), value)
        else:
            set_path(out, key, copy.deepcopy(value))
    return out


def parse_assignments(items):
    """``['optim.lr=1e-3', 'model.act=relu']`` -> ``{'optim.lr': 0.001, ...}``, YAML-typed."""
    out = {}
    for item in items or []:
        key, sep, text = item.partition('=')
        if not sep or not key:
            raise ValueError(f'expected KEY=VALUE, got {item!r}')
        out[key.strip()] = yaml_load(text)
    return out


def canonical_base(base, optional=None):
    """``base`` in the form ``frozen_hash`` hashes: optional settings at their default left out."""
    out = plain(copy.deepcopy(base))
    for path, default in (optional or {}).items():
        if get_path(out, path, default) == plain(default):
            pop_path(out, path)
    return out


def base_hash(base, optional=None) -> str:
    return config_hash(canonical_base(base, optional))


# --- cells ------------------------------------------------------------------------

def level_str(value) -> str:
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, float):
        return 'inf' if math.isinf(value) and value > 0 else f'{value:g}'
    return str(value)


def cell_label(cell) -> str:
    """``lr=0.001 batch_size=32``: a cell's name in logs, tables and figures."""
    return ' '.join(f'{k}={level_str(v)}' for k, v in cell.items())


def coerce(value):
    """A level as a comparable value: 'on'/'true' -> True, '0.01' -> 0.01, 'inf' -> inf."""
    if isinstance(value, str):
        text = value.strip()
        low = text.lower()
        if low in ('on', 'true', 'yes'):
            return True
        if low in ('off', 'false', 'no'):
            return False
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def same_level(a, b) -> bool:
    a, b = coerce(a), coerce(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def cell_matches(cell, partial) -> bool:
    """True if every entry of ``partial`` agrees with ``cell``."""
    return all(name in cell and same_level(cell[name], value) for name, value in partial.items())


def factor_grid(factors, exclude=()):
    """The full factorial of ``{name: [levels]}``, in a stable order, minus ``exclude``."""
    if not factors:
        return [{}]
    names = list(factors)
    cells = [dict(zip(names, values)) for values in
             itertools.product(*(list(factors[n]) for n in names))]
    return [c for c in cells if not any(cell_matches(c, e) for e in exclude or ())]


# --- the experiment file --------------------------------------------------------

@dataclass
class ExperimentConfig:
    experiment: str
    entry: str
    path: str
    global_seed: int = 0
    seeds: int = 1
    base: dict = field(default_factory=dict)
    optional: dict = field(default_factory=dict)
    factors: dict = field(default_factory=dict)
    exclude: list = field(default_factory=list)
    control: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    headline: Optional[str] = None
    replay_curves: Optional[list] = None
    require_frozen: bool = False
    profile: Optional[str] = None
    campaign: str = DEFAULT_CAMPAIGN
    extra: dict = field(default_factory=dict)       # every other top-level key, untouched

    def cells(self):
        return factor_grid(self.factors, self.exclude)

    def seed_indices(self):
        return list(range(int(self.seeds)))

    def base_hash(self, base=None):
        return base_hash(self.base if base is None else base, self.optional)

    def metric_direction(self, metric):
        return (self.metrics.get(metric) or {}).get('direction', 'min')

    def metric_bounds(self, metric):
        bounds = (self.metrics.get(metric) or {}).get('bounds')
        return tuple(bounds) if bounds else None


def read_raw(path):
    with open(path) as fh:
        return yaml_load(fh) or {}


def campaign_names(raw):
    """Every campaign the ``axes`` block crosses, first levels first; ``[]`` without axes."""
    axes = raw.get('axes') or {}
    if not axes:
        return []
    return [CAMPAIGN_SEP.join(levels) for levels in
            itertools.product(*(list(levels) for levels in axes.values()))]


def default_campaign(raw):
    names = campaign_names(raw)
    return names[0] if names else DEFAULT_CAMPAIGN


def profile_overrides(raw, profile):
    """What ``profile`` merges over the document, and the campaign name it runs under."""
    axes = raw.get('axes') or {}
    if profile is None:
        profile = default_campaign(raw) if axes else None
        if profile is None:
            return {}, DEFAULT_CAMPAIGN
    profiles = raw.get('profiles') or {}
    if profile in profiles:
        return copy.deepcopy(profiles[profile] or {}), profile
    if profile == DEFAULT_CAMPAIGN and not axes:
        return {}, DEFAULT_CAMPAIGN
    levels = profile.split(CAMPAIGN_SEP)
    if axes and len(levels) == len(axes):
        merged = {}
        for (axis, blocks), level in zip(axes.items(), levels):
            if level not in (blocks or {}):
                break
            merged = deep_merge(merged, blocks[level] or {})
        else:
            return merged, profile
    raise KeyError(f'no profile {profile!r}; profiles: {sorted(profiles)}, '
                   f'campaigns: {campaign_names(raw)}')


KNOWN = ('experiment', 'entry', 'global_seed', 'seeds', 'base', 'optional', 'factors',
         'exclude', 'control', 'metrics', 'headline', 'replay_curves', 'require_frozen')


def load_config(path, profile=None) -> ExperimentConfig:
    raw = read_raw(path)
    overrides, campaign = profile_overrides(raw, profile)
    doc = deep_merge({k: v for k, v in raw.items() if k not in RESERVED_KEYS}, overrides)
    if 'entry' not in doc:
        raise ValueError(f'{path}: an experiment needs an `entry: module:function`')
    control = doc.get('control') or {}
    unknown = set(control) - set(doc.get('factors') or {})
    if unknown:
        raise ValueError(f'{path}: control names factors that do not exist: {sorted(unknown)}')
    return ExperimentConfig(
        experiment=doc.get('experiment') or os.path.splitext(os.path.basename(path))[0],
        entry=doc['entry'], path=os.path.abspath(path),
        global_seed=int(doc.get('global_seed', 0)), seeds=int(doc.get('seeds', 1)),
        base=doc.get('base') or {}, optional=doc.get('optional') or {},
        factors=doc.get('factors') or {}, exclude=doc.get('exclude') or [],
        control=control, metrics=doc.get('metrics') or {}, headline=doc.get('headline'),
        replay_curves=doc.get('replay_curves'),
        require_frozen=bool(doc.get('require_frozen', False)),
        profile=profile, campaign=campaign,
        extra={k: v for k, v in doc.items() if k not in KNOWN})


def campaigns(path):
    """Every campaign of the experiment at ``path``; ``['default']`` without axes."""
    return campaign_names(read_raw(path)) or [DEFAULT_CAMPAIGN]


def profile_base(path, profile):
    """The ``base`` block a profile sets, before it is merged."""
    return (profile_overrides(read_raw(path), profile)[0].get('base')) or {}
