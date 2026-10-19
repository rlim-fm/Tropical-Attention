"""A grid search over base settings, selected by an in-distribution metric.

``expkit tune`` runs every candidate of a small grid over ``base`` settings, on
seeds of its own, into a store of its own. It picks the best candidate by one
metric and writes it into the campaign's ``frozen.yaml``, **unapproved**, with
the table that justifies it as evidence. Someone reads the table and runs
``expkit approve``. Only then does ``expkit run`` spend the grid's budget::

    tune:
      global_seed: 1000            # tuning seeds, disjoint from the grid's
      seeds: 2                     # seed indices 0 .. seeds-1 under that global seed
      only: {objective: fishr}     # the cells the search trains (default: every cell)
      base: {optim: {epochs: 20}}  # a reduced budget for the search only; never frozen
      space:                       # dotted base path -> candidate values (full factorial)
        fishr.lambda: [10.0, 100.0, 1000.0]
      select: {metric: val_loss, direction: min}

The block sits beside ``base`` in the experiment YAML, and profiles and campaign
axes merge over it like everything else.

Rules the search keeps:

* **Selection never sees test data.** A ``select.metric`` whose name contains
  ``ood`` or ``test`` is refused. Select on a validation metric of the training
  distribution.
* **Tuning seeds are not grid seeds.** ``tune.global_seed`` must differ from
  the experiment's, so the grid's results are not the ones the values were
  chosen on.
* **One value per campaign.** Every cell of the grid runs with the chosen
  values, so cells stay comparable and paired. A candidate's score is the
  mean of the metric over every ``only`` cell and tuning seed. A candidate
  with a diverged or missing run is not eligible.
* **Resumable.** Each candidate has its own base hash, so its runs have their
  own keys in ``<campaign>/tune/results/``. A killed search resumes, and
  widening the space trains only what is new.
"""
import math
import os

from expkit import campaign, frozen
from expkit.config import base_hash, cell_matches, factor_grid, plain, with_overrides
from expkit.record import RunSpec
from expkit.store import Store

FORBIDDEN_SELECT = ('ood', 'test')


def tune_dir(cfg, root=None):
    return os.path.join(campaign.campaign_root(cfg, root), 'tune')


def store_dir(cfg, root=None):
    return os.path.join(tune_dir(cfg, root), 'results')


def evidence_path(cfg, root=None):
    return os.path.join(tune_dir(cfg, root), 'tune.csv')


def spec_of(cfg):
    """The validated ``tune:`` block of ``cfg``."""
    spec = cfg.extra.get('tune')
    if not spec:
        raise SystemExit(f'{cfg.path}: no `tune:` block; nothing to search')
    space = spec.get('space') or {}
    if not space or not all(isinstance(v, list) and v for v in space.values()):
        raise SystemExit('tune.space must map dotted base paths to non-empty lists of values')
    select = spec.get('select') or {}
    metric, direction = select.get('metric'), select.get('direction', 'min')
    if not metric:
        raise SystemExit('tune.select.metric is required')
    if any(word in metric.lower() for word in FORBIDDEN_SELECT):
        raise SystemExit(f'tune.select.metric {metric!r} looks like a test or OOD metric; '
                         f'select on a validation metric of the training distribution')
    if direction not in ('min', 'max'):
        raise SystemExit(f'tune.select.direction must be min or max, got {direction!r}')
    seed = int(spec.get('global_seed', cfg.global_seed + 1000))
    if seed == cfg.global_seed:
        raise SystemExit('tune.global_seed equals the grid global_seed; '
                         'tuning must not run on the grid seeds')
    return {'space': space, 'metric': metric, 'direction': direction, 'global_seed': seed,
            'seeds': int(spec.get('seeds', 1)), 'only': spec.get('only') or {},
            'base': spec.get('base') or {}}


def candidates(cfg):
    """``[(values, base, frozen_hash, specs)]``, one per candidate of the space."""
    spec = spec_of(cfg)
    search_base = with_overrides(cfg.base, spec['base'])
    cells = [c for c in cfg.cells() if cell_matches(c, spec['only'])]
    if not cells:
        raise SystemExit(f"tune.only {spec['only']} matches no cell of the grid")
    streams = cfg.extra.get('seed_streams')
    out = []
    for values in factor_grid(spec['space']):
        base = with_overrides(search_base, values)
        h = base_hash(base, cfg.optional)
        specs = [RunSpec(experiment=cfg.experiment, entry=cfg.entry, frozen_hash=h, base=base,
                         cell=cell, seed_index=s, global_seed=spec['global_seed'],
                         seed_streams=list(streams) if streams else None)
                 for cell in cells for s in range(spec['seeds'])]
        out.append((values, base, h, specs))
    return out


def all_specs(cfg):
    return [s for _, _, _, specs in candidates(cfg) for s in specs]


def summarize(cfg, root=None):
    """The evidence table (a DataFrame, one row per candidate) and the best candidate's values.

    The best is None while any candidate's runs are missing.
    """
    import pandas as pd

    spec = spec_of(cfg)
    store = Store(store_dir(cfg, root))
    metric, sign = spec['metric'], (1 if spec['direction'] == 'min' else -1)
    rows, complete = [], True
    for values, _, h, specs in candidates(cfg):
        df = store.load_frame(h)
        done = len(df)
        diverged = int(df['diverged'].sum()) if done else 0
        scores = df[metric].astype(float) if done and metric in df else pd.Series(dtype=float)
        eligible = done == len(specs) and diverged == 0 and scores.notna().all() and len(scores)
        complete &= done == len(specs)
        rows.append({**values, 'runs': done, 'expected': len(specs), 'diverged': diverged,
                     f'{metric}_mean': float(scores.mean()) if len(scores) else math.nan,
                     f'{metric}_sd': float(scores.std(ddof=1)) if len(scores) > 1 else math.nan,
                     'eligible': bool(eligible), 'frozen_hash': h})
    table = pd.DataFrame(rows)
    best = None
    pool = table[table['eligible']]
    if complete and len(pool):
        row = pool.loc[(sign * pool[f'{metric}_mean']).idxmin()]
        best = {path: _native(row[path]) for path in spec['space']}
    table['selected'] = [best is not None and all(_same(r[p], best[p]) for p in spec['space'])
                         for _, r in table.iterrows()]
    return table, best


def finalize(cfg, root=None, log=print):
    """Write the evidence table, and the best candidate into an unapproved ``frozen.yaml``.

    Returns the frozen path, or None while runs are missing. A frozen file that
    is already approved with the same values is left alone.
    """
    table, best = summarize(cfg, root)
    path = evidence_path(cfg, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False)
    log(table.drop(columns=['frozen_hash']).to_string(index=False))
    if best is None:
        log(f'not frozen: the search is incomplete or no candidate is eligible ({path})')
        return None
    doc = frozen.read_doc(campaign.frozen_path(cfg, root))
    new_hash = base_hash(with_overrides(cfg.base, best), cfg.optional)
    if (doc and doc.get('approved') and plain(doc.get('tuned')) == plain(best)
            and doc.get('frozen_hash') == new_hash):
        log(f'unchanged: {campaign.frozen_path(cfg, root)} is approved with {best}')
        return campaign.frozen_path(cfg, root)
    fpath, _ = frozen.freeze(cfg, tuned=best, evidence=os.path.relpath(path), root=root)
    log(f'wrote {fpath} with {best} (unapproved). Read {os.path.relpath(path)}, then:\n'
        f'    expkit approve {campaign.cli_flags(cfg, root)}')
    return fpath


def _native(value):
    return value.item() if hasattr(value, 'item') else value


def _same(a, b):
    try:
        return math.isclose(float(a), float(b))
    except (TypeError, ValueError):
        return a == b
