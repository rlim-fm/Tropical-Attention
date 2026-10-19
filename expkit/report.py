"""The results report: CSVs, figures and a LaTeX document for one campaign.

    expkit report --config X [--profile P] [--pdf]

It writes into ``<campaign root>/report/``:

* ``stats_<analysis>.csv`` for every analysis ``expkit.stats.analyze`` ran;
* ``metric_<metric>.png``: mean with a bootstrap 95% CI against the first
  factor (or the trend factor), one line per combination of the rest;
* ``curve_<name>.png``: the seed-mean of a stored curve per cell, when the
  store has curves;
* ``report.tex`` (``templates/report.tex.j2``), and ``report.pdf`` with ``--pdf``.

Key results come first, on the headline metric. Every figure and table has a
caption and a ``\\label``, and the prose cites them with ``\\autoref``. A build
removes the figures that the new .tex no longer cites.

The template is the place to make the report your own. ``build`` hands it the
whole ``analyze`` output as lists of row dicts.
"""
import math
import os

import numpy as np

from expkit import campaign
from expkit.config import cell_label, level_str
from expkit.latex import compile_pdf, environment
from expkit.record import PACKAGE, now
from expkit.stats import analyze, facets

TEMPLATES = os.path.join(PACKAGE, 'templates')
ALPHA = 0.05


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _sortable(values):
    def key(v):
        try:
            return (0, float(v), '')
        except (TypeError, ValueError):
            return (1, 0.0, str(v))
    return sorted(values, key=key)


def metric_figure(summary, metric, x_factor, others, path, direction='min'):
    """Mean and CI of ``metric`` against ``x_factor``, one line per facet of ``others``.

    The x axis is ordinal (levels evenly spaced) so a log ladder, a categorical
    factor and an 'inf' control all draw the same way.
    """
    plt = _plt()
    rows = summary[summary['metric'] == metric]
    levels = _sortable(rows[x_factor].unique())
    pos = {level_str(v): i for i, v in enumerate(levels)}
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for facet, group in facets(rows, others):
        group = group.copy()
        group['_x'] = [pos[level_str(v)] for v in group[x_factor]]
        group = group.sort_values('_x')
        label = cell_label(facet) or None
        line, = ax.plot(group['_x'], group['mean'], marker='o', ms=3.5, lw=1.2, label=label)
        ax.fill_between(group['_x'], group['ci_low'], group['ci_high'],
                        color=line.get_color(), alpha=0.15, lw=0)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([level_str(v) for v in levels], fontsize=8)
    ax.set_xlabel(x_factor)
    ax.set_ylabel(f'{metric} ({"lower" if direction == "min" else "higher"} is better)')
    values = rows['mean'].to_numpy(float)
    if np.all(values[np.isfinite(values)] > 0) and np.nanmax(values) / max(np.nanmin(values), 1e-300) > 30:
        ax.set_yscale('log')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if others:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def curve_figure(frame, curves, name, factors, path):
    """The seed-mean of curve ``name`` per cell, on a log y axis when it is positive."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    drawn = 0
    positive = True
    for cell, group in facets(frame, factors):
        arrays = [curves[k][name] for k in group['key'] if k in curves and name in curves[k]]
        if not arrays:
            continue
        n = min(len(a) for a in arrays)
        mean = np.mean([np.asarray(a[:n], float) for a in arrays], axis=0)
        positive &= bool(np.all(mean > 0))
        ax.plot(np.arange(n), mean, lw=1.0, label=cell_label(cell))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    if positive:
        ax.set_yscale('log')
    ax.set_xlabel('index (as logged)')
    ax.set_ylabel(name)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if drawn <= 12:
        ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _records(frame):
    if frame is None or not len(frame):
        return []
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in frame.to_dict('records')]


def _prune(out_dir, keep):
    for name in os.listdir(out_dir):
        if name.endswith('.png') and name not in keep:
            os.remove(os.path.join(out_dir, name))


def build(cfg, store, frozen_hash, base, out_dir, *, pdf=False, n_boot=2000, log=print):
    """Write the report for the runs under ``frozen_hash``; returns the .tex path."""
    frame = store.load_frame(frozen_hash)
    if not len(frame):
        raise SystemExit(f'no runs under frozen hash {frozen_hash} in {store.root}')
    os.makedirs(out_dir, exist_ok=True)
    result = analyze(frame, cfg, n_boot=n_boot)
    for name, table in result.items():
        if len(table):
            table.to_csv(os.path.join(out_dir, f'stats_{name}.csv'), index=False)

    factors = [f for f in cfg.factors if f in frame]
    metrics = list(result['paths']['metric']) if len(result['paths']) else []
    headline = cfg.headline if cfg.headline in metrics else (metrics[0] if metrics else None)
    trend_spec = (cfg.extra.get('analysis') or {}).get('trend') or {}
    x_factor = trend_spec.get('factor') or (next(iter(cfg.control)) if cfg.control else None) \
        or (factors[0] if factors else None)
    others = [f for f in factors if f != x_factor]

    figures = {}
    if x_factor:
        for metric in metrics:
            name = f'metric_{metric}.png'
            metric_figure(result['summary'], metric, x_factor, others,
                          os.path.join(out_dir, name), cfg.metric_direction(metric))
            figures[metric] = name
    curve_names = []
    curves = store.load_curves(list(frame['key']))
    if curves:
        wanted = cfg.replay_curves or sorted({n for c in curves.values() for n in c})[:3]
        for name in wanted:
            path = curve_figure(frame, curves, name, factors,
                                os.path.join(out_dir, f'curve_{name}.png'))
            if path:
                curve_names.append(name)

    def rows(name, metric=None):
        t = result[name]
        if not len(t):
            return []
        return _records(t[t['metric'] == metric] if metric else t)

    sections = [{'metric': m, 'headline': m == headline, 'figure': figures.get(m),
                 'direction': cfg.metric_direction(m),
                 'summary': rows('summary', m), 'vs_control': rows('vs_control', m),
                 'trend': rows('trend', m), 'paired': rows('paired', m), 'ols': rows('ols', m),
                 'path': next(iter(rows('paths', m)), {})}
                for m in metrics]
    sections.sort(key=lambda s: not s['headline'])
    env = environment(TEMPLATES)
    tex = env.get_template('report.tex.j2').render(
        cfg=cfg, base=base, frozen_hash=frozen_hash, factors=factors, x_factor=x_factor,
        control=cfg.control, sections=sections, curves=curve_names, alpha=ALPHA,
        n_runs=len(frame), n_diverged=int(frame['diverged'].sum()),
        n_seeds=int(frame['seed_index'].nunique()), built=now(),
        store=store.root, failures=len(store.failures()),
        base_lines=_yaml_lines(base))
    path = os.path.join(out_dir, 'report.tex')
    with open(path, 'w') as fh:
        fh.write(tex)
    _prune(out_dir, set(figures.values()) | {f'curve_{n}.png' for n in curve_names})
    log(f'wrote {path}')
    if pdf:
        compile_pdf(path, log=log)
    return path


def _yaml_lines(base):
    import yaml
    return yaml.safe_dump(base, sort_keys=False, default_flow_style=False).splitlines()


def report_dir(cfg, root=None):
    return campaign.report_dir(cfg, root)
