"""The results report: CSVs, figures and a LaTeX document for one campaign.

    expkit report --config X [--profile P] [--pdf]

It writes into ``<campaign root>/report/``:

* ``stats_<analysis>.csv`` for every analysis ``expkit.stats.analyze`` ran;
* ``metrics.png``: one panel per metric, the mean with a bootstrap 95% CI
  against the trend factor (or the control's, or the first), one line per
  combination of the rest;
* ``curve_<name>.png``: the seed-mean of a stored curve, one panel per level of
  that factor, when the store has curves;
* ``effective_weight.png`` and ``stats_effective_weight.csv`` with
  ``report.effective_weight``;
* ``lengths.png`` and ``stats_lengths.csv`` with ``report.length_curve``;
* ``report.tex`` (``templates/report.tex.j2``), and ``report.pdf`` with ``--pdf``.

The main text is short: the setup, the metric grid, a table of the trend in
every metric per combination of the other factors, and the figures the
``report`` block asks for. Every table sits in the appendix, and each figure's
caption names the tables behind it. A build removes the figures that the new
.tex no longer cites.

The optional ``report`` block of the YAML (campaign overrides may set it too)::

    report:
      color_by: model                       # a hue per level ...
      shade_by: {factor: head, order: [a, b]}   # ... darkest first within a hue
      colors: {tropical: '#1b7f3b'}         # hues; tab10 for levels left out
      grid: [[acc, loss], [ood_acc, null]]  # rows of panels in metrics.png
      labels: {ood_acc: 'OOD accuracy'}     # panel and table names
      setup: '...'                          # LaTeX, rendered with base, cfg, opts
      main_curves: [train_loss]             # curve figures in the main text
      effective_weight: {curve: penalty, scale: lam, relative_to: train_loss, last: 10}
      length_curve: {families: [acc_L], relative_to: data.train_lengths,
                     levels: [min, median, max], reference: {acc_L: acc}}

The template is the place to make the report your own. ``build`` hands it the
whole ``analyze`` output as lists of row dicts.
"""
import math
import os

import numpy as np

from expkit import campaign
from expkit.config import cell_label, get_path, level_str, same_level
from expkit.latex import compile_pdf, environment
from expkit.record import PACKAGE, now
from expkit.stats import analyze, facets, mean_ci

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


def _mix(color, t):
    """``color`` darkened toward black (t < 0) or lightened toward white (t > 0)."""
    from matplotlib.colors import to_rgb
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1 - rgb) * t if t > 0 else rgb * (1 + t))


class Style:
    """One colour, marker and label per facet, the same in every figure.

    ``report.color_by`` names a factor whose levels each get a hue
    (``report.colors``, else the tab10 cycle); ``report.shade_by`` (a factor, or
    ``{factor, order}`` darkest first) varies the lightness and marker within a
    hue. Without them every facet gets its own tab10 colour.
    """

    MARKERS = 'osD^v'

    def __init__(self, opts, frame, keys, factor_levels):
        self.keys = list(keys)
        hue = opts.get('color_by')
        self.hue = hue if hue in self.keys else None
        shade = opts.get('shade_by') or {}
        shade = {'factor': shade} if isinstance(shade, str) else shade
        self.shade = shade.get('factor') if shade.get('factor') in self.keys else None

        def order(factor, given=None):
            levels = [level_str(v) for v in (given or factor_levels.get(factor) or [])]
            seen = [level_str(v) for v in _sortable(frame[factor].unique())]
            return levels + [v for v in seen if v not in levels]

        self.hues = order(self.hue) if self.hue else []
        self.shades = order(self.shade, shade.get('order')) if self.shade else []
        cycle = [f'C{i}' for i in range(10)]
        given = {str(k): v for k, v in (opts.get('colors') or {}).items()}
        self.palette = {h: given.get(h, cycle[i % 10]) for i, h in enumerate(self.hues)}
        self.cells = sorted((cell for cell, _ in facets(frame, self.keys)), key=self.sort_key) \
            if self.keys else [{}]

    def sort_key(self, cell):
        hue = self.hues.index(level_str(cell[self.hue])) if self.hue else 0
        shade = self.shades.index(level_str(cell[self.shade])) if self.shade else 0
        return hue, shade, cell_label(cell)

    def label(self, cell):
        named = [level_str(cell[f]) for f in (self.hue, self.shade) if f]
        rest = {k: v for k, v in cell.items() if k not in (self.hue, self.shade)}
        return ' / '.join(named + ([cell_label(rest)] if rest else [])) or None

    def __call__(self, cell):
        if not self.hue:
            labels = [cell_label(c) for c in self.cells]
            i = labels.index(cell_label(cell)) if cell_label(cell) in labels else 0
            return {'color': f'C{i % 10}', 'marker': 'o', 'label': cell_label(cell) or None}
        color = self.palette[level_str(cell[self.hue])]
        i, n = (self.shades.index(level_str(cell[self.shade])), len(self.shades)) if self.shade else (0, 1)
        t = 0.0 if n == 1 else -0.2 + 0.7 * i / (n - 1)     # darkest first, palest at +0.5
        return {'color': _mix(color, t), 'marker': self.MARKERS[i % len(self.MARKERS)],
                'label': self.label(cell)}


def _despine(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _shared_legend(fig, axes, style=None):
    """One legend above every panel, from the first panel that has labelled lines.

    With a hue and a shade factor, one column per hue (matplotlib fills columns
    first, and ``Style.cells`` runs hue-major), so each column is one model.
    """
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            wide = fig.get_size_inches()[0] >= 10
            per_hue = style is not None and style.hue and style.shade and len(labels) == len(style.cells)
            ncol = len(style.hues) if per_hue and not wide else min(len(labels), 6 if wide else 3)
            fig.legend(handles, labels, loc='upper center', ncol=ncol, frameon=False,
                       fontsize=8, bbox_to_anchor=(0.5, 1.0))
            rows = -(-len(labels) // ncol)
            return 0.035 * rows * 6.0 / fig.get_size_inches()[1]
    return 0.0


def _draw_metric(ax, summary, metric, x_factor, others, style, *, bands=True):
    rows = summary[summary['metric'] == metric]
    levels = _sortable(rows[x_factor].unique())
    pos = {level_str(v): i for i, v in enumerate(levels)}
    groups = dict((cell_label(cell), group) for cell, group in facets(rows, others))
    for cell in style.cells:
        group = groups.get(cell_label(cell))
        if group is None:
            continue
        group = group.copy()
        group['_x'] = [pos[level_str(v)] for v in group[x_factor]]
        group = group.sort_values('_x')
        s = style(cell)
        ax.plot(group['_x'], group['mean'], marker=s['marker'], ms=3.5, lw=1.3,
                color=s['color'], label=s['label'])
        if bands:
            ax.fill_between(group['_x'], group['ci_low'], group['ci_high'],
                            color=s['color'], alpha=0.12, lw=0)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([level_str(v) for v in levels], fontsize=7)
    values = rows['mean'].to_numpy(float)
    if np.all(values[np.isfinite(values)] > 0) and np.nanmax(values) / max(np.nanmin(values), 1e-300) > 30:
        ax.set_yscale('log')
    _despine(ax)


def metric_grid(summary, grid, x_factor, others, style, path, labels=None, directions=None):
    """The mean and CI of each metric against ``x_factor``, one panel per metric.

    ``grid`` is a list of rows of metric names (None leaves a panel empty). The
    x axis is ordinal (levels evenly spaced) so a log ladder, a categorical
    factor and an 'inf' control all draw the same way.
    """
    plt = _plt()
    labels, directions = labels or {}, directions or {}
    n_rows, n_cols = len(grid), max(len(r) for r in grid)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.3 * n_rows + 0.5),
                             squeeze=False)
    for r, row in enumerate(grid):
        for c in range(n_cols):
            ax = axes[r][c]
            metric = row[c] if c < len(row) else None
            if metric is None:
                ax.axis('off')
                continue
            _draw_metric(ax, summary, metric, x_factor, others, style)
            arrow = '\u2193' if directions.get(metric) == 'min' else '\u2191'
            ax.set_title(f'{labels.get(metric, metric)} {arrow}', fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel(x_factor, fontsize=8)
            ax.tick_params(axis='y', labelsize=7)
    top = _shared_legend(fig, axes, style)
    fig.tight_layout(rect=(0, 0, 1, 1 - top))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def curve_grid(frame, curves, name, x_factor, others, style, path, x_label='index (as logged)'):
    """The seed-mean of curve ``name``: one panel per level of ``x_factor``, one line
    per facet of the other factors, on a shared (log, when positive) y axis."""
    plt = _plt()
    panels = []
    for level, group in facets(frame, [x_factor] if x_factor else []):
        lines = []
        for cell, sub in facets(group, others):
            arrays = [curves[k][name] for k in sub['key'] if k in curves and name in curves[k]]
            if arrays:
                n = min(len(a) for a in arrays)
                lines.append((cell, np.mean([np.asarray(a[:n], float) for a in arrays], axis=0)))
        if lines:
            panels.append((level, lines))
    if not panels:
        return None
    n_cols = min(len(panels), 4)
    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.2 * n_rows + 0.5),
                             squeeze=False, sharex=True, sharey=True)
    positive = all(np.all(m > 0) for _, lines in panels for _, m in lines)
    for i, ax in enumerate(np.ravel(axes)):
        if i >= len(panels):
            ax.axis('off')
            continue
        level, lines = panels[i]
        drawn = dict((cell_label(c), m) for c, m in lines)
        for cell in style.cells:
            mean = drawn.get(cell_label(cell))
            if mean is not None:
                s = style(cell)
                ax.plot(np.arange(len(mean)), mean, lw=1.1, color=s['color'], label=s['label'])
        if positive:
            ax.set_yscale('log')
        ax.set_title(cell_label(level) or name, fontsize=9)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel(x_label, fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=7)
        _despine(ax)
    top = _shared_legend(fig, axes, style)
    fig.tight_layout(rect=(0, 0, 1, 1 - top))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def effective_weight(frame, curves, spec, x_factor, others, style, path):
    """How much a weighted penalty weighs against the loss at the end of training.

    ``spec``: ``{curve, scale, relative_to, last}``. Per run, the ratio is
    ``scale-level * mean(curve[-last:]) / mean(relative_to[-last:])``, where
    ``scale`` is a factor (a regularization weight). Draws the seed geometric
    mean (with the seed range) against ``x_factor`` and returns the per-cell table,
    or ``(None, rows)`` when nothing could be computed.
    """
    import pandas as pd
    name, scale, ref = spec['curve'], spec['scale'], spec['relative_to']
    last = int(spec.get('last', 10))
    rows = []
    for _, run in frame.iterrows():
        c = curves.get(run['key'], {})
        try:
            weight = float(run[scale])
        except (KeyError, TypeError, ValueError):
            continue
        if name not in c or ref not in c or weight <= 0:
            continue
        pen = float(np.mean(np.asarray(c[name], float)[-last:]))
        loss = float(np.mean(np.asarray(c[ref], float)[-last:]))
        if loss > 0 and np.isfinite(pen):
            rows.append({**{f: run[f] for f in [x_factor] + list(others)}, 'ratio': weight * pen / loss})
    if not rows:
        return None, []
    df = pd.DataFrame(rows)
    table = []
    for cell, group in facets(df, list(others) + [x_factor]):
        r = group['ratio'].to_numpy(float)
        r = r[r > 0]
        if len(r):
            table.append({**cell, 'gmean': float(np.exp(np.mean(np.log(r)))),
                          'low': float(r.min()), 'high': float(r.max()), 'n': len(r)})
    tab = pd.DataFrame(table)
    plt = _plt()
    levels = _sortable(tab[x_factor].unique())
    pos = {level_str(v): i for i, v in enumerate(levels)}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    groups = dict((cell_label(cell), g) for cell, g in facets(tab, others))
    for cell in style.cells:
        g = groups.get(cell_label(cell))
        if g is None:
            continue
        g = g.assign(_x=[pos[level_str(v)] for v in g[x_factor]]).sort_values('_x')
        s = style(cell)
        ax.plot(g['_x'], g['gmean'], marker=s['marker'], ms=3.5, lw=1.2, color=s['color'], label=s['label'])
        ax.fill_between(g['_x'], g['low'], g['high'], color=s['color'], alpha=0.12, lw=0)
    ax.axhline(1.0, color='0.4', lw=0.8, ls='--')
    ax.set_yscale('log')
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([level_str(v) for v in levels], fontsize=7)
    ax.set_xlabel(x_factor, fontsize=8)
    ax.set_ylabel(f'{scale} x {name} / {ref}', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    _despine(ax)
    top = _shared_legend(fig, [ax], style)
    fig.tight_layout(rect=(0, 0, 1, 1 - top))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, _records(tab)


def length_figure(frame, spec, base, factor, others, style, path, labels=None, n_boot=2000):
    """How a metric falls off with distance from the training lengths.

    ``spec``: ``{families, relative_to, levels, reference}``. Each family is a
    metric prefix whose runs store ``<family><length>``; x is that length over
    ``relative_to`` (a number, or a dotted path into ``base`` whose largest value
    is used), on a log axis. One panel per level of ``factor`` in ``levels``
    ('min', 'median', 'max' or a level), one row per family; ``reference``
    ({family: metric}) is drawn at x = 1. Returns ``(path, rows)`` or ``(None, [])``.
    """
    labels = labels or {}
    families = spec.get('families') or []
    scale = spec.get('relative_to', 1)
    if isinstance(scale, str):
        scale = get_path(base, scale)
    scale = float(max(scale) if isinstance(scale, (list, tuple)) else scale or 1)
    present = _sortable(frame[factor].unique())
    positive = [v for v in present if not isinstance(v, str) and float(v) > 0] or present

    def pick(want):
        if want == 'min':
            return positive[0]
        if want == 'max':
            return positive[-1]
        if want == 'median':
            return positive[(len(positive) - 1) // 2]
        hit = [v for v in present if same_level(v, want)]
        return hit[0] if hit else None

    levels = []
    for want in spec.get('levels') or ['min', 'median', 'max']:
        v = pick(want)
        if v is not None and not any(same_level(v, w) for w in levels):
            levels.append(v)
    columns = {}
    for fam in families:
        found = []
        for col in frame.columns:
            tail = col[len(fam):] if col.startswith(fam) else ''
            if tail.isdigit():
                found.append((int(tail), col))
        columns[fam] = sorted(found)
    families = [f for f in families if columns[f]]
    if not families or not levels:
        return None, []
    reference = spec.get('reference') or {}
    rows = []
    for level in levels:
        at = frame[frame[factor].map(lambda v: same_level(v, level)).astype(bool)]
        for cell, group in facets(at, others):
            for fam in families:
                points = [(1.0, reference[fam])] if reference.get(fam) in group else []
                points += [(length / scale, col) for length, col in columns[fam]]
                for x, col in points:
                    mean, low, high = mean_ci(group[col], n_boot=n_boot)
                    rows.append({factor: level, **cell, 'family': fam, 'x': x,
                                 'metric': col, 'mean': mean, 'ci_low': low, 'ci_high': high})
    plt = _plt()
    fig, axes = plt.subplots(len(families), len(levels), squeeze=False, sharex=True, sharey='row',
                             figsize=(3.6 * len(levels), 2.4 * len(families) + 0.5))
    for r, fam in enumerate(families):
        for c, level in enumerate(levels):
            ax = axes[r][c]
            for cell in style.cells:
                pts = [p for p in rows if p['family'] == fam and same_level(p[factor], level)
                       and all(level_str(p[k]) == level_str(cell[k]) for k in others)]
                if not pts:
                    continue
                pts.sort(key=lambda p: p['x'])
                s = style(cell)
                x = [p['x'] for p in pts]
                ax.plot(x, [p['mean'] for p in pts], marker=s['marker'], ms=3.5, lw=1.2,
                        color=s['color'], label=s['label'])
                ax.fill_between(x, [p['ci_low'] for p in pts], [p['ci_high'] for p in pts],
                                color=s['color'], alpha=0.12, lw=0)
            ax.set_xscale('log')
            ax.set_title(f'{factor}={level_str(level)}', fontsize=9)
            if c == 0:
                ax.set_ylabel(labels.get(fam, fam), fontsize=8)
            if r == len(families) - 1:
                ax.set_xlabel('test length / longest training length', fontsize=8)
            ax.tick_params(labelsize=7)
            _despine(ax)
    top = _shared_legend(fig, axes, style)
    fig.tight_layout(rect=(0, 0, 1, 1 - top))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, rows


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
    import pandas as pd
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

    opts = cfg.extra.get('report') or {}
    style = Style(opts, frame, others, cfg.factors)
    labels = opts.get('labels') or {}
    grid = [[m if m in metrics else None for m in row] for row in opts.get('grid') or []]
    grid = [row for row in grid if any(row)] or [metrics[i:i + 2] for i in range(0, len(metrics), 2)]
    figures = []
    if x_factor and metrics:
        metric_grid(result['summary'], grid, x_factor, others, style,
                    os.path.join(out_dir, 'metrics.png'), labels,
                    {m: cfg.metric_direction(m) for m in metrics})
        figures.append('metrics.png')
    curve_names, weight = [], None
    curves = store.load_curves(list(frame['key']))
    if curves:
        wanted = cfg.replay_curves or sorted({n for c in curves.values() for n in c})[:3]
        for name in wanted:
            fname = f'curve_{name}.png'
            if curve_grid(frame, curves, name, x_factor, others, style, os.path.join(out_dir, fname),
                          opts.get('curve_x', 'index (as logged)')):
                curve_names.append(name)
                figures.append(fname)
        spec = opts.get('effective_weight')
        if spec and x_factor:
            path, table = effective_weight(frame, curves, spec, x_factor, others, style,
                                           os.path.join(out_dir, 'effective_weight.png'))
            if path:
                figures.append('effective_weight.png')
                weight = {**spec, 'rows': table, 'keys': others + [x_factor]}
                pd.DataFrame(table).to_csv(os.path.join(out_dir, 'stats_effective_weight.csv'),
                                           index=False)
    main_curves = [c for c in opts.get('main_curves') or [] if c in curve_names]
    lengths = None
    if opts.get('length_curve') and x_factor:
        spec = opts['length_curve']
        path, table = length_figure(frame, spec, base, x_factor, others, style,
                                    os.path.join(out_dir, 'lengths.png'), labels, n_boot)
        if path:
            figures.append('lengths.png')
            pd.DataFrame(table).to_csv(os.path.join(out_dir, 'stats_lengths.csv'), index=False)
            xs = sorted({r['x'] for r in table})
            lengths = {**spec, 'xs': xs, 'reference': spec.get('reference') or {}}

    def rows(name, metric=None):
        t = result[name]
        if not len(t):
            return []
        return _records(t[t['metric'] == metric] if metric else t)

    sections = [{'metric': m, 'headline': m == headline, 'label': labels.get(m, m),
                 'direction': cfg.metric_direction(m),
                 'summary': rows('summary', m), 'vs_control': rows('vs_control', m),
                 'trend': rows('trend', m), 'paired': rows('paired', m), 'ols': rows('ols', m),
                 'path': next(iter(rows('paths', m)), {})}
                for m in metrics]
    order = [m for row in grid for m in row if m] + metrics
    sections.sort(key=lambda s: order.index(s['metric']))
    trend_keys = [f for f in others if any(f in r for s in sections for r in s['trend'])]
    trend_rows = []
    for cell in style.cells:
        row = {'cell': cell, 'label': style.label(cell) or 'all', 'values': []}
        for s in sections:
            hit = [r for r in s['trend'] if all(level_str(r.get(f)) == level_str(cell[f]) for f in trend_keys)]
            row['values'].append(hit[0] if hit else None)
        trend_rows.append(row)
    env = environment(TEMPLATES)
    setup = opts.get('setup')
    if setup:
        setup = env.from_string(setup).render(base=base, cfg=cfg, opts=opts)
    tex = env.get_template('report.tex.j2').render(
        cfg=cfg, base=base, frozen_hash=frozen_hash, factors=factors, x_factor=x_factor,
        control=cfg.control, sections=sections, curves=curve_names, alpha=ALPHA,
        headline=headline, grid=grid, trend_rows=trend_rows, setup=setup, weight=weight,
        main_curves=main_curves, has_grid='metrics.png' in figures, lengths=lengths,
        trend_spec=trend_spec,
        n_runs=len(frame), n_diverged=int(frame['diverged'].sum()),
        n_seeds=int(frame['seed_index'].nunique()), built=now(),
        store=store.root, failures=len(store.failures()),
        base_lines=_yaml_lines(base))
    path = os.path.join(out_dir, 'report.tex')
    with open(path, 'w') as fh:
        fh.write(tex)
    _prune(out_dir, set(figures))
    log(f'wrote {path}')
    if pdf:
        compile_pdf(path, log=log)
    return path


def _yaml_lines(base):
    import yaml
    return yaml.safe_dump(base, sort_keys=False, default_flow_style=False).splitlines()


def report_dir(cfg, root=None):
    return campaign.report_dir(cfg, root)
