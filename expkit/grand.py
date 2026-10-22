"""The grand report: every campaign of an experiment on one page set.

    expkit grand --config X [--no-pdf]

It writes into ``campaigns/<experiment>/grand/``:

* ``grand_status.csv``: per campaign, the tuned values, the tuning report's
  verdict, whether ``frozen.yaml`` is approved, and how many grid runs are stored;
* ``grand_<metric>.csv``: per campaign and facet, each level's mean with a 95%
  bootstrap CI, and each level's paired difference against the control;
* ``grand_<headline>.png``: the headline metric, campaign by cell;
* ``grand_report.tex`` (``templates/grand_report.tex.j2``), and its PDF when
  pdflatex is installed.

The columns are the levels of the control factor (the first key of
``control``); the other factors split each campaign into rows. A campaign
whose grid is not approved or not run yet still has its status row.
"""
import math
import os

import numpy as np
import pandas as pd

from expkit import campaign, frozen
from expkit.config import campaigns, cell_label, level_str, load_config, same_level
from expkit.latex import compile_pdf, environment
from expkit.record import now
from expkit.report import ALPHA, TEMPLATES, _plt, _sortable
from expkit.stats import analyze
from expkit.store import Store


def grand_dir(cfg):
    return os.path.join(campaign.campaigns_dir(), cfg.experiment, 'grand')


def _tune_verdict(cfg, doc):
    from expkit import tune_report
    path = os.path.join(tune_report.report_dir(cfg), 'warnings.json')
    if os.path.exists(path):
        import json
        with open(path) as fh:
            report = json.load(fh)
        counts = {level: sum(w['level'] == level for w in report['warnings'])
                  for level in (tune_report.BLOCK, tune_report.REVIEW)}
        return report['verdict'], counts
    return (doc or {}).get('verdict'), {}


def status(c):
    """One campaign's row: tuning, approval, grid progress, and its frame (or None)."""
    doc = frozen.read_doc(campaign.frozen_path(c))
    verdict, counts = _tune_verdict(c, doc) if c.extra.get('tune') else (None, {})
    row = {'campaign': c.campaign, 'tuned': cell_label((doc or {}).get('tuned') or {}),
           'tune_verdict': verdict or '--', 'blocks': counts.get('block', 0),
           'reviews': counts.get('review', 0),
           'approved': bool(doc and doc.get('approved')), 'stored': 0,
           'expected': len(c.cells()) * int(c.seeds), 'diverged': 0}
    try:
        _, h, _ = frozen.resolve_base(c)
    except SystemExit:
        return row, None
    frame = Store(campaign.store_dir(c)).load_frame(h)
    row.update(stored=len(frame), diverged=int(frame['diverged'].sum()) if len(frame) else 0)
    return row, (frame if len(frame) else None)


def tables(cfg, results, n_boot=2000):
    """``{metric: DataFrame}``: per campaign and facet, every level's mean and CI, and
    each non-control level's paired difference against the control."""
    col = next(iter(cfg.control))
    control = cfg.control[col]
    out = {}
    for name, (c, frame) in results.items():
        result = analyze(frame, c, n_boot=n_boot)
        facets = [f for f in c.factors if f != col and f in frame]
        summary, vs = result['summary'], result['vs_control']
        for metric, rows in summary.groupby('metric'):
            for key, group in (rows.groupby(facets, dropna=False) if facets else [((), rows)]):
                key = key if isinstance(key, tuple) else (key,)
                rec = {'campaign': name, **dict(zip(facets, key))}
                for r in group.to_dict('records'):
                    level = level_str(r[col])
                    rec.update({f'{level}_mean': r['mean'], f'{level}_ci_low': r['ci_low'],
                                f'{level}_ci_high': r['ci_high']})
                if len(vs):
                    match = vs[vs['metric'] == metric]
                    for f, v in zip(facets, key):
                        match = match[[same_level(x, v) for x in match[f]]]
                    for r in match.to_dict('records'):
                        level = level_str(r[col])
                        rec.update({f'{level}_diff': r['estimate'], f'{level}_diff_low': r['ci_low'],
                                    f'{level}_diff_high': r['ci_high'], f'{level}_p': r['p']})
                out.setdefault(metric, []).append(rec)
    levels = [level_str(v) for v in _sortable(cfg.factors[col])]
    levels = [level_str(control)] + [v for v in levels if not same_level(v, control)]
    return {m: pd.DataFrame(rows) for m, rows in out.items()}, col, levels


def headline_figure(table, levels, facets, metric, direction, path):
    """Campaigns down, cells across: the headline mean, darker is better."""
    plt = _plt()
    table = table.sort_values(['campaign', *facets])
    campaigns_ = list(dict.fromkeys(table['campaign']))
    combos = list(dict.fromkeys(tuple(r[f] for f in facets) for r in table.to_dict('records')))
    cols = [(combo, level) for combo in combos for level in levels]
    matrix = np.full((len(campaigns_), len(cols)), np.nan)
    for r in table.to_dict('records'):
        i = campaigns_.index(r['campaign'])
        combo = tuple(r[f] for f in facets)
        for level in levels:
            v = r.get(f'{level}_mean')
            if v is not None and np.isfinite(v):
                matrix[i, cols.index((combo, level))] = v
    fig, ax = plt.subplots(figsize=(max(6.2, 0.55 * len(cols) + 2), 0.38 * len(campaigns_) + 1.8))
    cmap = 'viridis' if direction == 'max' else 'viridis_r'
    im = ax.imshow(matrix, aspect='auto', cmap=cmap)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                r, g, b, _ = im.cmap(im.norm(matrix[i, j]))
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        color='black' if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else 'white')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{' '.join(map(str, combo))}\n{level}" for combo, level in cols],
                       fontsize=6, rotation=90)
    ax.set_yticks(range(len(campaigns_)))
    ax.set_yticklabels(campaigns_, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _records(table):
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in table.to_dict('records')]


def build(cfg, *, pdf=True, n_boot=2000, log=print):
    """Write the grand report for every campaign of ``cfg``'s experiment; returns the .tex path."""
    names = campaigns(cfg.path)
    rows, results = [], {}
    for name in names:
        c = load_config(cfg.path, None if name == 'default' else name)
        row, frame = status(c)
        rows.append(row)
        if frame is not None:
            results[c.campaign] = (c, frame)
    out_dir = grand_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'grand_status.csv'), index=False)

    sections, figure, col, levels, facets = [], None, None, [], []
    if results and cfg.control:
        by_metric, col, levels = tables(cfg, results, n_boot)
        facets = [f for f in cfg.factors if f != col]
        metrics = [m for m in cfg.metrics if m in by_metric] or sorted(by_metric)
        headline = cfg.headline if cfg.headline in metrics else metrics[0]
        for metric in sorted(metrics, key=lambda m: m != headline):
            table = by_metric[metric].sort_values(['campaign', *[f for f in facets
                                                                  if f in by_metric[metric]]])
            table.to_csv(os.path.join(out_dir, f'grand_{metric}.csv'), index=False)
            direction = cfg.metric_direction(metric)
            records = _records(table)
            for r in records:
                means = {lv: r.get(f'{lv}_mean') for lv in levels if r.get(f'{lv}_mean') is not None}
                r['best'] = ((min if direction == 'min' else max)(means, key=means.get)
                             if means else None)
            sections.append({'metric': metric, 'headline': metric == headline,
                             'direction': direction, 'rows': records})
        figure = f'grand_{headline}.png'
        headline_figure(by_metric[headline], levels, [f for f in facets if f in by_metric[headline]],
                        headline, cfg.metric_direction(headline), os.path.join(out_dir, figure))
    for name in os.listdir(out_dir):
        if name.endswith('.png') and name != figure:
            os.remove(os.path.join(out_dir, name))

    env = environment(TEMPLATES)
    tex = env.get_template('grand_report.tex.j2').render(
        cfg=cfg, status=rows, sections=sections, figure=figure, col=col, levels=levels,
        control=level_str(cfg.control[col]) if col else None, facets=facets, alpha=ALPHA,
        built=now(), n_results=len(results))
    path = os.path.join(out_dir, 'grand_report.tex')
    with open(path, 'w') as fh:
        fh.write(tex)
    log(f'wrote {path}: {len(results)} of {len(rows)} campaigns have grid results')
    if pdf:
        compile_pdf(path, log=log)
    return path
