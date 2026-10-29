"""The grand report: every campaign of an experiment on one page set.

    expkit grand --config X [--no-pdf]

It writes into ``campaigns/<experiment>/grand/``:

* ``grand_status.csv``: per campaign, how many grid runs are stored and diverged;
* ``grand_<metric>.csv``: per campaign and facet, each level's mean with a 95%
  bootstrap CI, and each level's paired difference against the control;
* ``len_<campaign>.png`` with ``report.lengths``: a metric against input length,
  a panel per block and a line per control-factor level;
* ``loss_<campaign>.png``: the loss curves (``report.loss_curves``, else
  ``replay_curves``) of every block and level;
* ``grand_report.tex`` (``templates/grand_report.tex.j2``), and its PDF when
  pdflatex is installed.

The report opens with the headline metric as colored tables, one block per
combination of the non-control factors (the first across, the rest down), a row
per campaign and a column per level of the control factor (the first key of
``control``). A campaign's name links to its length figure and a cell to its
panel of the loss figure. The other metrics get the same blocks in the appendix.
A campaign whose grid has not run yet still has its status row.

The optional ``report`` block of the YAML::

    report:
      labels: {ood_len_f1: Length-OOD F1}   # metric names in headings
      lengths: {family: ood_len_f1_L, reference: id_f1, train_lengths: data.train_lengths}
      loss_curves: [train_loss, id_val_loss]
      curve_x: epoch
"""
import itertools
import math
import os
import re
import warnings

import numpy as np
import pandas as pd

from expkit import campaign, frozen
from expkit.config import campaigns, get_path, level_str, load_config, same_level
from expkit.latex import ALPHA, TEMPLATES, compile_pdf, environment
from expkit.latex import plt as _plt, sortable as _sortable
from expkit.record import now
from expkit.stats import analyze
from expkit.store import Store


def grand_dir(cfg):
    return os.path.join(campaign.campaigns_dir(), cfg.experiment, 'grand')


def status(c):
    """One campaign's row (grid progress) and its frame (or None)."""
    row = {'campaign': c.campaign, 'stored': 0, 'expected': len(c.cells()) * int(c.seeds),
           'diverged': 0}
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


def _anchor(*parts):
    """A LaTeX label or hyperlink name: letters, digits and hyphens only."""
    return '-'.join(re.sub(r'[^A-Za-z0-9]+', '', str(p)) for p in parts)


def layout(cfg, col):
    """The blocks of the grid: one per combination of the non-control factors.

    The first non-control factor runs across (its levels in YAML order), the rest
    down. Returns ``(rows, across, down)``: ``rows`` is a list of rows of cells
    (``{factor: level}``)."""
    facets = [f for f in cfg.factors if f != col]
    if not facets:
        return [[{}]], None, []
    across, down = facets[0], facets[1:]
    downs = [dict(zip(down, combo)) for combo in itertools.product(*(cfg.factors[f] for f in down))]
    return [[{across: a, **d} for a in cfg.factors[across]] for d in downs], across, down


def _match(rows, cell):
    return [r for r in rows if all(same_level(r.get(f), v) for f, v in cell.items())]


def _colour(cmap, t):
    r, g, b, _ = cmap(t)
    fg = '000000' if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else 'FFFFFF'
    return '%02X%02X%02X' % tuple(int(round(255 * x)) for x in (r, g, b)), fg


def block_grid(table, blocks, levels, direction, bounds=None, alpha=ALPHA):
    """The colored tables of one metric: per block (a combination of the facets), a row
    per campaign and a column per control-factor level.

    The colour scale is per campaign, shared by all of its blocks (so models and
    levels compare within a task), spanning at least a tenth of ``bounds``.
    A level's mean is bold where it is the best of its row, and starred where its
    paired difference against the control has p < ``alpha``."""
    plt = _plt()
    cmap = plt.get_cmap('viridis' if direction == 'max' else 'viridis_r')
    records = _records(table)
    names = list(dict.fromkeys(r['campaign'] for r in records))
    floor = 0.1 * (bounds[1] - bounds[0]) if bounds else 0.0
    scale = {}
    for name in names:
        values = [r.get(f'{lv}_mean') for r in records if r['campaign'] == name for lv in levels]
        values = [v for v in values if v is not None and np.isfinite(v)]
        if values:
            lo, hi = min(values), max(values)
            if hi - lo < floor:
                mid = (lo + hi) / 2
                lo, hi = mid - floor / 2, mid + floor / 2
            scale[name] = (lo, hi)
    out = []
    for i_row, row in enumerate(blocks):
        out_row = []
        for i_col, cell in enumerate(row):
            index = i_row * len(row) + i_col
            lines = []
            for name in names:
                hit = _match([r for r in records if r['campaign'] == name], cell)
                r = hit[0] if hit else {}
                means = {lv: r.get(f'{lv}_mean') for lv in levels}
                means = {lv: v for lv, v in means.items() if v is not None and np.isfinite(v)}
                best = (min if direction == 'min' else max)(means, key=means.get) if means else None
                cells = []
                for j, lv in enumerate(levels):
                    v = means.get(lv)
                    entry = {'anchor': _anchor('loss', name, index, j), 'text': '--',
                             'bg': None, 'fg': '000000', 'bold': False, 'star': False}
                    if v is not None:
                        lo, hi = scale[name]
                        t = 0.5 if hi <= lo else min(1.0, max(0.0, (v - lo) / (hi - lo)))
                        entry['bg'], entry['fg'] = _colour(cmap, t if direction == 'max' else 1 - t)
                        text = f'{v:.2f}'
                        entry['text'] = text[1:] if text.startswith('0.') else text
                        entry['bold'] = lv == best
                        p = r.get(f'{lv}_p')
                        entry['star'] = p is not None and p < alpha
                    cells.append(entry)
                lines.append({'campaign': name, 'anchor': _anchor('fig', 'len', name), 'cells': cells})
            out_row.append({'title': ' / '.join(level_str(cell[f]) for f in cell) or 'all',
                            'rows': lines})
        out.append(out_row)
    return out


def _runs(frame, cell, col, level):
    """The runs (row dicts) of one cell at one level of ``col``."""
    return _match(frame.to_dict('records'), {**cell, col: level})


def length_figure(frame, blocks, col, levels, control, spec, base, path, label='F1'):
    """A metric against input length: a panel per block, a line per level of ``col``.

    ``spec``: ``{family, reference, train_lengths}``. Each run stores ``<family><n>``
    for every length-OOD length ``n``; ``reference`` (the ID metric, pooled over the
    training lengths) is drawn at their geometric mean, and a red line marks the
    longest training length, past which every length is OOD."""
    family = spec['family']
    lengths = sorted(int(c[len(family):]) for c in frame.columns
                     if c.startswith(family) and c[len(family):].isdigit())
    if not lengths:
        return None
    train = get_path(base, spec['train_lengths']) if spec.get('train_lengths') else []
    train = [float(x) for x in (train if isinstance(train, (list, tuple)) else [train])]
    ref = spec.get('reference')
    ref = ref if ref in frame else None
    xs = ([float(np.exp(np.mean(np.log(train))))] if ref and train else []) + [float(n) for n in lengths]
    plt = _plt()
    n_rows, n_cols = len(blocks), len(blocks[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.4, 2.9 * n_rows + 0.6), squeeze=False,
                             sharex=True, sharey=True)
    colours = _level_colours(levels, control)
    for i, row in enumerate(blocks):
        for j, cell in enumerate(row):
            ax = axes[i][j]
            if train:
                ax.axvspan(min(train), max(train), color='0.92', lw=0)
                ax.axvline(max(train), color='#d62728', lw=1.3, ls='--')
            for lv in levels:
                runs = _runs(frame, cell, col, lv)
                if not runs:
                    continue
                cols = ([ref] if ref and train else []) + [f'{family}{n}' for n in lengths]
                ys = [np.nanmean([float(r.get(c, np.nan)) for r in runs]) for c in cols]
                ax.plot(xs, ys, marker='o', ms=3, lw=1.3, color=colours[lv],
                        label=_level_label(lv, control))
            ax.set_xscale('log')
            ax.set_title(' / '.join(level_str(cell[f]) for f in cell), fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel('input length', fontsize=9)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelsize=8)
            _despine(ax)
    ticks = []
    for n in sorted({*train[:1], *train[-1:], *map(float, lengths)}):
        if not ticks or np.log10(n / ticks[-1]) >= 0.15:     # no two labels overlap
            ticks.append(n)
    ax = axes[0][0]
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=len(names), frameon=False, fontsize=8,
               title=col, title_fontsize=8, handlelength=1.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def loss_figure(frame, curves, blocks, col, levels, names, path, x_label='epoch'):
    """The loss curves of every cell: a row per block, a column per level of ``col``.

    Each panel draws the seed mean (and range) of every curve in ``names``; a grey
    dotted line marks the median selected epoch. Returns ``(path, anchors)``, with
    ``anchors`` the top-left corner of each panel as figure fractions, keyed
    ``(block index, level index)``, for hyperlink targets over the image."""
    flat = [cell for row in blocks for cell in row]
    plt = _plt()
    fig, axes = plt.subplots(len(flat), len(levels), squeeze=False, sharex=True, sharey='row',
                             figsize=(10.4, 1.05 * len(flat) + 0.7))
    drawn = False
    for i, cell in enumerate(flat):
        positive = True
        for j, lv in enumerate(levels):
            ax = axes[i][j]
            runs = _runs(frame, cell, col, lv)
            for k, name in enumerate(names):
                arrays = [np.asarray(curves[r['key']][name], float) for r in runs
                          if name in curves.get(r['key'], {})]
                if not arrays:
                    continue
                n = max(len(a) for a in arrays)
                stack = np.full((len(arrays), n), np.nan)
                for s, a in enumerate(arrays):
                    stack[s, :len(a)] = a
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mean = np.nanmean(stack, axis=0)
                    lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
                positive &= bool(np.all(stack[np.isfinite(stack)] > 0))
                x = np.arange(1, n + 1)
                ax.plot(x, mean, lw=0.9, color=f'C{k}', label=name)
                ax.fill_between(x, lo, hi, color=f'C{k}', alpha=0.2, lw=0)
                drawn = True
            best = [r.get('best_epoch') for r in runs if r.get('best_epoch') is not None]
            best = [b for b in best if np.isfinite(b)]
            if best:
                ax.axvline(float(np.median(best)) + 1, color='0.4', lw=0.7, ls=':')
            if not runs:
                ax.text(0.5, 0.5, 'no runs', transform=ax.transAxes, ha='center', va='center',
                        fontsize=7, color='0.5')
            if i == 0:
                ax.set_title(f'{col}={level_str(lv)}', fontsize=7)
            if j == 0:
                ax.set_ylabel(' / '.join(level_str(cell[f]) for f in cell), fontsize=7)
            if i == len(flat) - 1:
                ax.set_xlabel(x_label, fontsize=7)
            ax.tick_params(labelsize=6)
            _despine(ax)
        if positive and drawn:
            axes[i][0].set_yscale('log')
    if not drawn:
        plt.close(fig)
        return None, {}
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False,
                       fontsize=8)
            break
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_size_inches()[1]), h_pad=0.4, w_pad=0.3)
    fig.savefig(path, dpi=150)
    anchors = {}
    for i in range(len(flat)):
        for j in range(len(levels)):
            box = axes[i][j].get_position()
            anchors[(i, j)] = (box.x0, min(1.0, box.y1 + 0.02))
    plt.close(fig)
    return path, anchors


def _level_label(level, control):
    tail = ' (control)' if control is not None and same_level(level, control) else ''
    return f'{level_str(level)}{tail}'


def _level_colours(levels, control):
    """The control in black, the other levels along plasma in their order."""
    plt = _plt()
    others = [lv for lv in levels if not same_level(lv, control)]
    cmap = plt.get_cmap('plasma')
    out = {lv: cmap(0.88 * i / max(1, len(others) - 1)) for i, lv in enumerate(others)}
    out.update({lv: 'black' for lv in levels if same_level(lv, control)})
    return out


def _despine(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


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

    opts = cfg.extra.get('report') or {}
    labels = opts.get('labels') or {}
    sections, tasks, figures, col, levels, blocks = [], [], set(), None, [], [[{}]]
    if results and cfg.control:
        by_metric, col, _ = tables(cfg, results, n_boot)
        levels = [level_str(v) for v in _sortable(cfg.factors[col])]
        control = cfg.control[col]
        blocks, _, _ = layout(cfg, col)
        metrics = [m for m in cfg.metrics if m in by_metric] or sorted(by_metric)
        headline = cfg.headline if cfg.headline in metrics else metrics[0]
        for metric in sorted(metrics, key=lambda m: m != headline):
            facets = [f for f in cfg.factors if f != col and f in by_metric[metric]]
            table = by_metric[metric].sort_values(['campaign', *facets])
            table.to_csv(os.path.join(out_dir, f'grand_{metric}.csv'), index=False)
            direction = cfg.metric_direction(metric)
            sections.append({'metric': metric, 'label': labels.get(metric, metric),
                             'headline': metric == headline, 'direction': direction,
                             'grid': block_grid(table, blocks, levels, direction,
                                                (cfg.metrics.get(metric) or {}).get('bounds'))})
        spec = opts.get('lengths')
        loss_names = opts.get('loss_curves') or cfg.replay_curves or []
        for name, (c, frame) in sorted(results.items()):     # the grid's row order
            task = {'campaign': name, 'label': _anchor('fig', 'len', name), 'lengths': None,
                    'loss': None, 'anchors': []}
            if spec:
                base, _, _ = frozen.resolve_base(c)
                fname = f'len_{_anchor(name)}.png'
                if length_figure(frame, blocks, col, levels, control, spec, base,
                                 os.path.join(out_dir, fname), labels.get(spec['family'], 'F1')):
                    task['lengths'] = fname
            curves = Store(campaign.store_dir(c)).load_curves(list(frame['key'])) if loss_names else {}
            if curves:
                fname = f'loss_{_anchor(name)}.png'
                path, anchors = loss_figure(frame, curves, blocks, col, levels, loss_names,
                                            os.path.join(out_dir, fname), opts.get('curve_x', 'epoch'))
                if path:
                    task['loss'] = fname
                    task['anchors'] = [{'name': _anchor('loss', name, i, j), 'x': x, 'y': y}
                                       for (i, j), (x, y) in anchors.items()]
            figures.update(f for f in (task['lengths'], task['loss']) if f)
            tasks.append(task)
    for name in os.listdir(out_dir):
        if name.endswith('.png') and name not in figures:
            os.remove(os.path.join(out_dir, name))

    env = environment(TEMPLATES)
    tex = env.get_template('grand_report.tex.j2').render(
        cfg=cfg, status=rows, sections=sections, tasks=tasks, col=col, levels=levels,
        control=level_str(cfg.control[col]) if col else None, alpha=ALPHA,
        loss_names=opts.get('loss_curves') or cfg.replay_curves or [],
        lengths=opts.get('lengths'), built=now(), n_results=len(results))
    path = os.path.join(out_dir, 'grand_report.tex')
    with open(path, 'w') as fh:
        fh.write(tex)
    log(f'wrote {path}: {len(results)} of {len(rows)} campaigns have grid results')
    if pdf:
        compile_pdf(path, log=log)
    return path
