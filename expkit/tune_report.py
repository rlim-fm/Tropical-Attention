"""The tuning report: what ``expkit approve`` is about to accept, and how far to trust it.

    expkit tune-report --config X [--profile P] [--no-pdf]

``expkit tune`` builds it every time a search finishes, into ``<campaign>/tune/report/``:

* ``tune_report.tex`` (``templates/tune_report.tex.j2``), and its PDF when pdflatex is installed;
* ``tune_sensitivity.png``, ``tune_cells.png`` and ``tune_curve_<name>.png``;
* ``warnings.json``: the verdict and every warning. ``frozen.yaml`` and
  ``scripts/approve_all.sh`` repeat them.

The verdict is the worst warning: a ``block`` means do not approve, a ``review``
means read the report before approving, and ``info`` never changes it. Each
check is a pure function of the search's results that returns ``[(level, message)]``.

Metrics whose name contains ``ood`` or ``test`` appear only in a sealed appendix,
after the decision. The search never consulted them; the approver should not either.

Two optional checks need to know what the searched knob does. The ``tune:``
block names the paths:

    tune:
      report:
        warmup: fishr.anneal_iters      # the knob only acts from this step on
        penalty: {metric: fishr_penalty_final, weight: fishr.lambda, against: id_val_loss}
        best_epoch: {metric: best_epoch, epochs: optim.epochs}   # without curves: picked last epoch?
"""
import json
import math
import os

import numpy as np

from expkit import campaign, frozen, tune
from expkit.config import (base_hash, cell_label, cell_matches, flatten, get_path, level_str,
                           plain, with_overrides)
from expkit.latex import TEMPLATES, compile_pdf, environment, yaml_lines
from expkit.latex import plt as _plt
from expkit.record import now
from expkit.stats import mean_ci
from expkit.store import Store

BLOCK, REVIEW, INFO = 'block', 'review', 'info'
VERDICTS = {BLOCK: 'do not approve', REVIEW: 'review', None: 'ok'}
FLAT_TOLERANCE = 1e-3       # candidate means within this fraction of each other: a flat knob
PENALTY_FRACTION = 0.01     # weight * penalty below this fraction of the loss: negligible


def report_dir(cfg, root=None):
    return os.path.join(tune.tune_dir(cfg, root), 'report')


def sealed(metric):
    """True for a metric the approver must not see before deciding (OOD or test)."""
    return any(word in metric.lower() for word in tune.FORBIDDEN_SELECT)


def verdict(warnings):
    levels = {level for level, _ in warnings}
    return VERDICTS[BLOCK if BLOCK in levels else REVIEW if REVIEW in levels else None]


# --- the search's results -----------------------------------------------------

def load(cfg, root=None):
    """``(spec, table, best, store, candidates)``; one dict per candidate, in table order."""
    spec = tune.spec_of(cfg)
    table, best = tune.summarize(cfg, root)
    store = Store(tune.store_dir(cfg, root))
    cands = []
    for i, (values, base, h, specs) in enumerate(tune.candidates(cfg)):
        cands.append({'index': i, 'values': values, 'label': cell_label(values), 'base': base,
                      'hash': h, 'frame': store.load_frame(h), 'records': store.load_records(h),
                      'expected': len(specs), 'selected': bool(table.loc[i, 'selected'])})
    return spec, table, best, store, cands


def units(frame, factors, metric):
    """``{(cell levels..., seed index): value}``: the runs of one candidate, keyed for pairing."""
    if not len(frame) or metric not in frame:
        return {}
    cols = [f for f in factors if f in frame]
    return {tuple(level_str(r[c]) for c in cols) + (int(r['seed_index']),): float(r[metric])
            for r in frame.to_dict('records')}


def paired_gap(winner, other, sign, n_boot=2000):
    """``other`` minus ``winner`` on their shared runs, signed so positive is worse."""
    from scipy import stats as sps

    keys = sorted(set(winner) & set(other))
    diffs = np.array([sign * (other[k] - winner[k]) for k in keys], float)
    diffs = diffs[np.isfinite(diffs)]
    mean, lo, hi = mean_ci(diffs, n_boot=n_boot)
    worse, better = int((diffs > 0).sum()), int((diffs < 0).sum())
    p = (float(sps.binomtest(worse, worse + better).pvalue) if worse + better else math.nan)
    return {'n': len(diffs), 'mean': mean, 'ci_low': lo, 'ci_high': hi, 'worse': worse,
            'better': better, 'equal': len(diffs) - worse - better, 'p': p,
            'identical': len(diffs) > 0 and worse + better == 0}


# --- the checks -----------------------------------------------------------------

def check_complete(table, knobs):
    out = []
    for r in table.to_dict('records'):
        label = cell_label({k: r[k] for k in knobs})
        if r['runs'] < r['expected']:
            out.append((BLOCK, f"{label}: only {r['runs']} of {r['expected']} runs are stored"))
        if r['diverged']:
            out.append((REVIEW, f"{label}: {r['diverged']} of {r['runs']} runs diverged, "
                                f"so it was not eligible"))
    return out


def check_winner(best):
    if best is None:
        return [(BLOCK, 'no candidate was selected: the search is incomplete or none is eligible')]
    return []


def check_flat(unit_maps, means):
    """Every candidate gave the same numbers: the knob changed nothing."""
    maps = [m for m in unit_maps if m]
    if len(maps) < 2:
        return []
    if all(m == maps[0] for m in maps[1:]):
        return [(BLOCK, 'every candidate produced bit-identical results on every run: the '
                        'searched knob had no effect, so the choice between them is arbitrary')]
    means = np.asarray([m for m in means if np.isfinite(m)], float)
    scale = max(np.abs(means).max(), 1e-12) if len(means) else 0
    if len(means) > 1 and (means.max() - means.min()) / scale < FLAT_TOLERANCE:
        return [(REVIEW, f'the candidate means differ by less than {FLAT_TOLERANCE:.1%}: '
                         f'the metric barely responds to the knob')]
    return []


def check_tie(gaps):
    """The runner-up's paired gap to the winner is not distinguishable from zero."""
    rivals = [g for g in gaps if not g['selected'] and g['n'] and np.isfinite(g['mean'])]
    if not rivals:
        return []
    runner = min(rivals, key=lambda g: g['mean'])
    if runner['identical']:
        return [(REVIEW, f"{runner['label']} matches the winner exactly on every shared run")]
    if not np.isfinite(runner['ci_low']):
        return [(REVIEW, f"{runner['label']} trails the winner by {runner['mean']:.3g} on "
                         f"{runner['n']} run(s): too few to tell them apart")]
    if runner['ci_low'] <= 0:
        return [(REVIEW, f"the winner is not distinguishable from {runner['label']}: the paired "
                         f"gap's 95% CI is [{runner['ci_low']:.3g}, {runner['ci_high']:.3g}]")]
    return []


def check_boundary(space, best):
    out = []
    for path, values in (space or {}).items():
        if best is None or not all(isinstance(v, (int, float)) for v in values):
            continue
        ordered = sorted(float(v) for v in values)
        chosen = float(best[path])
        if len(ordered) < 3:
            out.append((INFO, f'{path} has only {len(ordered)} values searched; an interior '
                              f'optimum cannot show'))
        elif chosen in (ordered[0], ordered[-1]):
            side = 'smallest' if chosen == ordered[0] else 'largest'
            out.append((REVIEW, f'{path}={level_str(chosen)} is the {side} value searched: '
                                f'the optimum may lie beyond it, so widen the space'))
    return out


def check_warmup(cands, path):
    """Runs that stopped before the knob became active."""
    out = []
    for c in cands:
        start = get_path(c['base'], path)
        frame = c['frame']
        if start is None or not len(frame) or frame['steps'].isna().all():
            continue
        idle = int((frame['steps'] <= float(start)).sum())
        if idle:
            level = BLOCK if idle == len(frame) else REVIEW
            out.append((level, f"{c['label']}: {idle} of {len(frame)} runs stopped at or before "
                               f"step {level_str(start)} ({path}), so the knob never acted"))
    return out


def check_penalty(cands, metric, weight, against):
    """``weight * metric`` is a negligible fraction of ``against``."""
    out = []
    for c in cands:
        w, frame = get_path(c['base'], weight), c['frame']
        if w is None or not len(frame) or metric not in frame or against not in frame:
            continue
        ratio = float(np.nanmedian(float(w) * frame[metric] / frame[against].abs()))
        if np.isfinite(ratio) and ratio < PENALTY_FRACTION:
            out.append((REVIEW, f"{c['label']}: {weight} * {metric} is {ratio:.2g} of {against} "
                                f"(median over runs); the term barely enters the loss"))
    return out


def check_convergence(winner, curves, metric, direction):
    """The winner's runs were still improving on ``metric`` when the search budget ran out."""
    if winner is None:
        return []
    series = [curves[k][metric] for k in winner['frame'].get('key', []) if k in curves
              and metric in curves[k]]
    if not series:
        return []
    if all(len(s) < 2 for s in series):
        return [(INFO, f'one {metric} value is logged per run: convergence cannot be judged')]
    pick = np.nanargmin if direction == 'min' else np.nanargmax
    late = sum(int(pick(np.asarray(s, float))) == len(s) - 1 for s in series if len(s) > 1)
    if late * 2 > len(series):
        return [(REVIEW, f'{late} of {len(series)} winner runs reached their best {metric} at the '
                         f'last logged epoch: the search budget may be too short to rank candidates')]
    return []


def check_best_epoch(winner, metric, epochs_path):
    """The winner's runs picked their last epoch: still improving when the budget ran out.

    The curve-free twin of ``check_convergence``, for stores whose curves stayed behind.
    """
    if winner is None or metric not in winner['frame']:
        return []
    epochs = get_path(winner['base'], epochs_path)
    best = winner['frame'][metric].dropna()
    if epochs is None or not len(best):
        return []
    late = int((best >= int(epochs) - 1).sum())
    if late * 2 > len(best):
        return [(REVIEW, f'{late} of {len(best)} winner runs picked their last epoch '
                         f'({metric} = {int(epochs) - 1}): still improving when the search '
                         f'budget ran out, so candidates were ranked before converging')]
    return []


def check_budget(grid_base, search_base):
    """What the search changed against the grid's own budget."""
    grid = flatten(grid_base)
    return [(INFO, f'the search ran with {k}={level_str(v)}; the grid runs with '
                   f'{k}={level_str(grid.get(k))}')
            for k, v in flatten(search_base or {}).items() if grid.get(k) != v]


def check_hygiene(records):
    out = []
    if not records:
        return out
    dirty = sum(bool(r.git_dirty) for r in records)
    if dirty:
        out.append((REVIEW, f'{dirty} of {len(records)} runs came from a dirty working tree; the '
                            f'code that produced them is not a commit'))
    shas = {r.git_sha for r in records}
    if len(shas) > 1:
        out.append((REVIEW, f'the runs span {len(shas)} commits; candidates may differ in code, '
                            f'not only in the knob'))
    torch = {(r.versions or {}).get('torch') for r in records}
    if len(torch) > 1:
        out.append((REVIEW, f'the runs used torch versions {sorted(map(str, torch))}'))
    devices = sorted({str(r.device) for r in records})
    out.append((INFO, f"devices: {', '.join(devices)}"))
    return out


# --- figures --------------------------------------------------------------------

def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def sensitivity_figure(cands, factors, metric, direction, path, n_boot=2000):
    """The metric against the candidates: one faint line per cell, the mean with a CI in bold."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(cands))
    cols = [f for f in factors if any(f in c['frame'] for c in cands)]
    cells = {}
    for i, c in enumerate(cands):
        frame = c['frame']
        if not len(frame) or metric not in frame:
            continue
        for key, group in frame.groupby(cols, dropna=False) if cols else [((), frame)]:
            key = key if isinstance(key, tuple) else (key,)
            cells.setdefault(key, {})[i] = float(group[metric].mean())
    for key, points in cells.items():
        xs = sorted(points)
        ax.plot(xs, [points[i] for i in xs], color='0.6', lw=0.8, alpha=0.6)
    stats = [mean_ci(c['frame'][metric], n_boot=n_boot) if len(c['frame']) and metric in c['frame']
             else (math.nan,) * 3 for c in cands]
    means = np.array([s[0] for s in stats])
    ax.fill_between(x, [s[1] for s in stats], [s[2] for s in stats], color='C0', alpha=0.2, lw=0)
    ax.plot(x, means, color='C0', lw=2, marker='o', ms=4, label='mean over cells and seeds')
    for i, c in enumerate(cands):
        if c['selected']:
            ax.plot([i], [means[i]], marker='*', ms=14, color='C3', ls='', label='selected')
    ax.set_xticks(x)
    ax.set_xticklabels([c['label'] for c in cands], fontsize=7, rotation=20 if len(cands) > 4 else 0)
    ax.set_ylabel(f'{metric} ({"lower" if direction == "min" else "higher"} is better)')
    _style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _varying(cands, factors):
    """The factors whose level differs between the searched cells (the rest name no cell)."""
    return [f for f in factors
            if len({level_str(v) for c in cands if f in c['frame'] for v in c['frame'][f]}) > 1]


def cell_table(cands, factors, metric, direction):
    """``(cells, matrix)``: each searched cell's seed-mean per candidate."""
    cols = _varying(cands, factors)
    labels, rows = [], {}
    for i, c in enumerate(cands):
        frame = c['frame']
        if not len(frame) or metric not in frame:
            continue
        for key, group in frame.groupby(cols, dropna=False) if cols else [((), frame)]:
            key = key if isinstance(key, tuple) else (key,)
            label = cell_label(dict(zip(cols, key))) or 'all'
            if label not in rows:
                labels.append(label)
                rows[label] = [math.nan] * len(cands)
            rows[label][i] = float(group[metric].mean())
    return labels, np.array([rows[label] for label in labels], float).reshape(len(labels), len(cands))


def ranks(matrix, direction):
    """Rank 1 = best within each row; ties share the best rank."""
    from scipy.stats import rankdata
    sign = 1 if direction == 'min' else -1
    out = np.full(matrix.shape, np.nan)
    for r, row in enumerate(matrix):
        ok = np.isfinite(row)
        if ok.any():
            out[r, ok] = rankdata(sign * row[ok], method='min')
    return out


def cells_figure(labels, matrix, rank, cands, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 0.45 * len(labels) + 1.4))
    ax.imshow(rank, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=max(len(cands), 2))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if np.isfinite(matrix[r, c]):
                ax.text(c, r, f'{matrix[r, c]:.3g}\n#{int(rank[r, c])}', ha='center', va='center',
                        fontsize=6.5)
    ax.set_xticks(range(len(cands)))
    ax.set_xticklabels([c['label'] for c in cands], fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def curve_figure(cands, curves, name, path):
    """Curve ``name`` per candidate, averaged over its cells and seeds."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    drawn, positive = 0, True
    for c in cands:
        arrays = [curves[k][name] for k in c['frame'].get('key', []) if k in curves
                  and name in curves[k] and len(curves[k][name])]
        if not arrays:
            continue
        n = min(len(a) for a in arrays)
        mean = np.mean([np.asarray(a[:n], float) for a in arrays], axis=0)
        positive &= bool(np.all(mean > 0))
        style = dict(lw=2.0, color='C3') if c['selected'] else dict(lw=1.0)
        ax.plot(np.arange(n), mean, marker='o' if n < 4 else None, ms=3, label=c['label'], **style)
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    if positive:
        ax.set_yscale('log')
    ax.set_xlabel('epoch (as logged)')
    ax.set_ylabel(name)
    _style(ax)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --- the report -----------------------------------------------------------------

def _metric_names(cands):
    skip = {'key', 'seed_index', 'diverged', 'steps', 'wall_seconds', 'frozen_hash'}
    names = []
    for c in cands:
        for col in c['frame'].columns:
            if col not in skip and col not in names and c['frame'][col].dtype.kind == 'f':
                names.append(col)
    return names


def _means(cands, metrics):
    return [{'label': c['label'], 'selected': c['selected'],
             'values': [float(c['frame'][m].mean()) if len(c['frame']) and m in c['frame']
                        else None for m in metrics]} for c in cands]


def analyze(cfg, root=None, n_boot=2000):
    """Everything the report shows, without writing anything: a dict for the template."""
    spec, table, best, store, cands = load(cfg, root)
    metric, direction = spec['metric'], spec['direction']
    sign = 1 if direction == 'min' else -1
    factors = list(cfg.factors)
    unit_maps = [units(c['frame'], factors, metric) for c in cands]
    winner = next((c for c in cands if c['selected']), None)

    rows = []
    for c, r, u in zip(cands, table.to_dict('records'), unit_maps):
        values = np.array(list(u.values()), float)
        mean, lo, hi = mean_ci(values, n_boot=n_boot)
        rows.append({'label': c['label'], 'values': c['values'], 'runs': r['runs'],
                     'expected': r['expected'], 'diverged': r['diverged'], 'mean': mean,
                     'ci_low': lo, 'ci_high': hi,
                     'sd': float(values.std(ddof=1)) if len(values) > 1 else None,
                     'min': float(values.min()) if len(values) else None,
                     'max': float(values.max()) if len(values) else None,
                     'eligible': bool(r['eligible']), 'selected': c['selected']})
    order = sorted((i for i, r in enumerate(rows) if np.isfinite(r['mean'])),
                   key=lambda i: sign * rows[i]['mean'])
    for rank, i in enumerate(order, 1):
        rows[i]['rank'] = rank
    for r in rows:
        r.setdefault('rank', None)

    gaps = []
    if winner is not None:
        w = unit_maps[winner['index']]
        for c, u in zip(cands, unit_maps):
            if not c['selected']:
                gaps.append({'label': c['label'], 'selected': False,
                             **paired_gap(w, u, sign, n_boot)})

    curves = store.load_curves([k for c in cands for k in c['frame'].get('key', [])])
    report_spec = (cfg.extra.get('tune') or {}).get('report') or {}
    warnings = (check_winner(best) + check_complete(table, list(spec['space']))
                + check_flat(unit_maps, [r['mean'] for r in rows]) + check_tie(gaps)
                + check_boundary(spec['space'], best))
    if report_spec.get('warmup'):
        warnings += check_warmup(cands, report_spec['warmup'])
    if report_spec.get('penalty'):
        p = report_spec['penalty']
        warnings += check_penalty(cands, p['metric'], p['weight'], p['against'])
    warnings += check_convergence(winner, curves, metric, direction)
    if report_spec.get('best_epoch'):
        b = report_spec['best_epoch']
        warnings += check_best_epoch(winner, b['metric'], b['epochs'])
    warnings += check_budget(cfg.base, spec['base'])
    records = [r for c in cands for r in c['records']]
    warnings += check_hygiene(records)
    order = {BLOCK: 0, REVIEW: 1, INFO: 2}
    warnings.sort(key=lambda w: order[w[0]])

    names = _metric_names(cands)
    open_metrics = [m for m in names if m != metric and not sealed(m)]
    sealed_metrics = [m for m in names if sealed(m)]
    doc = frozen.read_doc(campaign.frozen_path(cfg, root)) or {}
    yaml_values = flatten(cfg.base)
    tuned = [{'path': k, 'value': v, 'yaml': yaml_values.get(k)} for k, v in (best or {}).items()]
    return {
        'cfg': cfg, 'spec': spec, 'metric': metric, 'direction': direction, 'factors': factors,
        'knobs': list(spec['space']), 'cands': cands, 'rows': rows, 'gaps': gaps, 'best': best,
        'winner': winner, 'curves': curves, 'warnings': warnings, 'verdict': verdict(warnings),
        'tuned': tuned, 'frozen_doc': doc,
        'frozen_hash': base_hash(with_overrides(cfg.base, best), cfg.optional) if best else None,
        'cells': [cell_label(c) for c in cfg.cells() if cell_matches(c, spec['only'])],
        'open_metrics': open_metrics, 'open_means': _means(cands, open_metrics),
        'sealed_metrics': sealed_metrics, 'sealed_means': _means(cands, sealed_metrics),
        'records': records, 'store': store,
    }


def _provenance(cands):
    out = []
    for c in cands:
        recs = c['records']
        out.append({'label': c['label'], 'hash': c['hash'], 'runs': len(recs),
                    'wall': sum(float(r.wall_seconds or 0) for r in recs),
                    'shas': sorted({(r.git_sha or '')[:10] + ('+' if r.git_dirty else '')
                                    for r in recs}),
                    'torch': sorted({str((r.versions or {}).get('torch')) for r in recs})})
    return out


def build(cfg, root=None, *, pdf=True, n_boot=2000, log=print):
    """Write the tuning report; returns ``{'path', 'pdf', 'verdict', 'warnings'}``."""
    a = analyze(cfg, root, n_boot)
    out_dir = report_dir(cfg, root)
    os.makedirs(out_dir, exist_ok=True)
    keep = set()
    figures = {}
    if any(len(c['frame']) for c in a['cands']):
        figures['sensitivity'] = 'tune_sensitivity.png'
        sensitivity_figure(a['cands'], a['factors'], a['metric'], a['direction'],
                           os.path.join(out_dir, figures['sensitivity']), n_boot)
        labels, matrix = cell_table(a['cands'], a['factors'], a['metric'], a['direction'])
        rank = ranks(matrix, a['direction'])
        figures['cells'] = 'tune_cells.png'
        cells_figure(labels, matrix, rank, a['cands'], os.path.join(out_dir, figures['cells']))
        a['cell_rows'] = [{'cell': label, 'best': [a['cands'][j]['label'] for j in
                                                   range(len(a['cands'])) if rank[i, j] == 1]}
                          for i, label in enumerate(labels)]
    else:
        a['cell_rows'] = []
    keep |= set(figures.values())
    wanted = list(dict.fromkeys([*(a['cfg'].replay_curves or []), a['metric']]))
    curve_names = []
    for name in wanted:
        if sealed(name):
            continue
        fname = f'tune_curve_{name}.png'
        if curve_figure(a['cands'], a['curves'], name, os.path.join(out_dir, fname)):
            curve_names.append(name)
            keep.add(fname)
    for name in os.listdir(out_dir):
        if name.endswith('.png') and name not in keep:
            os.remove(os.path.join(out_dir, name))

    warnings = [{'level': level, 'message': message} for level, message in a['warnings']]
    env = environment(TEMPLATES)
    tex = env.get_template('tune_report.tex.j2').render(
        **{k: v for k, v in a.items() if k not in ('curves', 'records', 'store')},
        warning_rows=warnings, figures=figures, curve_names=curve_names,
        provenance=_provenance(a['cands']), built=now(), store=a['store'].root,
        flags=campaign.cli_flags(cfg, root), base_lines=yaml_lines(
            with_overrides(cfg.base, a['best'] or {})))
    path = os.path.join(out_dir, 'tune_report.tex')
    with open(path, 'w') as fh:
        fh.write(tex)
    with open(os.path.join(out_dir, 'warnings.json'), 'w') as fh:
        json.dump({'verdict': a['verdict'], 'warnings': warnings, 'built': now()}, fh, indent=1)
    log(f'wrote {path}: verdict {a["verdict"].upper()}')
    for w in warnings:
        if w['level'] != INFO:
            log(f"  [{w['level']}] {w['message']}")
    pdf_path = compile_pdf(path, log=log) if pdf else None
    return {'path': path, 'pdf': pdf_path, 'verdict': a['verdict'], 'warnings': warnings,
            'best': a['best']}


def stamp(cfg, report, root=None):
    """Write ``report``'s verdict into an unapproved ``frozen.yaml`` that holds the values it
    judged (one frozen before this report existed). Returns the path, or None if left alone."""
    path = campaign.frozen_path(cfg, root)
    doc = frozen.read_doc(path)
    if (doc is None or doc.get('approved') or report['best'] is None
            or plain(doc.get('tuned')) != plain(report['best'])):
        return None
    doc.update(report=os.path.relpath(report['pdf'] or report['path']), verdict=report['verdict'])
    return frozen.write_doc(path, doc)
