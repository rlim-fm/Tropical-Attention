"""Inferential statistics over a factorial grid with paired seeds.

Every analysis takes the results frame (``Store.load_frame``) and returns a
DataFrame with one row per comparison. Each row carries an effect size with a
confidence interval, not just a p-value, and a ``test_path`` column that
records which test produced it.

The seeds are paired across cells: a seed index fixes the data and the
initialization (``expkit.seeding``). That pairing is what makes the
signed-rank and paired tests valid.

Test paths. A metric that piles up at a floor or ceiling (accuracy near 1,
R^2 near 1, ...) violates the normal-error assumptions of the parametric
tests. So when more than ``BOUND_FRACTION`` of a metric's values lie within
``BOUND_EPS`` of one of its bounds, ``analyze`` switches that metric to the
rank-based counterparts:

    analysis          parametric                       rank-based
    trend             OLS slope on x, t-test           Spearman rho, seed-bootstrap CI
    (companion)       Jonckheere-Terpstra ordered trend, reported on both paths
    vs control        Dunnett (each level vs control)  Wilcoxon signed-rank by seed, Holm
    paired a - b      paired t-test by seed, Holm      Wilcoxon signed-rank by seed, Holm
    OLS term          statsmodels formula              same model on ranks of y

The analyses ``analyze`` runs come from the experiment YAML:

    control: {lr: 0.001}                 # vs_control on factor lr, faceted by the rest
    analysis:
      trend: {factor: lr, transform: log10}
      paired: [{factor: augment, a: true, b: false}]
      ols: [{formula: 'y ~ np.log10(lr) * batch_size', term: 'np.log10(lr):batch_size'}]
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from expkit.config import same_level

BOUND_FRACTION = 0.2
BOUND_EPS = 0.01
DEFAULT_BOOT = 2000
DUNNETT_SEED = 0
TRANSFORMS = {None: lambda x: x, 'none': lambda x: x, 'log10': np.log10, 'log': np.log,
              'log2': np.log2}


# --- helpers ----------------------------------------------------------------

def near_bound(values, bounds):
    """True if more than BOUND_FRACTION of ``values`` sit within BOUND_EPS of a bound."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not bounds:
        return False
    lo, hi = bounds
    close = np.zeros(len(values), dtype=bool)
    if lo is not None:
        close |= values <= lo + BOUND_EPS * max(1.0, abs(lo))
    if hi is not None:
        close |= values >= hi - BOUND_EPS * max(1.0, abs(hi))
    return close.mean() > BOUND_FRACTION


def holm(pvalues):
    """Holm-adjusted p-values; NaN entries stay NaN and are not counted."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from statsmodels.stats.multitest import multipletests
        out[ok] = multipletests(p[ok], method='holm')[1]
    return out


def bootstrap_ci(statistic, units, n_boot, rng, confidence=0.95):
    """Percentile CI of ``statistic(resampled units)``, resampling whole units (seeds)."""
    units = list(units)
    draws = []
    for _ in range(n_boot):
        sample = [units[i] for i in rng.integers(0, len(units), len(units))]
        value = statistic(sample)
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < 10:
        return math.nan, math.nan
    alpha = (1 - confidence) / 2
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def mean_ci(values, n_boot=DEFAULT_BOOT, seed=0, confidence=0.95):
    """Mean and percentile bootstrap CI, every resample drawn as one array."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return (float(values.mean()), float(np.quantile(draws, alpha)),
            float(np.quantile(draws, 1 - alpha)))


def slope(x, y):
    """The OLS slope of y on x, without ``linregress``'s overhead (bootstrap inner loop)."""
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def spearman(x, y):
    """Spearman's rho as the Pearson correlation of the ranks (average ties)."""
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else math.nan


def facets(df, keys):
    """``(dict of levels, group)`` per combination of ``keys``; one facet when empty."""
    keys = list(keys or ())
    if not keys:
        yield {}, df
        return
    for values, group in df.groupby(keys, sort=True, dropna=False):
        yield dict(zip(keys, values if isinstance(values, tuple) else (values,))), group


def is_level(series, level):
    return series.map(lambda v: same_level(v, level)).astype(bool)


# --- Jonckheere-Terpstra --------------------------------------------------------

@dataclass
class JTResult:
    J: float
    z: float
    p: float
    tau: float          # effect size: 2 J / (cross pairs) - 1, in [-1, 1]
    method: str


def _jt_statistics(values, earlier, chunk=256):
    """J for each row of ``values`` (k, n): the cross-group pairs (a, b), a's
    group before b's (``earlier[a, b]``), with b > a, ties counting a half.
    Rows are done ``chunk`` at a time, bounding the (chunk, n, n) temporary."""
    out = np.empty(len(values))
    for i in range(0, len(values), chunk):
        v = values[i:i + chunk]
        order = np.sign(v[:, None, :] - v[:, :, None])        # [p, a, b] = sign(v_b - v_a)
        out[i:i + chunk] = ((order + 1) * earlier).sum(axis=(1, 2)) / 2
    return out


def jonckheere_terpstra(groups, n_perm=5000, seed=0, exact_below=100):
    """Ordered-alternative trend test over ``groups`` (in increasing factor order).

    J counts cross-group pairs (a in an earlier group, b in a later one) with
    b > a, ties counting a half. The two-sided p comes from a Monte-Carlo
    permutation distribution when the total n is below ``exact_below``, and
    otherwise from the normal approximation (no tie correction). scipy has no
    JT test.
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    sizes = np.array([len(g) for g in groups])
    n = int(sizes.sum())
    pairs = float((n ** 2 - np.sum(sizes ** 2)) / 2)
    if len(groups) < 2 or pairs == 0:
        return JTResult(math.nan, math.nan, math.nan, math.nan, 'not enough groups')

    pooled = np.concatenate(groups)
    label = np.repeat(np.arange(len(groups)), sizes)
    earlier = label[:, None] < label[None, :]
    J = float(_jt_statistics(pooled[None, :], earlier)[0])
    mean = pairs / 2
    var = (n ** 2 * (2 * n + 3) - np.sum(sizes ** 2 * (2 * sizes + 3))) / 72.0
    z = (J - mean) / math.sqrt(var) if var > 0 else math.nan
    tau = 2 * J / pairs - 1
    if n < exact_below:
        rng = np.random.default_rng(seed)
        permuted = rng.permuted(np.broadcast_to(pooled, (n_perm, n)), axis=1)
        null = _jt_statistics(permuted, earlier)
        extreme = int(np.sum(np.abs(null - mean) >= abs(J - mean) - 1e-12))
        return JTResult(J, z, (extreme + 1) / (n_perm + 1), tau, 'permutation')
    return JTResult(J, z, float(2 * sps.norm.sf(abs(z))), tau, 'normal approximation')


# --- analyses ---------------------------------------------------------------

def cell_summary(df, metric, by, n_boot=DEFAULT_BOOT):
    """Mean with a bootstrap CI, SD and n per cell of the ``by`` factors."""
    rows = []
    for cell, group in facets(df, by):
        mean, lo, hi = mean_ci(group[metric], n_boot)
        rows.append(dict(**cell, metric=metric, mean=mean, ci_low=lo, ci_high=hi,
                         sd=float(group[metric].std(ddof=1)) if len(group) > 1 else math.nan,
                         n=int(group[metric].notna().sum()),
                         diverged=int(group['diverged'].sum()) if 'diverged' in group else 0))
    return pd.DataFrame(rows)


def vs_control(df, metric, factor, control, by=(), *, rank=False):
    """Each level of ``factor`` against its ``control`` level, within each facet of ``by``."""
    rows = []
    data = df.dropna(subset=[metric])
    for facet, group in facets(data, by):
        ref = group[is_level(group[factor], control)].sort_values('seed_index')
        levels = [v for v in pd.unique(group[factor]) if not same_level(v, control)]
        levels = sorted(levels, key=_sort_key)
        samples = [group[is_level(group[factor], v)].sort_values('seed_index') for v in levels]
        if len(ref) < 2 or not levels or any(len(s) < 2 for s in samples):
            continue
        if rank:
            ps, facet_rows = [], []
            for v, s in zip(levels, samples):
                paired = s.merge(ref, on='seed_index', suffixes=('', '_c'))
                diff = (paired[metric] - paired[f'{metric}_c']).to_numpy(float)
                p = (sps.wilcoxon(diff).pvalue if len(diff) and np.any(diff != 0) else math.nan)
                est, lo, hi = mean_ci(diff)
                facet_rows.append(dict(metric=metric, **facet, **{factor: v}, n=len(diff),
                                       estimate=float(np.median(diff)) if len(diff) else math.nan,
                                       mean_diff=est, ci_low=lo, ci_high=hi, p_raw=p))
                ps.append(p)
            for r, p in zip(facet_rows, holm(ps)):
                rows.append(dict(r, p=p, effect='median paired difference vs control',
                                 test_path='rank: Wilcoxon signed-rank by seed, Holm'))
            continue
        # Dunnett's p-values come from a quasi-Monte-Carlo integration; a fixed
        # generator makes the report reproducible to the last digit.
        result = sps.dunnett(*[s[metric].to_numpy(float) for s in samples],
                             control=ref[metric].to_numpy(float),
                             rng=np.random.default_rng(DUNNETT_SEED))
        ci = result.confidence_interval()
        for i, (v, s) in enumerate(zip(levels, samples)):
            rows.append(dict(metric=metric, **facet, **{factor: v}, n=len(s),
                             estimate=float(s[metric].mean() - ref[metric].mean()),
                             ci_low=float(ci.low[i]), ci_high=float(ci.high[i]),
                             p=float(result.pvalue[i]), effect='mean difference vs control',
                             test_path='parametric: Dunnett vs control'))
    return pd.DataFrame(rows)


def paired_difference(df, metric, factor, a, b, by=(), *, rank=False):
    """``factor = a`` minus ``factor = b``, paired by seed, within each facet; Holm across them."""
    rows, ps = [], []
    data = df.dropna(subset=[metric])
    for facet, group in facets(data, by):
        on = group[is_level(group[factor], a)].set_index('seed_index')[metric]
        off = group[is_level(group[factor], b)].set_index('seed_index')[metric]
        shared = on.index.intersection(off.index)
        if len(shared) < 2:
            continue
        diff = (on[shared] - off[shared]).to_numpy(float)
        if np.all(diff == 0):
            p = math.nan
        elif rank:
            p = sps.wilcoxon(diff).pvalue
        else:
            p = sps.ttest_rel(on[shared], off[shared]).pvalue
        mean = float(diff.mean())
        if len(diff) > 1 and np.std(diff, ddof=1) > 0:
            half = sps.t.ppf(0.975, len(diff) - 1) * np.std(diff, ddof=1) / math.sqrt(len(diff))
        else:
            half = math.nan
        rows.append(dict(metric=metric, **facet, contrast=f'{factor}: {a} - {b}', n=len(diff),
                         estimate=mean, ci_low=mean - half, ci_high=mean + half, p_raw=float(p),
                         effect='mean paired difference',
                         test_path=('rank: Wilcoxon signed-rank by seed, Holm' if rank
                                    else 'parametric: paired t-test by seed, Holm')))
        ps.append(p)
    out = pd.DataFrame(rows)
    if len(out):
        out['p'] = holm(ps)
    return out


def trend(df, metric, factor, by=(), *, transform=None, rank=False, n_boot=DEFAULT_BOOT, seed=0):
    """The metric against ``transform(factor)`` within each facet of ``by``.

    Levels whose transform is not finite (log of 0, an 'inf' control) are left
    out. Enter them through ``vs_control`` instead.
    """
    fn = TRANSFORMS[transform]
    rows = []
    data = df.dropna(subset=[metric]).copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        data['_x'] = fn(pd.to_numeric(data[factor], errors='coerce').astype(float))
    data = data[np.isfinite(data['_x'])]
    for facet, group in facets(data, by):
        x = group['_x'].to_numpy(float)
        y = group[metric].to_numpy(float)
        levels = sorted(group['_x'].unique())
        row = dict(metric=metric, **facet, factor=factor, transform=transform or 'none',
                   n=len(y), levels=len(levels))
        # Each seed as plain arrays: the bootstrap below fits thousands of
        # resamples, and concatenating DataFrames per draw costs ~100x the fit.
        units = [(g['_x'].to_numpy(float), g[metric].to_numpy(float))
                 for _, g in group.groupby('seed_index')]
        if len(levels) < 2 or np.ptp(y) == 0:
            rows.append(dict(row, estimate=math.nan, ci_low=math.nan, ci_high=math.nan,
                             p=math.nan, test_path='degenerate: fewer than 2 levels or constant'))
            continue
        rng = np.random.default_rng(seed)

        def resampled(fit):
            def statistic(sample):
                xs = np.concatenate([u[0] for u in sample])
                if np.ptp(xs) == 0:
                    return math.nan
                return fit(xs, np.concatenate([u[1] for u in sample]))
            return statistic

        if rank:
            rho, p = sps.spearmanr(x, y)
            lo, hi = bootstrap_ci(resampled(spearman), units, n_boot, rng)
            row.update(estimate=float(rho), ci_low=lo, ci_high=hi, p=float(p),
                       effect='Spearman rho', test_path='rank: Spearman rho, seed-bootstrap CI')
        else:
            fit = sps.linregress(x, y)
            lo, hi = bootstrap_ci(resampled(slope), units, n_boot, rng)
            row.update(estimate=float(fit.slope), se=float(fit.stderr), ci_low=lo, ci_high=hi,
                       p=float(fit.pvalue), effect=f'slope per unit of {transform or ""} {factor}'.replace('  ', ' '),
                       test_path='parametric: OLS slope t-test, seed-bootstrap CI')
        jt = jonckheere_terpstra([group.loc[group['_x'] == v, metric] for v in levels], seed=seed)
        row.update(jt_tau=jt.tau, jt_z=jt.z, jt_p=jt.p, jt_method=jt.method)
        rows.append(row)
    return pd.DataFrame(rows)


def ols_term(df, metric, formula, term, *, rank=False):
    """One coefficient of a statsmodels formula, ``y`` standing for the metric."""
    import statsmodels.formula.api as smf
    data = df.dropna(subset=[metric]).copy()
    data['y'] = sps.rankdata(data[metric]) if rank else data[metric].astype(float)
    base = dict(metric=metric, term=term, formula=formula, n=len(data))
    try:
        fit = smf.ols(formula, data=data).fit()
        lo, hi = fit.conf_int().loc[term]
    except Exception as exc:  # noqa: BLE001 -- a degenerate design is a row, not a crash
        return pd.DataFrame([dict(base, estimate=math.nan, ci_low=math.nan, ci_high=math.nan,
                                  p=math.nan, test_path=f'skipped: {type(exc).__name__}: {exc}')])
    return pd.DataFrame([dict(base, estimate=float(fit.params[term]), ci_low=float(lo),
                              ci_high=float(hi), p=float(fit.pvalues[term]),
                              r2=float(fit.rsquared),
                              test_path='rank: OLS on ranks' if rank else 'parametric: OLS')])


def _sort_key(v):
    try:
        return (0, float(v), '')
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def _concat(frames):
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def analyze(df, cfg, metrics=None, n_boot=DEFAULT_BOOT, seed=0):
    """Every configured analysis for every metric: ``{name: DataFrame}`` plus ``paths``."""
    factors = [f for f in cfg.factors if f in df]
    spec = cfg.extra.get('analysis') or {}
    metrics = metrics or [m for m in (cfg.metrics or {}) if m in df and df[m].notna().any()]
    if not metrics:
        reserved = {'key', 'seed_index', 'diverged', 'steps', 'wall_seconds', 'frozen_hash',
                    *factors}
        metrics = [c for c in df.columns if c not in reserved
                   and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    if len(cfg.control) > 1:
        raise ValueError('control names one factor and its reference level')
    out = {k: [] for k in ('summary', 'vs_control', 'trend', 'paired', 'ols')}
    paths = []
    for metric in metrics:
        bounds = cfg.metric_bounds(metric)
        rank = near_bound(df[metric], bounds)
        paths.append(dict(metric=metric, path='rank-based' if rank else 'parametric',
                          reason=(f'more than {BOUND_FRACTION:.0%} of values within '
                                  f'{BOUND_EPS} of a bound {bounds}' if rank
                                  else 'values away from the metric bounds')))
        out['summary'].append(cell_summary(df, metric, factors, n_boot))
        for factor, level in cfg.control.items():
            others = [f for f in factors if f != factor]
            out['vs_control'].append(vs_control(df, metric, factor, level, others, rank=rank))
        if spec.get('trend'):
            t = spec['trend']
            others = [f for f in factors if f != t['factor']]
            out['trend'].append(trend(df, metric, t['factor'], others,
                                      transform=t.get('transform'), rank=rank,
                                      n_boot=n_boot, seed=seed))
        for p in spec.get('paired') or []:
            others = [f for f in factors if f != p['factor']]
            out['paired'].append(paired_difference(df, metric, p['factor'], p['a'], p['b'],
                                                   others, rank=rank))
        for o in spec.get('ols') or []:
            out['ols'].append(ols_term(df, metric, o['formula'], o['term'], rank=rank))
    result = {k: _concat(v) for k, v in out.items()}
    result['paths'] = pd.DataFrame(paths)
    return result
