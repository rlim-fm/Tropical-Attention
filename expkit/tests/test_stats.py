import itertools

import numpy as np
import pandas as pd

from expkit import stats


def brute_jt(groups):
    J = 0.0
    for i, j in itertools.combinations(range(len(groups)), 2):
        for a in groups[i]:
            for b in groups[j]:
                J += 1.0 if b > a else 0.5 if b == a else 0.0
    return J


def test_jt_statistic_matches_brute_force():
    rng = np.random.default_rng(0)
    groups = [rng.integers(0, 5, 6).astype(float) for _ in range(4)]
    assert stats.jonckheere_terpstra(groups, n_perm=200).J == brute_jt(groups)
    up = stats.jonckheere_terpstra([np.arange(5) + 10 * k for k in range(3)], n_perm=500)
    assert up.tau == 1.0 and up.p < 0.05


def test_holm_skips_nan():
    out = stats.holm([0.01, np.nan, 0.04])
    assert np.isnan(out[1]) and np.allclose(out[[0, 2]], [0.02, 0.04])


def frame():
    rng = np.random.default_rng(1)
    rows = []
    for seed in range(6):
        shift = rng.normal()
        for lr in (0.001, 0.01, 0.1):
            for aug in (True, False):
                y = np.log10(lr) + (0.5 if aug else 0.0) + shift + 0.05 * rng.normal()
                rows.append(dict(lr=lr, aug=aug, seed_index=seed, y=y, diverged=False))
    return pd.DataFrame(rows)


def test_trend_vs_control_and_paired():
    df = frame()
    t = stats.trend(df, 'y', 'lr', ['aug'], transform='log10', n_boot=200)
    assert len(t) == 2 and np.allclose(t['estimate'], 1.0, atol=0.1)
    c = stats.vs_control(df, 'y', 'lr', 0.001, ['aug'])
    assert len(c) == 4 and (c['p'] < 0.05).all()
    p = stats.paired_difference(df, 'y', 'aug', True, False, ['lr'])
    assert len(p) == 3 and np.allclose(p['estimate'], 0.5, atol=0.1)
    r = stats.vs_control(df, 'y', 'lr', 0.001, ['aug'], rank=True)
    assert set(r['test_path']) == {'rank: Wilcoxon signed-rank by seed, Holm'}


def test_near_bound_switches_path():
    assert stats.near_bound([1.0] * 5 + [0.5], (None, 1.0))
    assert not stats.near_bound([0.5, 0.6, 0.7], (0, 1))
