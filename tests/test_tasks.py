"""Every task yields a valid subset target, one token per position."""
import itertools

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

from dataloaders import (FloydWarshallDataset, LISDataset, SubsetSumDecisionDataset,
                         ThreeSumDecisionDataset, calculate_lis_length, lis_mask,
                         subset_sum_witness, three_sum_witness)
from expkit.config import load_config
from tasks import (SPLITS, TASKS, make_env, make_envs, ood_noise_settings, ood_value_range,
                   split_seed, tokens_per_length)

CONFIG = 'configs/tropical.yaml'


def campaign_tasks():
    import yaml
    with open(CONFIG) as fh:
        return list(yaml.safe_load(fh)['axes']['dataset'])


@pytest.mark.parametrize('campaign', campaign_tasks())
@pytest.mark.parametrize('length', [4, 7])
def test_every_task_is_a_subset_mask(campaign, length):
    task = load_config(CONFIG, campaign).base['task']
    X, Y = make_env(task, length, 20, seed=3)
    S = tokens_per_length(task['name'], length)
    assert X.shape[:2] == (len(X), S) and Y.shape == (len(X), S)
    assert set(Y.unique().tolist()) <= {0.0, 1.0}


def test_every_registered_task_has_a_campaign():
    names = {load_config(CONFIG, c).base['task']['name'] for c in campaign_tasks()}
    assert names == set(TASKS)


def test_subset_sum_witness():
    rng = np.random.default_rng(0)
    for _ in range(200):
        xs = sorted(rng.integers(-5, 6, size=rng.integers(1, 9)).tolist())
        sums = {sum(c) for r in range(len(xs) + 1) for c in itertools.combinations(xs, r)}
        T = int(rng.integers(-10, 11))
        mask = subset_sum_witness(xs, T)
        if T in sums and T != 0:
            assert sum(x for x, m in zip(xs, mask) if m) == T and any(mask)
        else:
            assert not any(mask)


def test_subset_sum_dataset_labels():
    ds = SubsetSumDecisionDataset(n_samples=100, length_range=(6, 6), value_range=(-5, 5), seed=1)
    for x, y in ds.data:
        T = x[0, 1].item()
        if y.any():
            assert x[y.bool(), 0].sum().item() == T


def test_three_sum_witness():
    rng = np.random.default_rng(1)
    for _ in range(200):
        xs = sorted(rng.integers(-10, 11, size=rng.integers(3, 10)).tolist())
        T = int(rng.integers(-5, 6))
        mask = three_sum_witness(xs, T)
        exists = any(sum(c) == T for c in itertools.combinations(xs, 3))
        assert sum(mask) == (3 if exists else 0)
        if exists:
            assert sum(x for x, m in zip(xs, mask) if m) == T
    ds = ThreeSumDecisionDataset(n_samples=50, length_range=(6, 6), seed=0)
    assert all(int(y.sum()) in (0, 3) for _, y in ds.data)


def test_lis_mask():
    rng = np.random.default_rng(2)
    for _ in range(200):
        xs = rng.integers(1, 10, size=rng.integers(1, 12)).tolist()
        mask = lis_mask(xs)
        chosen = [x for x, m in zip(xs, mask) if m]
        assert all(a < b for a, b in zip(chosen, chosen[1:]))
        assert len(chosen) == calculate_lis_length(xs)
    ds = LISDataset(n_samples=20, length_range=(8, 8), seed=0)
    for x, y in ds.data:
        assert int(y.sum()) == calculate_lis_length(x[:, 0].tolist())


def test_floyd_warshall_mask_is_shortest_path_edges():
    ds = FloydWarshallDataset(n_samples=20, length_range=(5, 5), value_range=(1, 15), seed=0)
    for x, y in ds.data:
        n = 5
        W = x[:, 0].reshape(n, n).numpy().astype(float)
        mask = y.reshape(n, n).numpy()
        D = floyd_warshall(csr_matrix(np.where(W > 0, W, 0)), directed=False)
        for u, v in zip(*np.nonzero(mask)):
            assert u != v and W[u, v] > 0
            # (u, v) is the last edge of a shortest path from some source
            assert any(np.isclose(D[i, u] + W[u, v], D[i, v]) for i in range(n))


# Hashemi et al. 2025, Table 5: (OOD value range, perturbative noise range).
TABLE5 = {
    'SubsetSumDecision': ((-20, 20), (10, 30)),
    'Knapsack': ((11, 21), (10, 30)),
    'FractionalKnapsack': ((11, 21), (1, 5)),
    'MinCoinChange': ((11, 21), (1, 5)),
    'Quickselect': ((11, 21), (1, 5)),
    'BalancedPartition': ((11, 100), (10, 30)),
    'BinPacking': ((11, 100), (10, 30)),
    'ConvexHull': ((11, 21), (1, 5)),
    'ThreeSumDecision': ((-375, 375), (40, 60)),
    'FloydWarshall': ((16, 30), (1, 10)),
    'SCC': (0.1, None),
}
GRAPH_CAMPAIGNS = {'FloydWarshall', 'SCC'}


@pytest.mark.parametrize('campaign', campaign_tasks())
def test_ood_protocol_follows_the_paper(campaign):
    base = load_config(CONFIG, campaign).base
    task, data = base['task'], base['data']
    # Length OOD: 64 for set tasks, 16 nodes for the graph tasks (App. F).
    assert (16 if campaign in GRAPH_CAMPAIGNS else 64) == max(data['ood_lengths'])
    assert data['ood_noise_prob'] == 0.5
    value, noise = ood_value_range(task['name'], task.get('value_range'),
                                   data.get('ood_value_range')), ood_noise_settings(data)
    if campaign in TABLE5:
        want_value, want_noise = TABLE5[campaign]
        assert value == want_value
        assert noise.get('adversarial_range') == want_noise
    else:   # not in Table 5: the old doubling rule and the task's own noise range
        assert value == ood_value_range(task['name'], task['value_range'])
        assert 'adversarial_range' not in noise


def test_ood_value_range_fallback():
    assert ood_value_range('KnapsackDataset', (1, 10)) == (1, 20)
    assert ood_value_range('MaxSubsetSumDataset', (-5, 5)) == (-10, 10)
    assert ood_value_range('SCCDataset', 0.001) == 0.6
    assert ood_value_range('KnapsackDataset', (1, 10), [11, 21]) == (11, 21)
    assert ood_value_range('SCCDataset', 0.001, 0.1) == 0.1


def test_ood_noise_settings():
    assert ood_noise_settings({}) == {'noise_prob': 0.5}
    assert ood_noise_settings({'ood_noise_prob': 0.2, 'ood_noise_range': [1, 5]}) == \
        {'noise_prob': 0.2, 'adversarial_range': (1, 5)}


def test_noise_split_perturbs_inputs():
    task = load_config(CONFIG, 'Knapsack').base['task']
    assert task['noise_prob'] == 0.0
    X, _ = make_env(task, 8, 200, seed=5)
    Xn, _ = make_env(task, 8, 200, seed=5, noise_prob=0.5, adversarial_range=(10, 30))
    # Clean Knapsack values and weights are in [1, 10]; a perturbed one is at least 11.
    assert X[..., :2].max() <= 10
    frac = (Xn[..., :2] > 10).float().mean().item()
    assert 0.3 < frac < 0.7


def test_splits_draw_distinct_seeds():
    seeds = {split_seed(0, s, 8) for s in SPLITS}
    assert len(seeds) == len(SPLITS)
    assert 'ood_noise' in SPLITS
    envs = make_envs({'name': 'KnapsackDataset'}, [4, 5], 3, 0, 'ood_noise', noise_prob=0.5)
    assert set(envs) == {4, 5}


def test_dry_run_uses_knapsacks_ood_settings():
    # A named profile applies no axis level, so `base` must hold the first campaign's settings.
    assert load_config(CONFIG, 'dry_run').base['task'] == load_config(CONFIG, 'Knapsack').base['task']
    data = load_config(CONFIG, 'dry_run').base['data']
    assert (data['ood_value_range'], data['ood_noise_range']) == ([11, 21], [10, 30])
