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
from tasks import TASKS, make_env, tokens_per_length

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
