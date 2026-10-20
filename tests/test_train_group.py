"""``train.run_group`` trains several seeds of a cell as ``train.run`` trains each one."""
import itertools

import numpy as np
import pytest
import torch

import train
from expkit.api import RunContext
from expkit.config import load_config
from expkit.seeding import DEFAULT_STREAMS, derive_seeds

CONFIG = 'configs/tropical.yaml'


def _ctxs(cell, n=3, base=None):
    base = base or load_config(CONFIG, 'dry_run').base
    return [RunContext(base=base, cell=dict(cell), seed_index=i,
                       seeds=derive_seeds(0, i, tuple(DEFAULT_STREAMS))) for i in range(n)]


def _deviation(a, b):
    worst = max(abs(a.metrics[k] - b.metrics[k]) for k in a.metrics)
    for name, curve in a.curves.items():
        worst = max(worst, float(np.max(np.abs(np.asarray(curve) - np.asarray(b.curves[name])))))
    return worst


@pytest.mark.parametrize('model_type,head,objective', list(itertools.product(
    ('vanilla', 'adaptive', 'tropical'), ('seq2seq', 'pointer'), ('erm', 'fishr'))))
def test_group_matches_single_runs(model_type, head, objective):
    cell = {'model_type': model_type, 'head': head, 'objective': objective}
    singles = [train.run(ctx) for ctx in _ctxs(cell)]
    group = train.run_group(_ctxs(cell))
    assert len(group) == len(singles)
    for a, b in zip(singles, group):
        assert set(a.metrics) == set(b.metrics) and set(a.curves) == set(b.curves)
        assert a.steps == b.steps and not b.diverged
        assert _deviation(a, b) < 1e-5


def test_a_diverging_seed_leaves_the_others_alone(monkeypatch):
    build = train._model

    def poisoned(ctx, input_dim):
        model = build(ctx, input_dim)
        if ctx.seed_index == 1:
            with torch.no_grad():
                model.input_linear.weight.fill_(float('nan'))
        return model

    monkeypatch.setattr(train, '_model', poisoned)
    cell = {'model_type': 'vanilla', 'head': 'seq2seq', 'objective': 'fishr'}
    singles = [train.run(ctx) for ctx in _ctxs(cell)]
    group = train.run_group(_ctxs(cell))
    assert singles[1].diverged and group[1].diverged
    assert group[1].steps == singles[1].steps
    assert len(group[1].curves['train_loss']) == len(singles[1].curves['train_loss']) == 1
    for i in (0, 2):
        assert not group[i].diverged and _deviation(singles[i], group[i]) < 1e-5


def test_one_seed_falls_back_to_run():
    cell = {'model_type': 'vanilla', 'head': 'seq2seq', 'objective': 'erm'}
    (alone,) = train.run_group(_ctxs(cell, n=1))
    assert _deviation(train.run(_ctxs(cell, n=1)[0]), alone) == 0.0


def test_group_rejects_mixed_cells():
    a = _ctxs({'model_type': 'vanilla', 'head': 'seq2seq', 'objective': 'erm'}, n=1)
    b = _ctxs({'model_type': 'tropical', 'head': 'seq2seq', 'objective': 'erm'}, n=1)
    with pytest.raises(ValueError, match='one cell'):
        train.run_group(a + b)
