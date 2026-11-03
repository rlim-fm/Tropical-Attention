"""One tiny run reports every metric the config declares, the OOD splits included."""
import pytest
import torch

from expkit.api import RunContext
from expkit.config import load_config
from train import _splits, run

CONFIG = 'configs/tropical.yaml'


@pytest.mark.parametrize('campaign', ['Knapsack', 'SCC'])
def test_run_reports_declared_metrics(campaign):
    cfg = load_config(CONFIG, campaign)
    base = cfg.base
    base['data'].update(train_lengths=[4, 5], ood_lengths=[6], n_per_length=16,
                        n_eval_per_length=8, curve_eval_limit=4)
    base['optim'].update(epochs=1, batch_size=8)
    ctx = RunContext(base=base, cell={'model_type': 'tropical', 'head': 'pointer', 'fishr_lambda': 0.0},
                     seeds={'data': 0, 'init': 0, 'sampler': 0}, seed_index=0)
    result = run(ctx)
    assert not result.diverged
    assert {'ood_noise_f1', 'ood_noise_exact', 'ood_val_exact'} <= set(cfg.metrics)
    for name in cfg.metrics:
        assert 0.0 <= result.metrics[name] <= 1.0, name
    timing = result.extra['timing']
    assert set(timing) == {'setup', 'train', 'epoch_eval', 'final', 'far'} and min(timing.values()) >= 0


def _tiny(config, campaign):
    base = load_config(config, campaign).base
    base['data'].update(ood_lengths=[6], n_per_length=16, n_eval_per_length=8, curve_eval_limit=4,
                        ood_far_lengths=[])
    base['optim'].update(epochs=1, batch_size=8)
    return base


@pytest.mark.parametrize('campaign', ['Knapsack', 'FloydWarshall'])
def test_value_envs_run(campaign):
    base = _tiny('configs/tropical_value.yaml', campaign)
    base['data'].update(train_lengths=[5])
    ctx = RunContext(base=base, cell={'model_type': 'tropical', 'head': 'pointer', 'fishr_lambda': 1.0},
                     seeds={'data': 0, 'init': 0, 'sampler': 0}, seed_index=0)
    assert set(_splits(ctx)['train']) == {0, 1, 2}
    result = run(ctx)
    assert not result.diverged and 0.0 <= result.metrics['ood_val_f1'] <= 1.0


def test_gumbel_scale_zero_is_plain_erm():
    base = _tiny('configs/tropical_gumbel.yaml', 'Knapsack')
    base['data'].update(train_lengths=[4, 5])
    seeds = {'data': 0, 'init': 0, 'sampler': 0}
    cell = {'model_type': 'vanilla', 'head': 'seq2seq'}
    erm = run(RunContext(base=base, cell={**cell, 'fishr_lambda': 0.0}, seeds=seeds, seed_index=0))
    zero = run(RunContext(base=base, cell={**cell, 'gumbel_scale': 0.0}, seeds=seeds, seed_index=0))
    noisy_ctx = RunContext(base=base, cell={**cell, 'gumbel_scale': 1.0}, seeds=seeds, seed_index=0)
    assert erm.metrics == zero.metrics
    clean, noisy = _splits(RunContext(base=base, cell=cell, seeds=seeds, seed_index=0)), _splits(noisy_ctx)
    assert not torch.equal(clean['train'][4][0], noisy['train'][4][0])
    assert torch.equal(clean['id_val'][4][0], noisy['id_val'][4][0])     # evaluation stays clean
    assert not run(noisy_ctx).diverged
