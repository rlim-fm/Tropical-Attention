"""One tiny run reports every metric the config declares, the OOD splits included."""
import pytest

from expkit.api import RunContext
from expkit.config import load_config
from train import run

CONFIG = 'configs/tropical.yaml'


@pytest.mark.parametrize('campaign', ['Knapsack', 'SCC'])
def test_run_reports_declared_metrics(campaign):
    cfg = load_config(CONFIG, campaign)
    base = cfg.base
    base['data'].update(train_lengths=[4, 5], ood_lengths=[6], n_per_length=16,
                        n_eval_per_length=8, curve_eval_limit=4)
    base['optim'].update(epochs=1, batch_size=8)
    ctx = RunContext(base=base, cell={'model_type': 'tropical', 'head': 'pointer', 'objective': 'erm'},
                     seeds={'data': 0, 'init': 0, 'sampler': 0}, seed_index=0)
    result = run(ctx)
    assert not result.diverged
    assert {'ood_noise_f1', 'ood_noise_exact', 'ood_val_exact'} <= set(cfg.metrics)
    for name in cfg.metrics:
        assert 0.0 <= result.metrics[name] <= 1.0, name
