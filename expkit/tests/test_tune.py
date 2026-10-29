"""The tune stage: a grid search into an unapproved frozen.yaml, on seeds of its own."""
import os

import pandas as pd
import pytest

from expkit import cli, frozen, slurm, tune
from expkit.config import load_config
from expkit.store import Store
from expkit.tests.helpers import write_config

# quadratic: mse = (lr - 0.1)^2 + noise + offset, so offset 0.0 must win.
TUNE = {'global_seed': 7, 'seeds': 2, 'only': {'lr': 0.1},
        'space': {'offset': [0.5, 0.0, 1.0]}, 'select': {'metric': 'mse', 'direction': 'min'}}


def run(*argv):
    return cli.main(list(argv))


def test_tune_freezes_the_best_candidate_unapproved(tmp_path, campaigns_tmp):
    cfg_path = write_config(tmp_path / 'quad.yaml', tune=TUNE, require_frozen=True)
    with pytest.raises(SystemExit, match='require_frozen'):    # refused before tuning
        run('run', '--config', cfg_path)
    assert run('tune', '--config', cfg_path, '--cpu-workers', '2') == 0
    cfg = load_config(cfg_path)
    doc = frozen.read_doc(os.path.join(campaigns_tmp, 'quad', 'default', 'frozen.yaml'))
    assert doc['approved'] is False and doc['tuned'] == {'offset': 0.0}
    assert doc['evidence'].endswith(os.path.join('tune', 'tune.csv'))
    table = pd.read_csv(tune.evidence_path(cfg))
    assert len(table) == 3 and table['selected'].sum() == 1
    assert table.loc[table['selected'], 'offset'].item() == 0.0

    # only the `only` cell, on the tuning seeds, in the tuning store; the grid store is untouched
    records = Store(tune.store_dir(cfg)).load_records()
    assert len(records) == 3 * 2
    assert {r.cell['lr'] for r in records} == {0.1}
    assert {r.global_seed for r in records} == {7}
    assert not Store(os.path.join(campaigns_tmp, 'quad', 'default', 'results')).runs()

    with pytest.raises(SystemExit):                      # still unapproved
        run('run', '--config', cfg_path)
    assert run('approve', '--config', cfg_path) == 0
    assert run('run', '--config', cfg_path) == 0


def test_retuning_an_approved_campaign_is_a_no_op(tmp_path, campaigns_tmp, capsys):
    cfg_path = write_config(tmp_path / 'quad.yaml', tune=TUNE)
    run('tune', '--config', cfg_path)
    run('approve', '--config', cfg_path)
    path = os.path.join(campaigns_tmp, 'quad', 'default', 'frozen.yaml')
    approved_at = frozen.read_doc(path)['approved_at']
    assert run('tune', '--config', cfg_path) == 0
    assert 'unchanged' in capsys.readouterr().out
    doc = frozen.read_doc(path)
    assert doc['approved'] and doc['approved_at'] == approved_at


@pytest.mark.parametrize('metric', ['ood_f1', 'test_loss'])
def test_selecting_on_test_or_ood_metrics_is_refused(tmp_path, metric):
    bad = {**TUNE, 'select': {'metric': metric, 'direction': 'max'}}
    cfg_path = write_config(tmp_path / 'quad.yaml', tune=bad)
    with pytest.raises(SystemExit, match='validation metric'):
        run('tune', '--config', cfg_path)


def test_tuning_on_the_grid_seeds_is_refused(tmp_path):
    cfg_path = write_config(tmp_path / 'quad.yaml', tune={**TUNE, 'global_seed': 0})
    with pytest.raises(SystemExit, match='grid seeds'):
        run('tune', '--config', cfg_path)


def test_submit_tune_stage(tmp_path, monkeypatch, capsys):
    axes = {'noise': {'clean': {}, 'noisy': {'base': {'offset': 1.0}}}}
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=axes, tune=TUNE)
    monkeypatch.setattr(slurm, 'queued', lambda name, **_: [])
    run('submit', '--stage', 'tune', '--config', cfg_path, '--cores', '4', '--plan')
    out = capsys.readouterr().out
    assert '--job-name=xk-quad-clean-tune' in out and '--job-name=xk-quad-noisy-tune' in out
    assert ' tune --config' in out and 'report' not in out
