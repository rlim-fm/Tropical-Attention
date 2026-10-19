"""End to end through the CLI, the way a user drives it."""
import os

import numpy as np
import pytest

from expkit import cli, frozen, slurm
from expkit.config import load_config
from expkit.store import Store
from expkit.tests.helpers import write_config


def run(*argv):
    return cli.main(list(argv))


def test_run_skip_replay_report_clear(tmp_path, campaigns_tmp):
    cfg_path = write_config(tmp_path / 'quad.yaml')
    assert run('run', '--config', cfg_path, '--cpu-workers', '2') == 0
    cfg = load_config(cfg_path)
    store = Store(os.path.join(campaigns_tmp, 'quad', 'default', 'results'))
    assert len(store.runs()) == 9
    assert run('run', '--config', cfg_path) == 0                        # nothing pending
    assert len(store.attempts()) == 9
    assert run('replay', '--config', cfg_path, '--only', 'lr=0.3') == 0  # bit-identical
    assert run('report', '--config', cfg_path, '--n-boot', '100') == 0
    report = os.path.join(campaigns_tmp, 'quad', 'default', 'report')
    assert os.path.exists(os.path.join(report, 'report.tex'))
    assert os.path.exists(os.path.join(report, 'stats_vs_control.csv'))
    assert run('clear', '--config', cfg_path, '--only', 'lr=0.3', '--yes') == 0
    assert len(store.runs()) == 6 and not os.path.isdir(report)
    assert cfg.experiment == 'quad'


def test_replay_catches_a_changed_run(tmp_path, monkeypatch):
    cfg_path = write_config(tmp_path / 'quad.yaml', seeds=1)
    run('run', '--config', cfg_path)
    import expkit.tests.entries as entries
    real = entries.quadratic

    def drifted(ctx):
        out = real(ctx)
        out.curves['train_loss'] = np.asarray(out.curves['train_loss']) + 1e-12
        return out
    monkeypatch.setattr(entries, 'quadratic', drifted)
    with pytest.raises(SystemExit, match='does not reproduce'):
        run('replay', '--config', cfg_path, '--only', 'lr=0.1')


def test_frozen_gate_and_derive(tmp_path, campaigns_tmp):
    axes = {'noise': {'clean': {}, 'noisy': {'base': {'offset': 1.0}}}}
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=axes, require_frozen=True)
    with pytest.raises(SystemExit, match='require_frozen'):
        run('run', '--config', cfg_path, '--plan')
    run('freeze', '--config', cfg_path, '--set', 'scale=2')
    with pytest.raises(frozen.NotApproved):
        run('run', '--config', cfg_path, '--plan')
    run('approve', '--config', cfg_path)
    assert run('run', '--config', cfg_path, '--plan') == 0
    run('freeze', '--config', cfg_path, '--profile', 'noisy', '--from-default')
    doc = frozen.read_doc(os.path.join(campaigns_tmp, 'quad', 'noisy', 'frozen.yaml'))
    assert doc['approved'] and doc['base'] == {'offset': 1.0, 'scale': 2}
    assert doc['derived_from']['overrides'] == {'offset': 1.0}
    assert doc['tuned'] == {'scale': 2}
    assert run('run', '--config', cfg_path, '--profile', 'noisy', '--plan') == 0


def test_derive_refuses_a_profile_that_moves_a_tuned_value(tmp_path):
    axes = {'noise': {'clean': {}, 'noisy': {'base': {'offset': 1.0}}}}
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=axes)
    run('freeze', '--config', cfg_path, '--set', 'offset=0.5')
    run('approve', '--config', cfg_path)
    noisy = load_config(cfg_path, 'noisy')
    default = load_config(cfg_path)
    with pytest.raises(SystemExit, match='changes tuned values'):
        frozen.derive(noisy, default)


def test_submit_plans_one_singleton_job_per_campaign(tmp_path, monkeypatch, capsys):
    axes = {'noise': {'clean': {}, 'noisy': {'base': {'offset': 1.0}}}}
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=axes)
    monkeypatch.setattr(slurm, 'queued', lambda name, **_: [])
    run('submit', '--config', cfg_path, '--cores', '4', '--plan', '--report')
    out = capsys.readouterr().out
    assert out.count('--dependency=singleton') == 2
    assert '--job-name=xk-quad-noisy' in out and '--profile noisy' in out
    assert out.count('--dependency=afterany:') == 2
