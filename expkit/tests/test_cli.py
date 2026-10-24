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


def test_submit_routes_campaigns_and_starts_the_longest_first(tmp_path, monkeypatch, capsys):
    axes = {'noise': {'clean': {'slurm': {'device': 'cpu'}},
                      'noisy': {'base': {'offset': 1.0}, 'slurm': {'cost': 10}}}}
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=axes)
    monkeypatch.setattr(slurm, 'queued', lambda name, **_: [])
    run('submit', '--config', cfg_path, '--gpus', '1', '--per-gpu', '8', '--gpu-type', 'pro6000',
        '--cpus-per-gpu', '40', '--partition', 'mixed', '--cpu-partition', 'main',
        '--cpu-cores', '32', '--group-seeds', '--plan', '--report')
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith('$ sbatch')]
    noisy, clean = lines[0], lines[1]                  # cost 10 x 3 runs before cost 1 x 3
    assert '--job-name=xk-quad-noisy' in noisy and '-p mixed' in noisy
    assert '--cpus-per-gpu=40' in noisy and '--gres=gpu:pro6000:1' in noisy and ' -n ' not in noisy
    assert '--per-gpu 8' in noisy and '--group-seeds' in noisy
    assert '--job-name=xk-quad-clean' in clean and '-p main' in clean and '-n 32' in clean
    assert '--gres' not in clean and '--cpu-workers 32' in clean
    assert all('-p main' in l for l in lines[2:]) and len(lines) == 4   # the report jobs


def test_slurm_settings_rejects_an_unknown_device(tmp_path):
    cfg_path = write_config(tmp_path / 'quad.yaml', slurm={'device': 'tpu'})
    with pytest.raises(SystemExit, match='slurm.device'):
        cli.slurm_settings(load_config(cfg_path))


def test_submit_spreads_gpu_campaigns_over_lanes(tmp_path, monkeypatch, capsys):
    axes = {'noise': {'a': {'slurm': {'cost': 10}}, 'b': {'base': {'offset': 1.0}, 'slurm': {'cost': 4}},
                      'c': {'base': {'offset': 2.0}, 'slurm': {'cost': 3}}}}
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=axes)
    monkeypatch.setattr(slurm, 'queued', lambda name, **_: [])
    run('submit', '--config', cfg_path, '--gpus', '1', '--plan',
        '--lanes', 'mixed:pro6000:1:40::12 gpu:a100:1:2:14:4')
    a, b, c = [l for l in capsys.readouterr().out.splitlines() if l.startswith('$ sbatch')]
    assert '-p mixed' in a and '--gres=gpu:pro6000:1' in a and '--per-gpu 12' in a
    assert '-p gpu' in b and '--gres=gpu:a100:1' in b and '--cpus-per-gpu=2' in b
    assert '--mem=14G' in b and '--per-gpu 4' in b
    assert '-p gpu' in c                               # 4 + 3 < 10: the gpu lane is still emptier


def test_parse_lanes_rejects_a_short_spec():
    with pytest.raises(SystemExit, match='lane'):
        slurm.parse_lanes('mixed:pro6000:2')


def test_submit_sends_any_campaigns_to_cpu_lanes_too(tmp_path, monkeypatch, capsys):
    axes = {'noise': {'a': {'slurm': {'cost': 10, 'device': 'any'}},
                      'b': {'base': {'offset': 1.0}, 'slurm': {'cost': 4, 'device': 'any'}},
                      'c': {'base': {'offset': 2.0}, 'slurm': {'cost': 3, 'device': 'gpu'}}}}
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=axes)
    monkeypatch.setattr(slurm, 'queued', lambda name, **_: [])
    run('submit', '--config', cfg_path, '--gpus', '1', '--plan',
        '--lanes', 'mixed:pro6000:1:40::12 main:cpu:2:32::12:0.5')
    a, b, c = [l for l in capsys.readouterr().out.splitlines() if l.startswith('$ sbatch')]
    assert '-p mixed' in a                             # 10 on the card beats 20 on cores
    assert '-p main' in b and '-n 32' in b and '--cpu-workers 12' in b and '--gres' not in b
    assert '-p mixed' in c                             # device: gpu never takes a CPU lane


def test_assign_lane_counts_slots_not_lane_capacity():
    fast, slow = slurm.Lane('mixed', 'pro6000', 1, 40, None, 12, 1.0), \
        slurm.Lane('main', 'cpu', 10, 32, None, 12, 0.2)
    filled = [[0.0], [0.0] * 10]
    assert slurm.assign_lane([fast, slow], filled, 10) is fast      # 10 vs 50 on one CPU job
    assert slurm.assign_lane([fast, slow], filled, 10) is fast      # 20 vs 50
    assert slurm.assign_lane([fast, slow], filled, 1) is slow       # 21 vs 5
