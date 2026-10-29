"""The grand report: every campaign's status and results, and the jobs that build it."""
import os

import pandas as pd

from expkit import cli, grand, slurm
from expkit.config import load_config
from expkit.tests.helpers import write_config

AXES = {'noise': {'clean': {}, 'noisy': {'base': {'offset': 1.0}}}}


def run(*argv):
    return cli.main(list(argv))


def test_grand_report_covers_every_campaign(tmp_path, campaigns_tmp):
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=AXES)
    assert run('run', '--config', cfg_path, '--profile', 'clean') == 0   # noisy never runs
    assert run('grand', '--config', cfg_path) == 0
    out = grand.grand_dir(load_config(cfg_path))
    status = pd.read_csv(os.path.join(out, 'grand_status.csv'))
    assert list(status['campaign']) == ['clean', 'noisy']
    assert list(status['stored']) == [9, 0]
    table = pd.read_csv(os.path.join(out, 'grand_mse.csv'))
    assert list(table['campaign']) == ['clean']
    assert {'0.1_mean', '0.01_mean', '0.3_mean', '0.01_diff', '0.3_diff'} <= set(table.columns)
    assert '0.1_diff' not in table      # the control has no difference against itself
    with open(os.path.join(out, 'grand_report.tex')) as fh:
        tex = fh.read()
    assert 'clean' in tex and 'noisy' in tex and r'at a glance}' in tex
    assert r'\cellcolor[HTML]' in tex                      # the colored blocks
    # each cell links to its panel of the campaign's loss figure
    assert r'\hyperlink{loss-clean-0-0}' in tex and r'\hypertarget{loss-clean-0-0}' in tex
    assert os.path.exists(os.path.join(out, 'loss_clean.png'))


def test_submit_queues_the_grand_report_after_every_job(tmp_path, monkeypatch, capsys):
    cfg_path = write_config(tmp_path / 'quad.yaml', axes=AXES)
    monkeypatch.setattr(slurm, 'queued', lambda name, **_: [])
    run('submit', '--config', cfg_path, '--cores', '4', '--plan', '--grand')
    out = capsys.readouterr().out.replace("'", '')
    grand_line = [line for line in out.splitlines() if '--job-name=xk-quad-grand' in line]
    assert len(grand_line) == 1
    assert '--dependency=afterany:<xk-quad-clean>:<xk-quad-noisy>' in grand_line[0]
