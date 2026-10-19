"""The shipped example runs, stores, replays bit for bit and reports."""
import os

import pytest

from expkit import cli

pytest.importorskip('torch')
CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples', 'toy.yaml')


def test_toy_dry_run_replays_exactly(campaigns_tmp):
    assert cli.main(['run', '--config', CONFIG, '--profile', 'dry_run', '--cpu-workers', '2']) == 0
    assert cli.main(['replay', '--config', CONFIG, '--profile', 'dry_run', '--only', 'seed=0']) == 0
    assert cli.main(['report', '--config', CONFIG, '--profile', 'dry_run', '--n-boot', '100']) == 0
    assert os.path.exists(os.path.join(campaigns_tmp, 'toy', 'dry_run', 'report', 'report.tex'))
