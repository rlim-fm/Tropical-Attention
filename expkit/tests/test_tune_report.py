"""The tuning report: its checks, its verdict, and the approve gate it drives."""
import json
import os

import pandas as pd
import pytest

from expkit import cli, frozen, tune_report as tr
from expkit.config import load_config
from expkit.tests.helpers import write_config
from expkit.tests.test_tune import TUNE


def run(*argv):
    return cli.main(list(argv))


def frame(steps, **metrics):
    n = len(steps)
    return pd.DataFrame({'key': [f'k{i}' for i in range(n)], 'seed_index': range(n),
                         'steps': steps, **metrics})


def cand(label, base, steps=(10, 10), selected=False, **metrics):
    return {'label': label, 'base': base, 'frame': frame(list(steps), **metrics),
            'selected': selected}


def levels(warnings):
    return [level for level, _ in warnings]


def test_verdict_is_the_worst_warning():
    assert tr.verdict([]) == 'ok'
    assert tr.verdict([(tr.INFO, 'x')]) == 'ok'
    assert tr.verdict([(tr.INFO, 'x'), (tr.REVIEW, 'y')]) == 'review'
    assert tr.verdict([(tr.REVIEW, 'y'), (tr.BLOCK, 'z')]) == 'do not approve'


def test_identical_candidates_block_and_a_tiny_spread_needs_review():
    same = {('a', 0): 1.0, ('a', 1): 2.0}
    assert levels(tr.check_flat([same, dict(same)], [1.5, 1.5])) == [tr.BLOCK]
    other = {('a', 0): 1.0, ('a', 1): 2.0000001}
    assert levels(tr.check_flat([same, other], [1.5, 1.50000005])) == [tr.REVIEW]
    assert tr.check_flat([same, {('a', 0): 3.0, ('a', 1): 4.0}], [1.5, 3.5]) == []


def test_a_runner_up_inside_the_noise_is_a_tie():
    winner = {(s,): float(s) for s in range(6)}
    noisy = {(s,): float(s) + (0.1 if s % 2 else -0.05) for s in range(6)}
    clear = {(s,): float(s) + 1.0 for s in range(6)}
    tie = dict(label='b', selected=False, **tr.paired_gap(winner, noisy, sign=1))
    assert levels(tr.check_tie([tie])) == [tr.REVIEW]
    gap = dict(label='c', selected=False, **tr.paired_gap(winner, clear, sign=1))
    assert gap['worse'] == 6 and gap['ci_low'] > 0
    assert tr.check_tie([gap]) == []


def test_a_winner_on_the_edge_of_the_space_needs_review():
    space = {'lam': [1.0, 10.0, 100.0]}
    assert levels(tr.check_boundary(space, {'lam': 100.0})) == [tr.REVIEW]
    assert tr.check_boundary(space, {'lam': 10.0}) == []
    assert levels(tr.check_boundary({'lam': [1.0, 10.0]}, {'lam': 1.0})) == [tr.INFO]


def test_runs_that_stop_before_the_warmup_never_used_the_knob():
    base = {'fishr': {'anneal_iters': 10}}
    assert levels(tr.check_warmup([cand('a', base, steps=(5, 10))], 'fishr.anneal_iters')) == [tr.BLOCK]
    assert levels(tr.check_warmup([cand('a', base, steps=(5, 20))], 'fishr.anneal_iters')) == [tr.REVIEW]
    assert tr.check_warmup([cand('a', base, steps=(20, 20))], 'fishr.anneal_iters') == []


def test_a_negligible_penalty_needs_review():
    small = cand('a', {'fishr': {'lambda': 1.0}}, pen=[1e-4, 1e-4], loss=[0.5, 0.5])
    large = cand('b', {'fishr': {'lambda': 1000.0}}, pen=[1e-4, 1e-4], loss=[0.5, 0.5])
    out = tr.check_penalty([small, large], 'pen', 'fishr.lambda', 'loss')
    assert levels(out) == [tr.REVIEW] and out[0][1].startswith('a:')


def test_a_winner_still_improving_at_the_last_epoch_needs_review():
    winner = cand('a', {}, selected=True)
    falling = {'k0': {'val': [3.0, 2.0, 1.0]}, 'k1': {'val': [3.0, 2.0, 1.5]}}
    settled = {'k0': {'val': [3.0, 1.0, 1.2]}, 'k1': {'val': [3.0, 1.0, 1.1]}}
    assert levels(tr.check_convergence(winner, falling, 'val', 'min')) == [tr.REVIEW]
    assert tr.check_convergence(winner, settled, 'val', 'min') == []
    assert levels(tr.check_convergence(winner, {'k0': {'val': [1.0]}}, 'val', 'min')) == [tr.INFO]


def test_a_winner_that_picked_its_last_epoch_needs_review():
    base = {'optim': {'epochs': 20}}
    late = cand('a', base, selected=True, best_epoch=[19.0, 19.0])
    early = cand('a', base, selected=True, best_epoch=[12.0, 19.0])
    assert levels(tr.check_best_epoch(late, 'best_epoch', 'optim.epochs')) == [tr.REVIEW]
    assert tr.check_best_epoch(early, 'best_epoch', 'optim.epochs') == []


def test_missing_and_diverged_runs():
    table = pd.DataFrame([{'lam': 1.0, 'runs': 1, 'expected': 2, 'diverged': 0},
                          {'lam': 2.0, 'runs': 2, 'expected': 2, 'diverged': 1}])
    assert levels(tr.check_complete(table, ['lam'])) == [tr.BLOCK, tr.REVIEW]


def test_the_budget_gap_is_reported():
    out = tr.check_budget({'optim': {'epochs': 100}}, {'optim': {'epochs': 5}})
    assert out == [(tr.INFO, 'the search ran with optim.epochs=5; the grid runs with optim.epochs=100')]


def test_tune_writes_the_report_and_the_verdict_into_frozen(tmp_path, campaigns_tmp):
    cfg_path = write_config(tmp_path / 'quad.yaml', tune=TUNE)
    assert run('tune', '--config', cfg_path) == 0
    cfg = load_config(cfg_path)
    out = tr.report_dir(cfg)
    assert os.path.exists(os.path.join(out, 'tune_report.tex'))
    assert os.path.exists(os.path.join(out, 'tune_sensitivity.png'))
    with open(os.path.join(out, 'warnings.json')) as fh:
        report = json.load(fh)
    # offset 0.0 wins, and it is the smallest value searched
    assert report['verdict'] == 'review'
    assert any('smallest value searched' in w['message'] for w in report['warnings'])
    doc = frozen.read_doc(os.path.join(campaigns_tmp, 'quad', 'default', 'frozen.yaml'))
    assert doc['verdict'] == 'review' and 'tune_report' in doc['report']
    assert run('approve', '--config', cfg_path) == 0


def test_a_knob_with_no_effect_blocks_approval(tmp_path, campaigns_tmp):
    flat = {**TUNE, 'space': {'unused': [1.0, 2.0]}}
    cfg_path = write_config(tmp_path / 'quad.yaml', tune=flat)
    assert run('tune', '--config', cfg_path) == 0
    doc = frozen.read_doc(os.path.join(campaigns_tmp, 'quad', 'default', 'frozen.yaml'))
    assert doc['verdict'] == 'do not approve'
    with pytest.raises(SystemExit, match='do not approve'):
        run('approve', '--config', cfg_path)
    assert run('approve', '--config', cfg_path, '--force') == 0
    doc = frozen.read_doc(os.path.join(campaigns_tmp, 'quad', 'default', 'frozen.yaml'))
    assert doc['approved'] and doc['forced_at_approval']


def test_sealed_metrics_stay_out_of_the_decision(tmp_path, campaigns_tmp):
    cfg_path = write_config(tmp_path / 'quad.yaml', tune=TUNE)
    run('tune', '--config', cfg_path)
    a = tr.analyze(load_config(cfg_path))
    assert not any(tr.sealed(m) for m in a['open_metrics'])
    assert tr.sealed('ood_len_f1') and tr.sealed('test_loss') and not tr.sealed('id_val_loss')
