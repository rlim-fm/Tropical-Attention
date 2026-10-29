from expkit.executor import execute
from expkit.record import RunSpec
from expkit.store import Store


def specs(entry, lrs=(0.01, 0.1, 0.3), seeds=2, base=None):
    return [RunSpec('e', f'expkit.tests.entries:{entry}', 'h', base or {'offset': 0.0},
                    {'lr': lr}, s, 0) for lr in lrs for s in range(seeds)]


def test_parallel_runs_land_in_the_store(tmp_path):
    todo = specs('quadratic')
    out = execute(todo, str(tmp_path), ['cpu'] * 3, log=lambda *a: None)
    assert sorted(out['ok']) == sorted(s.key for s in todo) and not out['failed']
    assert Store(tmp_path).pending(todo) == []


def test_a_failure_is_logged_and_retried(tmp_path):
    marker = tmp_path / 'markers'
    marker.mkdir()
    todo = specs('fails_once', lrs=(0.1,), seeds=1, base={'offset': 0.0, 'marker_dir': str(marker)})
    out = execute(todo, str(tmp_path), ['cpu'], retries=1, log=lambda *a: None)
    assert out['ok'] == [todo[0].key]
    statuses = [a['status'] for a in Store(tmp_path).attempts()]
    assert statuses == ['failed', 'ok']


def test_exhausted_retries_are_failures(tmp_path):
    todo = specs('always_fails', lrs=(0.1,), seeds=1)
    out = execute(todo, str(tmp_path), ['cpu'], retries=1, log=lambda *a: None)
    assert out['failed'] == [todo[0].key]
    assert len(Store(tmp_path).failures()) == 1
    assert 'ValueError: broken' in Store(tmp_path).failures()[0]['error']


def test_an_out_of_memory_retry_waits_for_the_pool_to_drain(tmp_path):
    marker = tmp_path / 'markers'
    marker.mkdir()
    todo = specs('oom_once', lrs=(0.01, 0.3), seeds=1,
                 base={'offset': 0.0, 'marker_dir': str(marker)})
    out = execute(todo, str(tmp_path), ['cpu'] * 2, retries=1, log=lambda *a: None)
    assert sorted(out['ok']) == sorted(s.key for s in todo)
    oom, slow = todo[0].key, todo[1].key
    order = [(a['key'], a['status']) for a in Store(tmp_path).attempts()]
    assert order == [(oom, 'failed'), (slow, 'ok'), (oom, 'ok')]


def test_a_dead_worker_charges_only_its_own_run(tmp_path):
    todo = specs('dies_on', seeds=2)
    victim = next(s.key for s in todo if s.cell['lr'] == 0.3 and s.seed_index == 0)
    out = execute(todo, str(tmp_path), ['cpu'] * 2, retries=1, log=lambda *a: None)
    assert out['failed'] == [victim]
    assert sorted(out['ok']) == sorted(s.key for s in todo if s.key != victim)
    charged = {a['key'] for a in Store(tmp_path).attempts() if a['status'] == 'failed'}
    # a worker killed alongside the victim may share its pool, but is never charged
    # once the culprit is identified
    assert victim in charged


def _records(tmp_path):
    return {r.key: r for r in Store(tmp_path).load_records()}


def test_seed_groups_train_each_cell_once_and_store_every_seed(tmp_path):
    todo = specs('quadratic', seeds=3)
    out = execute(todo, str(tmp_path), ['cpu'] * 2, group_seeds=True, log=lambda *a: None)
    assert sorted(out['ok']) == sorted(s.key for s in todo) and not out['failed']
    assert {r.extra.get('seed_group') for r in _records(tmp_path).values()} == {3}
    alone = tmp_path / 'alone'
    execute(todo, str(alone), ['cpu'], log=lambda *a: None)
    assert ({k: r.metrics for k, r in _records(tmp_path).items()}
            == {k: r.metrics for k, r in _records(alone).items()})


def test_seed_groups_take_only_the_missing_seeds(tmp_path):
    todo = specs('quadratic', lrs=(0.1,), seeds=3)
    execute(todo[:1], str(tmp_path), ['cpu'], log=lambda *a: None)
    rest = Store(tmp_path).pending(todo)
    assert len(rest) == 2
    execute(rest, str(tmp_path), ['cpu'], group_seeds=True, log=lambda *a: None)
    groups = {r.seed_index: r.extra.get('seed_group') for r in _records(tmp_path).values()}
    assert groups == {0: None, 1: 2, 2: 2}


def test_a_failing_group_falls_back_to_single_runs(tmp_path):
    todo = specs('group_raises', lrs=(0.1,), seeds=2)
    out = execute(todo, str(tmp_path), ['cpu'], retries=1, group_seeds=True, log=lambda *a: None)
    assert sorted(out['ok']) == sorted(s.key for s in todo)
    statuses = sorted((a['seed_index'], a['status']) for a in Store(tmp_path).attempts())
    assert statuses == [(0, 'failed'), (0, 'ok'), (1, 'failed'), (1, 'ok')]


def test_a_seed_failing_inside_a_group_is_retried_alone(tmp_path):
    todo = specs('bad_seed', lrs=(0.1,), seeds=3)
    out = execute(todo, str(tmp_path), ['cpu'], retries=1, group_seeds=True, log=lambda *a: None)
    assert sorted(out['ok']) == sorted(s.key for s in todo)
    failed = [a for a in Store(tmp_path).attempts() if a['status'] == 'failed']
    assert [a['seed_index'] for a in failed] == [1] and 'seed 1 broke' in failed[0]['error']
    groups = {r.seed_index: r.extra.get('seed_group') for r in _records(tmp_path).values()}
    assert groups == {0: 3, 1: None, 2: 3}


def test_an_entry_without_a_group_variant_runs_one_by_one(tmp_path):
    todo = specs('fails_once', lrs=(0.1,), seeds=2,
                 base={'offset': 0.0, 'marker_dir': str(tmp_path)})
    out = execute(todo, str(tmp_path), ['cpu'], retries=1, group_seeds=True, log=lambda *a: None)
    assert sorted(out['ok']) == sorted(s.key for s in todo)
