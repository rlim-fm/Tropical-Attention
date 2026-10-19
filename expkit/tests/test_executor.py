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
