import json

from expkit.record import RunSpec, run_key, selector_matches
from expkit.seeding import derive_seeds
from expkit.store import Store, append_jsonl, read_jsonl


def test_torn_tail_is_skipped_then_cut_off(tmp_path):
    path = tmp_path / 'x.jsonl'
    append_jsonl(path, [{'a': 1}])
    with open(path, 'a') as fh:
        fh.write('{"a": 2')                       # a crash mid-write
    assert read_jsonl(path) == [{'a': 1}]
    append_jsonl(path, [{'a': 3}])
    assert read_jsonl(path) == [{'a': 1}, {'a': 3}]
    assert all(json.loads(line) for line in open(path))


def test_seeds_depend_on_the_seed_index_only():
    assert derive_seeds(0, 3) == derive_seeds(0, 3)
    assert derive_seeds(0, 3) != derive_seeds(0, 4)
    s = derive_seeds(0, 3)
    assert len(set(s.values())) == 3
    # appending a stream keeps the earlier ones
    assert derive_seeds(0, 3, ('init', 'data', 'sampler', 'extra'))['data'] == s['data']


def test_run_key_is_the_cell_and_seed_under_one_hash():
    k = run_key('e', 'h', {'lr': 0.1}, 0)
    assert k == run_key('e', 'h', {'lr': 0.1}, 0)
    assert k != run_key('e', 'h2', {'lr': 0.1}, 0)
    assert k != run_key('e', 'h', {'lr': 0.1}, 1)


def test_selectors_coerce_levels():
    cell = {'lr': 0.01, 'aug': True, 'opt': 'adam'}
    assert selector_matches({'lr': '0.01', 'aug': 'on', 'seed': '2'}, cell, 2)
    assert not selector_matches({'aug': 'off'}, cell, 2)
    assert not selector_matches({'seed': '1'}, cell, 2)
    assert selector_matches({'opt': 'adam'}, cell, 0)


def test_last_line_wins_and_pending(tmp_path):
    from expkit.executor import run_one
    store = Store(tmp_path / 'store')
    spec = RunSpec('e', 'expkit.tests.entries:quadratic', 'h', {'offset': 0.0}, {'lr': 0.1}, 0, 0)
    other = RunSpec('e', 'expkit.tests.entries:quadratic', 'h', {'offset': 0.0}, {'lr': 0.3}, 0, 0)
    out = run_one(spec.to_dict(), store.root)
    store.append_run(out['record'], out['curves'])
    assert store.pending([spec, other]) == [other]
    out['record'].metrics = {'mse': -1.0}
    store.append_run(out['record'], out['curves'])
    assert store.load_records()[0].metrics == {'mse': -1.0}
    assert len(read_jsonl(store.configs_path)) == 1        # the base is stored once
    frame = store.load_frame('h')
    assert list(frame['lr']) == [0.1] and frame['mse'][0] == -1.0
