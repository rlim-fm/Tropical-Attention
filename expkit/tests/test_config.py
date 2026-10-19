import math

import pytest

from expkit.config import (base_hash, campaign_names, canonical_json, cell_label, config_hash,
                           factor_grid, load_config, parse_assignments, with_overrides)
from expkit.tests.helpers import write_config


def test_hash_ignores_key_order_and_spells_inf():
    assert config_hash({'a': 1, 'b': [1, 2]}) == config_hash({'b': [1, 2], 'a': 1})
    assert canonical_json({'T': math.inf}) == '{"T":"inf"}'


def test_optional_settings_at_default_leave_the_hash_alone():
    base = {'optim': {'lr': 0.1}}
    optional = {'optim.weight_decay': 0.0}
    assert base_hash(base, optional) == base_hash(with_overrides(base, {'optim.weight_decay': 0.0}),
                                                  optional)
    assert base_hash(base, optional) != base_hash(with_overrides(base, {'optim.weight_decay': 1e-4}),
                                                  optional)


def test_factor_grid_is_the_full_factorial_minus_exclusions():
    cells = factor_grid({'lr': [0.1, 1.0], 'aug': [True, False]}, exclude=[{'lr': 1.0, 'aug': 'off'}])
    assert len(cells) == 3
    assert {'lr': 1.0, 'aug': False} not in cells
    assert cell_label(cells[0]) == 'lr=0.1 aug=on'


def test_campaign_axes_cross_and_merge_in_order(tmp_path):
    axes = {'noise': {'clean': {}, 'noisy': {'base': {'offset': 1.0}}},
            'method': {'plain': {}, 'big': {'base': {'offset': 2.0}, 'seeds': 7}}}
    path = write_config(tmp_path / 'q.yaml', axes=axes,
                        profiles={'dry': {'seeds': 1}})
    raw = {'axes': axes}
    assert campaign_names(raw) == ['clean/plain', 'clean/big', 'noisy/plain', 'noisy/big']
    assert load_config(path).campaign == 'clean/plain'
    c = load_config(path, 'noisy/big')
    assert c.base['offset'] == 2.0 and c.seeds == 7      # method merged over noise
    assert load_config(path, 'dry').seeds == 1
    with pytest.raises(KeyError):
        load_config(path, 'noisy/nope')


def test_assignments_are_yaml_typed():
    assert parse_assignments(['a.b=1e-3', 'c=true', 'd=relu']) == {'a.b': 0.001, 'c': True, 'd': 'relu'}


def test_exponent_floats_load_as_floats(tmp_path):
    path = write_config(tmp_path / 'q.yaml')
    with open(path, 'a') as fh:
        fh.write('extra_lr: 1e-3\n')
    assert load_config(path).extra['extra_lr'] == 0.001


def test_control_must_name_a_factor(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path / 'q.yaml', control={'nope': 1}))
