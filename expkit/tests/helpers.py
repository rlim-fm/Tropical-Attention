"""Shared by the tests."""


def write_config(path, entry='expkit.tests.entries:quadratic', **extra):
    import yaml
    doc = {'experiment': 'quad', 'entry': entry, 'global_seed': 0, 'seeds': 3,
           'base': {'offset': 0.0}, 'factors': {'lr': [0.01, 0.1, 0.3]},
           'control': {'lr': 0.1}, 'metrics': {'mse': {'direction': 'min'}},
           'headline': 'mse', 'replay_curves': ['train_loss']}
    doc.update(extra)
    with open(path, 'w') as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    return str(path)
