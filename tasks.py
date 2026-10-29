"""Task registry and sequence-length environments.

Every task is a subset-selection problem: an instance is a sequence of S
tokens ``x`` (S, d) and its target is a 0/1 mask ``y`` (S,) marking the
selected tokens (``classification=True`` in ``dataloaders``).

An *environment* is one sequence length L. All its instances have the same
number of tokens (L for set tasks, L*L for the graph tasks), so a batch never
needs padding.
"""
import torch

from dataloaders import (
    BalancedPartitionDataset, BinPackingDataset, ConvexHullDataset, FloydWarshallDataset,
    FractionalKnapsackDataset, KnapsackDataset, LISDataset, MaxSubsetSumDataset,
    MinCoinChangeDataset, QuickselectDataset, SCCDataset, SubsetSumDecisionDataset,
    ThreeSumDecisionDataset,
)

# name -> (dataset class, graph task: one token per node pair)
TASKS = {
    'SubsetSumDecisionDataset': (SubsetSumDecisionDataset, False),
    'MaxSubsetSumDataset': (MaxSubsetSumDataset, False),
    'KnapsackDataset': (KnapsackDataset, False),
    'FractionalKnapsackDataset': (FractionalKnapsackDataset, False),
    'MinCoinChangeDataset': (MinCoinChangeDataset, False),
    'QuickselectDataset': (QuickselectDataset, False),
    'BalancedPartitionDataset': (BalancedPartitionDataset, False),
    'BinPackingDataset': (BinPackingDataset, False),
    'ConvexHullDataset': (ConvexHullDataset, False),
    'ThreeSumDecisionDataset': (ThreeSumDecisionDataset, False),
    'LISDataset': (LISDataset, False),
    'FloydWarshallDataset': (FloydWarshallDataset, True),
    'SCCDataset': (SCCDataset, True),
}

RANGE_KEYS = ('value_range', 'target_range', 'weight_range', 'adversarial_range')

# Split ids, folded into the data seed so every split is its own draw.
SPLITS = {'train': 0, 'id_val': 1, 'id_test': 2, 'ood_len': 3, 'ood_val': 4, 'ood_noise': 5,
          'ood_far': 6}


def tokens_per_length(name, length):
    return length * length if TASKS[name][1] else length


def _as_range(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def ood_value_range(name, value_range, override=None):
    """The value range of the value-shift test.

    ``override`` is the campaign's ``data.ood_value_range`` (Hashemi et al. 2025,
    Table 5, "OOD Value Range"). Tasks the table leaves out fall back to doubling
    the training range, as the original evaluation did.
    """
    if override is not None:
        return _as_range(override)
    if name == 'SCCDataset':
        return 0.6
    lo, hi = value_range
    return (lo * 2 if lo not in (0, 1) else lo, hi * 2)


def ood_noise_settings(data):
    """Dataset overrides of the perturbative-noise test (Hashemi et al. 2025, App. F).

    Each input is perturbed with probability ``data.ood_noise_prob`` (0.5 in the
    paper) by a random draw from ``data.ood_noise_range`` (Table 5, "Perturbative
    Noise Range"); a null range keeps the task's own ``adversarial_range``.
    """
    out = {'noise_prob': float(data.get('ood_noise_prob', 0.5))}
    if data.get('ood_noise_range') is not None:
        out['adversarial_range'] = _as_range(data['ood_noise_range'])
    return out


def split_seed(data_seed, split, length):
    return (int(data_seed) + 7919 * SPLITS[split] + 104729 * int(length)) % 2 ** 32


def make_env(task, length, n_samples, seed, value_range=None, **overrides):
    """One environment: ``(X, Y)`` with X (N, S, d) float and Y (N, S) float 0/1.

    ``overrides`` replace task settings for this split only (e.g. ``noise_prob``).
    """
    cls, _ = TASKS[task['name']]
    # A null setting (a campaign clearing the default task's) means the dataset's own default.
    kwargs = {k: _as_range(v) for k, v in task.items() if k != 'name' and v is not None}
    if value_range is not None:
        kwargs['value_range'] = value_range
    kwargs.update(overrides)
    ds = cls(n_samples=int(n_samples), length_range=(int(length), int(length)),
             classification=True, seed=int(seed), **kwargs)
    xs, ys = zip(*ds.data)
    X = torch.stack([x.float() for x in xs])
    Y = torch.stack([y.float().reshape(-1) for y in ys])
    expected = tokens_per_length(task['name'], length)
    if X.shape[1] != expected or Y.shape[1] != expected:
        raise ValueError(f"{task['name']} at length {length}: got X {tuple(X.shape)}, "
                         f"Y {tuple(Y.shape)}, expected {expected} tokens")
    return X, Y


def make_envs(task, lengths, n_samples, data_seed, split, value_range=None, **overrides):
    """``{length: (X, Y)}`` for one split."""
    return {int(L): make_env(task, L, n_samples, split_seed(data_seed, split, L), value_range,
                             **overrides)
            for L in lengths}
