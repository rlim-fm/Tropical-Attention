"""Tiny numpy-only entries for the tests: no torch, milliseconds per run."""
import os

import numpy as np

from expkit.api import RunResult


def quadratic(ctx):
    """metric = (lr - 0.1)^2 + noise from the data seed; a curve from the sampler seed."""
    rng = np.random.default_rng(ctx.seeds['data'])
    lr = float(ctx.cell['lr'])
    mse = (lr - 0.1) ** 2 + 0.01 * rng.standard_normal() ** 2 + float(ctx.base.get('offset', 0))
    curve = np.random.default_rng(ctx.seeds['sampler']).random(5) * lr
    ctx.hook('on_epoch_end', 0)
    return RunResult(metrics={'mse': mse}, curves={'train_loss': curve}, steps=5)


def fails_once(ctx):
    """Raises on the first attempt of each run (a marker file per key), then succeeds."""
    marker = os.path.join(ctx.base['marker_dir'], f"{ctx.cell['lr']}-{ctx.seed_index}")
    if not os.path.exists(marker):
        open(marker, 'w').close()
        raise RuntimeError('transient failure')
    return quadratic(ctx)


def always_fails(ctx):
    raise ValueError('broken')


def dies_on(ctx):
    """The worker process dies outright (as under the OOM killer) on lr = 0.3, seed 0."""
    if float(ctx.cell['lr']) == 0.3 and ctx.seed_index == 0:
        os._exit(9)
    return quadratic(ctx)
