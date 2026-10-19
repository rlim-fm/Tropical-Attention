"""A toy experiment: a small MLP fit to a noisy sine, over learning rate x batch size.

It shows the whole ``entry`` contract (``expkit.api``) in about sixty lines:
every source of randomness comes from ``ctx.seeds``, the batch order from a
CPU generator, and the result is a function of (base, cell, seeds), so
``expkit replay`` reproduces it bit for bit.
"""
import math

import torch

from expkit.api import RunContext, RunResult


def make_data(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    n, noise = int(cfg['n']), float(cfg.get('noise', 0.0))
    x = torch.rand(n, 1, generator=g) * 2 * math.pi - math.pi
    y = torch.sin(x) + noise * torch.randn(n, 1, generator=g)
    x_eval = torch.linspace(-math.pi, math.pi, 512).unsqueeze(1)
    return x, y, x_eval, torch.sin(x_eval)


def make_model(cfg):
    layers, width = [], int(cfg['width'])
    for i in range(int(cfg['depth'])):
        layers += [torch.nn.Linear(1 if i == 0 else width, width), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(width, 1))


def run(ctx: RunContext) -> RunResult:
    base, cell = ctx.base, ctx.cell
    x, y, x_eval, y_eval = make_data(base['data'], ctx.seeds['data'])
    torch.manual_seed(ctx.seeds['init'])
    model = make_model(base['model']).to(ctx.device)
    optim = torch.optim.Adam(model.parameters(), lr=float(cell['lr']),
                             weight_decay=float(base['optim'].get('weight_decay', 0.0)))
    order = torch.Generator().manual_seed(ctx.seeds['sampler'])
    x, y = x.to(ctx.device), y.to(ctx.device)
    x_eval, y_eval = x_eval.to(ctx.device), y_eval.to(ctx.device)
    B, steps = int(cell['batch_size']), 0
    train_loss, eval_loss = [], []
    for epoch in range(int(base['optim']['epochs'])):
        perm = torch.randperm(len(x), generator=order).to(ctx.device)
        total = 0.0
        for start in range(0, len(x), B):
            idx = perm[start:start + B]
            loss = torch.mean((model(x[idx]) - y[idx]) ** 2)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.detach()) * len(idx)
            steps += 1
        train_loss.append(total / len(x))
        with torch.no_grad():
            eval_loss.append(float(torch.mean((model(x_eval) - y_eval) ** 2)))
        ctx.hook('on_epoch_end', epoch, model)
        if not math.isfinite(train_loss[-1]):
            return RunResult(curves={'train_loss': train_loss, 'eval_loss': eval_loss},
                             diverged=True, steps=steps)
    mse = eval_loss[-1]
    r2 = 1 - mse / float(torch.var(y_eval, unbiased=False))
    return RunResult(metrics={'mse': mse, 'r2': r2},
                     curves={'train_loss': train_loss, 'eval_loss': eval_loss}, steps=steps)
