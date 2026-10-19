"""The expkit entry: train one subset-selection model, ERM or Fishr over length environments.

    python -m expkit run --config configs/tropical.yaml --profile dry_run --cpu-workers 4

A run is a function of (base, cell, seeds) (``expkit.api``):

* ``ctx.seeds['data']`` draws every split (each split and length its own seed),
* ``ctx.seeds['init']`` seeds the weights (and dropout),
* ``ctx.seeds['sampler']`` seeds a CPU generator for the batch order.

Cells (``configs/tropical.yaml``): ``model_type`` x ``head`` x ``objective``.
Every step takes one minibatch from each training length. ERM and Fishr see
the same batches, so the objective contrast is seed-paired.
"""
import copy
import math

import schedulefree
import torch
import torch.nn.functional as F

from expkit.api import RunContext, RunResult
from fishr import FishrPenalty, gradient_variance, per_sample_head_grads
from models import build_model
from tasks import make_envs, ood_value_range


def _optimizer(model, lr):
    opt = schedulefree.RAdamScheduleFree(model.parameters(), lr=lr)
    opt.train()
    return opt


def _counts(logits, y):
    pred = logits > 0
    truth = y > 0.5
    tp = (pred & truth).sum().item()
    fp = (pred & ~truth).sum().item()
    fn = (~pred & truth).sum().item()
    exact = (pred == truth).all(dim=1).sum().item()
    return tp, fp, fn, exact


def _f1(tp, fp, fn):
    # A split with nothing to select and nothing selected is perfect, not 0.
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


@torch.no_grad()
def evaluate(model, envs, batch_size, device, limit=None):
    """Token-level F1 (micro over every token of every length), exact-subset rate, mean BCE.

    Returns ``(overall, {length: (f1, exact)})``.
    """
    model.eval()
    totals, per_length = [0, 0, 0, 0, 0, 0.0], {}
    for L, (X, Y) in envs.items():
        X, Y = (X[:limit], Y[:limit]) if limit else (X, Y)
        c = [0, 0, 0, 0]
        loss = 0.0
        for start in range(0, len(X), batch_size):
            x, y = X[start:start + batch_size].to(device), Y[start:start + batch_size].to(device)
            logits = model(x)
            loss += F.binary_cross_entropy_with_logits(logits, y, reduction='sum').item() / y.size(1)
            c = [a + b for a, b in zip(c, _counts(logits, y))]
        per_length[L] = (_f1(*c[:3]), c[3] / len(X))
        totals = [a + b for a, b in zip(totals, c + [len(X), loss])]
    tp, fp, fn, exact, n, loss = totals
    model.train()
    return {'f1': _f1(tp, fp, fn), 'exact': exact / n, 'loss': loss / n}, per_length


def run(ctx: RunContext) -> RunResult:
    base, cell, device = ctx.base, ctx.cell, ctx.device
    task, data, model_cfg, optim, fishr_cfg = (base['task'], base['data'], base['model'],
                                               base['optim'], base['fishr'])
    train_lengths = [int(L) for L in data['train_lengths']]
    ood_lengths = [int(L) for L in data.get('ood_lengths') or [2 * max(train_lengths)]]
    n_train, n_eval = int(data['n_per_length']), int(data['n_eval_per_length'])
    seed = ctx.seeds['data']

    # ---- data: one environment per training length ----
    train = make_envs(task, train_lengths, n_train, seed, 'train')
    id_val = make_envs(task, train_lengths, n_eval, seed, 'id_val')
    id_test = make_envs(task, train_lengths, n_eval, seed, 'id_test')
    ood_len = make_envs(task, ood_lengths, n_eval, seed, 'ood_len')
    ood_val = make_envs(task, train_lengths, n_eval, seed, 'ood_val',
                        value_range=ood_value_range(task['name'], task.get('value_range')))
    train = {L: (X.to(device), Y.to(device)) for L, (X, Y) in train.items()}
    input_dim = next(iter(train.values()))[0].size(-1)

    # ---- model ----
    torch.manual_seed(ctx.seeds['init'])
    model = build_model(cell['model_type'], cell['head'], input_dim,
                        d_model=int(model_cfg['d_model']), n_heads=int(model_cfg['n_heads']),
                        num_layers=int(model_cfg['num_layers']), dropout=float(model_cfg['dropout']),
                        tropical_impl=model_cfg.get('tropical_impl', 'legacy'), device=device)
    lr = float(optim['lr'])
    opt = _optimizer(model, lr)

    use_fishr = cell['objective'] == 'fishr'
    if cell['objective'] not in ('erm', 'fishr'):
        raise ValueError(f"Unknown objective {cell['objective']!r}")
    penalty_fn = FishrPenalty(ema=float(fishr_cfg['ema']))
    lam, anneal = float(fishr_cfg['lambda']), int(fishr_cfg['anneal_iters'])

    order = torch.Generator().manual_seed(ctx.seeds['sampler'])
    bs = int(optim['batch_size'])           # per environment
    eval_bs = int(optim.get('eval_batch_size', 256))
    steps_per_epoch = max(1, min(len(X) for X, _ in train.values()) // bs)
    curve_limit = int(data.get('curve_eval_limit', 256))

    curves = {'train_loss': [], 'fishr_penalty': [], 'id_val_loss': [], 'ood_len_f1': []}
    best_val, best_state, best_epoch, step = math.inf, None, -1, 0
    for epoch in range(int(optim['epochs'])):
        perms = {L: torch.randperm(len(X), generator=order) for L, (X, _) in train.items()}
        epoch_loss, epoch_pen = 0.0, 0.0
        for s in range(steps_per_epoch):
            if use_fishr and step == anneal and fishr_cfg.get('reset_optimizer', True):
                opt = _optimizer(model, lr)          # as DomainBed does when the penalty switches on
            losses, variances = [], {}
            for L, (X, Y) in train.items():
                idx = perms[L][s * bs:(s + 1) * bs].to(X.device)
                x, y = X[idx], Y[idx]
                feats = model.features(x)
                losses.append(F.binary_cross_entropy_with_logits(model.head(feats), y))
                # ERM computes the penalty too (on detached features) so the two can be compared.
                grads = per_sample_head_grads(model.head, feats if use_fishr else feats.detach(), y)
                variances[L] = gradient_variance(grads if use_fishr else grads.detach())
            erm = torch.stack(losses).mean()
            penalty = penalty_fn(variances)
            weight = lam if (use_fishr and step >= anneal) else 0.0
            loss = erm + weight * penalty if use_fishr else erm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            epoch_loss += erm.item()
            epoch_pen += penalty.item()
        curves['train_loss'].append(epoch_loss / steps_per_epoch)
        curves['fishr_penalty'].append(epoch_pen / steps_per_epoch)
        if not math.isfinite(curves['train_loss'][-1]):
            return RunResult(curves=curves, diverged=True, steps=step)

        # ---- training-domain validation; schedule-free evaluates its averaged weights ----
        opt.eval()
        val = evaluate(model, id_val, eval_bs, device)[0]
        curves['id_val_loss'].append(val['loss'])
        curves['ood_len_f1'].append(evaluate(model, ood_len, eval_bs, device, curve_limit)[0]['f1'])
        if val['loss'] < best_val:
            best_val, best_epoch = val['loss'], epoch
            best_state = copy.deepcopy(model.state_dict())
        opt.train()

    # ---- final metrics at the selected epoch ----
    opt.eval()
    model.load_state_dict(best_state)
    id_m = evaluate(model, id_test, eval_bs, device)[0]
    ood_m, ood_per_length = evaluate(model, ood_len, eval_bs, device)
    val_m = evaluate(model, ood_val, eval_bs, device)[0]
    metrics = {
        'id_f1': id_m['f1'], 'id_exact': id_m['exact'],
        'ood_len_f1': ood_m['f1'], 'ood_len_exact': ood_m['exact'],
        'ood_val_f1': val_m['f1'], 'ood_val_exact': val_m['exact'],
        'fishr_penalty_final': curves['fishr_penalty'][-1],
        'best_epoch': best_epoch,
        'id_val_loss': best_val,             # the model-selection criterion; what `tune` selects on
    }
    for L, (f1, exact) in ood_per_length.items():
        metrics[f'ood_len_f1_L{L}'] = f1
        metrics[f'ood_len_exact_L{L}'] = exact
    return RunResult(metrics=metrics, curves=curves, steps=step)
