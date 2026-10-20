"""The expkit entry: train one subset-selection model, ERM or Fishr over length environments.

    python -m expkit run --config configs/tropical.yaml --profile dry_run --cpu-workers 4

A run is a function of (base, cell, seeds) (``expkit.api``):

* ``ctx.seeds['data']`` draws every split (each split and length its own seed),
* ``ctx.seeds['init']`` seeds the weights (and dropout),
* ``ctx.seeds['sampler']`` seeds a CPU generator for the batch order.

Cells (``configs/tropical.yaml``): ``model_type`` x ``head`` x ``objective``.
Every step takes one minibatch from each training length. ERM and Fishr see
the same batches, so the objective contrast is seed-paired.

``run_group`` trains several seeds of one cell at once (``expkit run
--group-seeds``): the models are stacked (``torch.func.stack_module_state``)
and one ``vmap``-ed step trains them all, so a small model fills a large GPU
with S times fewer kernel launches. Each seed keeps its own data, init and
batch order, and the optimizer is elementwise, so every seed follows the path
``run`` takes; only the batched kernels' rounding differs.
"""
import copy
import math

import schedulefree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

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


def _splits(ctx):
    """Every split of one seed's data: {split: {length: (X, Y)}}, on the CPU."""
    task, data = ctx.base['task'], ctx.base['data']
    train_lengths = [int(L) for L in data['train_lengths']]
    ood_lengths = [int(L) for L in data.get('ood_lengths') or [2 * max(train_lengths)]]
    n_train, n_eval = int(data['n_per_length']), int(data['n_eval_per_length'])
    seed = ctx.seeds['data']
    # one environment per training length
    return {
        'train': make_envs(task, train_lengths, n_train, seed, 'train'),
        'id_val': make_envs(task, train_lengths, n_eval, seed, 'id_val'),
        'id_test': make_envs(task, train_lengths, n_eval, seed, 'id_test'),
        'ood_len': make_envs(task, ood_lengths, n_eval, seed, 'ood_len'),
        'ood_val': make_envs(task, train_lengths, n_eval, seed, 'ood_val',
                             value_range=ood_value_range(task['name'], task.get('value_range'))),
    }


def _model(ctx, input_dim):
    model_cfg, cell = ctx.base['model'], ctx.cell
    torch.manual_seed(ctx.seeds['init'])
    return build_model(cell['model_type'], cell['head'], input_dim,
                       d_model=int(model_cfg['d_model']), n_heads=int(model_cfg['n_heads']),
                       num_layers=int(model_cfg['num_layers']), dropout=float(model_cfg['dropout']),
                       tropical_impl=model_cfg.get('tropical_impl', 'legacy'), device=ctx.device)


def _metrics(id_m, ood_m, ood_per_length, val_m, curves, best_epoch, best_val):
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
    return metrics


def run(ctx: RunContext) -> RunResult:
    base, cell, device = ctx.base, ctx.cell, ctx.device
    data, optim, fishr_cfg = base['data'], base['optim'], base['fishr']

    # ---- data: one environment per training length ----
    splits = _splits(ctx)
    id_val, id_test, ood_len, ood_val = (splits[k] for k in ('id_val', 'id_test', 'ood_len', 'ood_val'))
    train = {L: (X.to(device), Y.to(device)) for L, (X, Y) in splits['train'].items()}
    input_dim = next(iter(train.values()))[0].size(-1)

    # ---- model ----
    model = _model(ctx, input_dim)
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
        # Summed on the device in float64 (as Python floats were), read once per epoch.
        epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        epoch_pen = torch.zeros((), dtype=torch.float64, device=device)
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
            epoch_loss += erm.detach().double()
            epoch_pen += penalty.detach().double()
        curves['train_loss'].append(epoch_loss.item() / steps_per_epoch)
        curves['fishr_penalty'].append(epoch_pen.item() / steps_per_epoch)
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
    metrics = _metrics(id_m, ood_m, ood_per_length, val_m, curves, best_epoch, best_val)
    return RunResult(metrics=metrics, curves=curves, steps=step)


# ---- several seeds of one cell at once ----------------------------------------------

class _Features(nn.Module):
    """``model.features`` as a module's forward, for ``functional_call``."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.features(x)


def _stack_envs(splits, name, device):
    """{length: (X [S, N, ...], Y [S, N, ...])} of one split over the stacked seeds."""
    return {L: (torch.stack([s[name][L][0] for s in splits]).to(device),
                torch.stack([s[name][L][1] for s in splits]).to(device))
            for L in splits[0][name]}


@torch.no_grad()
def evaluate_group(forward, state, envs, batch_size, limit=None):
    """``evaluate`` for S stacked models: a list of S ``(overall, {length: (f1, exact)})``.

    ``forward(state, x)`` is one model's logits; ``envs`` holds stacked data
    ([S, N, ...]) on the models' device.
    """
    S = next(iter(envs.values()))[0].size(0)
    totals = [[0, 0, 0, 0, 0, 0.0] for _ in range(S)]
    per_length = [{} for _ in range(S)]
    for L, (X, Y) in envs.items():
        X, Y = (X[:, :limit], Y[:, :limit]) if limit else (X, Y)
        n = X.size(1)
        c = [[0, 0, 0, 0] for _ in range(S)]
        loss = [0.0] * S
        for start in range(0, n, batch_size):
            x, y = X[:, start:start + batch_size], Y[:, start:start + batch_size]
            logits = vmap(forward)(state, x)                        # [S, b, L]
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none').sum(dim=(1, 2))
            pred, truth = logits > 0, y > 0.5
            counts = torch.stack([(pred & truth).sum(dim=(1, 2)), (pred & ~truth).sum(dim=(1, 2)),
                                  (~pred & truth).sum(dim=(1, 2)),
                                  (pred == truth).all(dim=2).sum(dim=1)], dim=1).tolist()
            for i, (b, k) in enumerate(zip((bce / y.size(2)).tolist(), counts)):
                loss[i] += b
                c[i] = [a + b_ for a, b_ in zip(c[i], k)]
        for i in range(S):
            per_length[i][L] = (_f1(*c[i][:3]), c[i][3] / n)
            totals[i] = [a + b for a, b in zip(totals[i], c[i] + [n, loss[i]])]
    out = []
    for i in range(S):
        tp, fp, fn, exact, n, loss = totals[i]
        out.append(({'f1': _f1(tp, fp, fn), 'exact': exact / n, 'loss': loss / n}, per_length[i]))
    return out


def run_group(ctxs):
    """Train the seeds ``ctxs`` of one cell together. Returns one ``RunResult`` per
    context, in order, or the exception that seed's run would have raised.

    A seed whose loss turns non-finite gets the result ``run`` returns at that
    epoch, and the others carry on: the models share no tensor, so its NaNs
    stay in its own slice.
    """
    ctxs = list(ctxs)
    first = ctxs[0]
    for ctx in ctxs[1:]:
        if (ctx.base, ctx.cell, ctx.device) != (first.base, first.cell, first.device):
            raise ValueError('run_group trains the seeds of one cell: base, cell and device must match')
    if len(ctxs) == 1 or float(first.base['model']['dropout']) > 0:
        # Dropout would need per-seed randomness under vmap; one seed needs no stacking.
        return [run(ctx) for ctx in ctxs]
    base, cell, device = first.base, first.cell, first.device
    data, optim, fishr_cfg = base['data'], base['optim'], base['fishr']
    S = len(ctxs)

    splits = [_splits(ctx) for ctx in ctxs]
    train = _stack_envs(splits, 'train', device)
    id_val, id_test, ood_len, ood_val = (_stack_envs(splits, k, device)
                                         for k in ('id_val', 'id_test', 'ood_len', 'ood_val'))
    input_dim = next(iter(train.values()))[0].size(-1)

    models = [_model(ctx, input_dim) for ctx in ctxs]
    params, buffers = stack_module_state(models)          # leaves [S, ...]
    skeleton = copy.deepcopy(models[0]).to('meta').eval()  # structure only; eval: no dropout
    del models
    features = _Features(skeleton)

    def split_state(p, b):
        state = {**p, **b}
        head = {k[len('head.'):]: v for k, v in state.items() if k.startswith('head.')}
        return {f'model.{k}': v for k, v in state.items()}, head

    def forward(state, x):
        full, head = split_state(*state)
        return functional_call(skeleton.head, head, (functional_call(features, full, (x,)),))

    use_fishr = cell['objective'] == 'fishr'
    if cell['objective'] not in ('erm', 'fishr'):
        raise ValueError(f"Unknown objective {cell['objective']!r}")

    def seed_step(p, b, xs, ys):
        full, head = split_state(p, b)
        losses, variances = [], []
        for x, y in zip(xs, ys):
            feats = functional_call(features, full, (x,))
            losses.append(F.binary_cross_entropy_with_logits(functional_call(skeleton.head, head, (feats,)), y))
            grads = per_sample_head_grads(skeleton.head, feats if use_fishr else feats.detach(), y,
                                          state=head)
            variances.append(gradient_variance(grads if use_fishr else grads.detach()))
        return torch.stack(losses).mean(), torch.stack(variances)

    step_all = vmap(seed_step)
    lr = float(optim['lr'])

    def optimizer():
        opt = schedulefree.RAdamScheduleFree(list(params.values()), lr=lr)
        opt.train()
        return opt

    opt = optimizer()
    penalty_fn = FishrPenalty(ema=float(fishr_cfg['ema']))
    lam, anneal = float(fishr_cfg['lambda']), int(fishr_cfg['anneal_iters'])

    orders = [torch.Generator().manual_seed(ctx.seeds['sampler']) for ctx in ctxs]
    bs = int(optim['batch_size'])           # per environment
    eval_bs = int(optim.get('eval_batch_size', 256))
    steps_per_epoch = max(1, min(X.size(1) for X, _ in train.values()) // bs)
    curve_limit = int(data.get('curve_eval_limit', 256))
    rows = torch.arange(S, device=device).unsqueeze(1)

    curves = [{'train_loss': [], 'fishr_penalty': [], 'id_val_loss': [], 'ood_len_f1': []}
              for _ in range(S)]
    results = [None] * S                     # filled as each seed finishes
    best_val, best_epoch = [math.inf] * S, [-1] * S
    best_state = {k: v.detach().clone() for k, v in params.items()}
    alive = torch.ones(S, dtype=torch.bool, device=device)
    step = 0
    for epoch in range(int(optim['epochs'])):
        # Each seed draws its permutations from its own generator, in the order `run` does.
        perms = [{L: torch.randperm(X.size(1), generator=g) for L, (X, _) in train.items()}
                 for g in orders]
        perms = {L: torch.stack([p[L] for p in perms]).to(device) for L in train}
        epoch_loss = torch.zeros(S, dtype=torch.float64, device=device)
        epoch_pen = torch.zeros(S, dtype=torch.float64, device=device)
        for s in range(steps_per_epoch):
            if use_fishr and step == anneal and fishr_cfg.get('reset_optimizer', True):
                opt = optimizer()          # as DomainBed does when the penalty switches on
            xs, ys = [], []
            for L, (X, Y) in train.items():
                idx = perms[L][:, s * bs:(s + 1) * bs]
                xs.append(X[rows, idx])
                ys.append(Y[rows, idx])
            erm, var = step_all(params, buffers, tuple(xs), tuple(ys))   # [S], [S, E, P]
            penalty = penalty_fn({L: var[:, e] for e, L in enumerate(train)})
            weight = lam if (use_fishr and step >= anneal) else 0.0
            loss = erm + weight * penalty if use_fishr else erm
            opt.zero_grad(set_to_none=True)
            # A finished (diverged) seed contributes nothing; the sum keeps the
            # seeds' gradients apart, each its own run's.
            torch.where(alive, loss, torch.zeros_like(loss)).sum().backward()
            opt.step()
            step += 1
            epoch_loss += erm.detach().double()
            epoch_pen += penalty.detach().double()
        live = alive.tolist()
        for i, (tl, fp) in enumerate(zip(epoch_loss.tolist(), epoch_pen.tolist())):
            if not live[i]:
                continue
            curves[i]['train_loss'].append(tl / steps_per_epoch)
            curves[i]['fishr_penalty'].append(fp / steps_per_epoch)
            if not math.isfinite(curves[i]['train_loss'][-1]):
                results[i] = RunResult(curves=copy.deepcopy(curves[i]), diverged=True, steps=step)
                live[i] = False
        alive = torch.tensor(live, dtype=torch.bool, device=device)
        if not any(live):
            break

        # ---- training-domain validation; schedule-free evaluates its averaged weights ----
        opt.eval()
        state = (params, buffers)
        vals = evaluate_group(forward, state, id_val, eval_bs)
        oods = evaluate_group(forward, state, ood_len, eval_bs, curve_limit)
        improved = []
        for i in range(S):
            if not live[i]:
                continue
            val = vals[i][0]
            curves[i]['id_val_loss'].append(val['loss'])
            curves[i]['ood_len_f1'].append(oods[i][0]['f1'])
            if val['loss'] < best_val[i]:
                best_val[i], best_epoch[i] = val['loss'], epoch
                improved.append(i)
        if improved:
            with torch.no_grad():
                for k, v in params.items():
                    best_state[k][improved] = v[improved]
        opt.train()

    # ---- final metrics at each seed's selected epoch ----
    opt.eval()
    state = (best_state, buffers)
    id_ms = evaluate_group(forward, state, id_test, eval_bs)
    ood_ms = evaluate_group(forward, state, ood_len, eval_bs)
    val_ms = evaluate_group(forward, state, ood_val, eval_bs)
    for i in range(S):
        if results[i] is not None:
            continue
        if best_epoch[i] < 0:
            results[i] = RuntimeError('no epoch had a finite validation loss to select')
            continue
        metrics = _metrics(id_ms[i][0], ood_ms[i][0], ood_ms[i][1], val_ms[i][0], curves[i],
                           best_epoch[i], best_val[i])
        results[i] = RunResult(metrics=metrics, curves=curves[i], steps=step)
    return results
