import torch
import torch.nn.functional as F

from fishr import FishrPenalty, gradient_variance, per_sample_head_grads
from models import build_model


def _loop_grads(head, feats, y):
    rows = []
    params = [p for p in head.parameters() if p.requires_grad]
    for f, t in zip(feats, y):
        loss = F.binary_cross_entropy_with_logits(head(f.unsqueeze(0)).squeeze(0), t)
        gs = torch.autograd.grad(loss, params, allow_unused=True)
        rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                               for g, p in zip(gs, params)]))
    return torch.stack(rows)


def test_per_sample_grads_match_loop():
    torch.manual_seed(0)
    for model_type in ('vanilla', 'tropical'):
        for head in ('seq2seq', 'pointer'):
            model = build_model(model_type, head, 3, d_model=8, n_heads=2)
            feats = torch.randn(5, 6, 8)
            y = (torch.rand(5, 6) > 0.5).float()
            fast = per_sample_head_grads(model.head, feats, y)
            slow = _loop_grads(model.head, feats, y)
            assert torch.allclose(fast, slow, atol=1e-5), (model_type, head)


def test_penalty_zero_for_identical_envs_and_trains_the_encoder():
    torch.manual_seed(0)
    model = build_model('tropical', 'seq2seq', 3, d_model=8, n_heads=2)
    x = torch.randn(6, 5, 3)
    y = (torch.rand(6, 5) > 0.5).float()
    variances = {}
    for env in (4, 5):
        feats = model.features(x)
        variances[env] = gradient_variance(per_sample_head_grads(model.head, feats, y))
    penalty = FishrPenalty(ema=0.95)(variances)
    assert penalty.item() < 1e-10

    x2 = torch.randn(6, 7, 3) * 3
    variances[5] = gradient_variance(per_sample_head_grads(model.head, model.features(x2), y[:, :1].repeat(1, 7)))
    penalty = FishrPenalty(ema=0.95)(variances)
    assert penalty.item() > 0
    penalty.backward()
    assert model.input_linear.weight.grad is not None
    assert model.input_linear.weight.grad.abs().sum() > 0


def test_ema_matches_domainbed():
    pen = FishrPenalty(ema=0.9)
    a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 0.0])
    pen({0: a, 1: b})
    # second update: ema * prev + (1 - ema) * new, divided by (1 - ema)
    out = pen._smooth(0, a)
    prev = 0.1 * a
    assert torch.allclose(out, (0.9 * prev + 0.1 * a) / 0.1)
