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


def test_stacked_penalty_is_each_models_own():
    torch.manual_seed(0)
    per_model = [{L: torch.rand(7) for L in (4, 5, 6)} for _ in range(3)]
    singles = [FishrPenalty(ema=0.9) for _ in per_model]
    stacked = FishrPenalty(ema=0.9)
    for _ in range(3):                                   # several steps, so the EMA matters
        expected = torch.stack([fn(v) for fn, v in zip(singles, per_model)])
        got = stacked({L: torch.stack([v[L] for v in per_model]) for L in (4, 5, 6)})
        assert got.shape == (3,) and torch.allclose(got, expected)
        per_model = [{L: v * 1.5 for L, v in m.items()} for m in per_model]


def test_head_grads_from_a_given_state_match_the_heads_own():
    torch.manual_seed(0)
    model = build_model('tropical', 'pointer', 3, d_model=8, n_heads=2)
    feats, y = torch.randn(5, 6, 8), (torch.rand(5, 6) > 0.5).float()
    state = {n: p.detach().clone() for n, p in model.head.named_parameters()}
    state.update({n: b for n, b in model.head.named_buffers()})
    assert torch.allclose(per_sample_head_grads(model.head, feats, y, state=state),
                          per_sample_head_grads(model.head, feats, y))
