import pytest
import torch

from models import TropicalAttention, VanillaAttention, build_model


@pytest.mark.parametrize('model_type', ['vanilla', 'adaptive', 'tropical'])
@pytest.mark.parametrize('head', ['seq2seq', 'pointer'])
@pytest.mark.parametrize('impl', ['legacy', 'kernel'])
def test_heads_give_one_logit_per_token(model_type, head, impl):
    torch.manual_seed(0)
    model = build_model(model_type, head, 3, d_model=8, n_heads=2, num_layers=2, tropical_impl=impl)
    for S in (4, 9):
        out = model(torch.randn(5, S, 3))
        assert out.shape == (5, S) and torch.isfinite(out).all()


def test_layers_do_not_share_attention():
    model = build_model('tropical', 'seq2seq', 3, d_model=8, n_heads=2, num_layers=2)
    a, b = (layer.attn for layer in model.encoder.layers)
    assert a is not b


def test_tropical_scores_match_forward():
    torch.manual_seed(0)
    attn = TropicalAttention(8, 2, 'cpu')
    x = torch.randn(3, 5, 8)
    _, merged = attn(x)                             # [B*H, S, S]
    assert torch.allclose(attn.scores(x, x).reshape(6, 5, 5), merged)


def test_vanilla_scores_cross_lengths():
    attn = VanillaAttention(8, 2)
    assert attn.scores(torch.randn(3, 1, 8), torch.randn(3, 6, 8)).shape == (3, 2, 1, 6)
