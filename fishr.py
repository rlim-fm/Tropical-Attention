"""Fishr (Rame, Dancette & Cord, ICML 2022): invariant gradient variances across environments.

Here an environment is one sequence length. Following the DomainBed
reference implementation, the penalty matches, across environments, the
variance of the per-sample gradients of the loss with respect to the head's
parameters (the "classifier"). Each environment's variance is smoothed with
an exponential moving average, and the penalty is the mean squared distance
of each environment's variance from their mean, averaged over environments.

Per-sample gradients come from ``torch.func`` (vmap over grad), not BackPACK,
so they work for any head (the seq2seq linear layer or the pointer head). They
stay differentiable with respect to the encoder features, so the penalty
trains the encoder too.
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def per_sample_head_grads(head, feats, y):
    """[B, P]: the gradient of each sample's mean token BCE w.r.t. the head's parameters."""
    names = [n for n, p in head.named_parameters() if p.requires_grad]
    params = {n: p for n, p in head.named_parameters() if p.requires_grad}

    def loss_one(p, f, t):
        logits = functional_call(head, p, (f.unsqueeze(0),)).squeeze(0)
        return F.binary_cross_entropy_with_logits(logits, t)

    g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, feats, y)
    return torch.cat([g[n].reshape(feats.size(0), -1) for n in names], dim=1)


def gradient_variance(grads):
    """[P]: the per-coordinate variance of per-sample gradients [B, P] (biased, as in DomainBed)."""
    return (grads - grads.mean(dim=0, keepdim=True)).pow(2).mean(dim=0)


class FishrPenalty:
    """Exponential moving averages of each environment's gradient variance, and the penalty."""

    def __init__(self, ema: float = 0.95):
        self.ema = float(ema)
        self.state = {}

    def _smooth(self, env, value):
        # DomainBed's MovingAverage(oneminusema_correction=True): starts from zeros, and
        # the 1/(1-ema) factor keeps the gradient through `value` independent of ema.
        prev = self.state.get(env, torch.zeros_like(value))
        out = self.ema * prev + (1 - self.ema) * value
        self.state[env] = out.detach()
        return out / (1 - self.ema)

    def __call__(self, variances):
        """``variances``: {env: [P]} for this step. Returns the scalar penalty."""
        smoothed = {e: self._smooth(e, v) for e, v in variances.items()}
        mean = torch.stack(list(smoothed.values())).mean(dim=0)
        return torch.stack([(v - mean).pow(2).mean() for v in smoothed.values()]).mean()
