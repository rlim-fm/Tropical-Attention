# 🌴 Tropical-Attention
**[NeurIPS 2025] Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms.** ([paper](https://arxiv.org/abs/2505.17190))

*Can algebraic geometry enhance the sharpness, robustness, and interpretability of modern neural reasoning models  by equipping them with a mathematically grounded inductive bias?*
To answer this, we introduce Tropical Attention, an attention mechanism grounded in tropical geometry that lifts the attention kernel into tropical projective space, where reasoning is piecewise-linear and 1-Lipschitz, thus preserving the polyhedral decision structure inherent to combinatorial reasoning. We prove that Multi-Head Tropical Attention (MHTA) stacks universally approximate tropical circuits and realize tropical transitive closure through composition, achieving polynomial resource bounds without invoking recurrent mechanisms. These guarantees explain why the induced polyhedral decision boundaries remain sharp and scale-invariant, rather than smoothed by Softmax. Empirically, we show that Tropical Attention delivers stronger out-of-distribution generalization in both length and value, with high robustness against perturbative noise, and substantially faster inference with fewer parameters compared to Softmax-based and recurrent attention baselines, respectively. For the first time, we push the domain of neural algorithmic reasoning beyond **PTIME** problems to **NP-hard/complete** problems, paving the way toward  sharper and more expressive Large Reasoning Models (LRMs) capable of tackling complex combinatorial challenges in Phylogenetics, Cryptography, Particle Physics, and Mathematical Discovery.

---
We experiment on 11 combinatorial tasks:

1. **Floyd–Warshall** - All-pairs shortest paths on a weighted undirected graph. (Both Regression and Classification)
2. **Quickselect** — Find the k-th smallest elements in a set. (Classification)
3. **3SUM (Decision)** — Decide if there exist a, b, c with a+b+c=T. (Classification)
4. **Balanced Partition** - Split numbers into two subsets with equal sum. (NP-complete Classification)
5. **Convex Hull** - Given 2D points, identify the hull. (Classification)
6. **Subset Sum (Decision)** - Decide if some subset sums to a target T. (NP-complete Classification)
7. **0/1 Knapsack** — Maximize value under capacity with binary item choices. (NP-hard Classification)
8. **Fractional Knapsack** — Items can be taken fractionally; predict optimal value. (Regression)
9. **Strongly Connected Components (SCC)** Decompose a directed graph with community structure; predict pairwise same-component. (Classification)
10. **Bin Packing** — Pack items into the fewest bins of fixed capacity. (NP-hard Classification)
11. **Min Coin Change** — Minimum number of coins to reach a target T with each coin used at most once. (Classification)

---

## Tropical kernel

If you just want the **Tropical Attention kernel**, use **`TropicalAttention.py`**.

<summary>Example (instantiate inside your model)</summary>

```python
from TropicalAttention import TropicalAttention
import torch

attn = TropicalAttention(
    d_model=128,
    n_heads=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    tropical_proj=True,
    tropical_norm=False,
    symmetric=True,
)

x = torch.randn(2, 32, 128)     # [batch, seq, d_model]
y, scores = attn(x)
```

---

### Running experiments (expkit)

Every task is posed as **subset selection**: the target is a 0/1 mask over the
input tokens (`dataloaders.py`, `classification=True`). Experiments run through
[expkit](expkit/README.md) from one YAML, `configs/tropical.yaml`, whose grid is

- `model_type`: `vanilla` (softmax), `adaptive` (adaptive-temperature softmax), `tropical`
- `head`: `seq2seq` (a linear logit per token) or `pointer` (the logits are the attention
  scores of a pooled query over the input positions)
- `objective`: `erm` or `fishr` (Fishr, with one environment per training sequence length)

Each dataset is a campaign. Metrics are token F1 and exact-subset match in
distribution, at longer lengths (`ood_len_*`, the headline) and at doubled values (`ood_val_*`).

```bash
pip install -r requirements.txt
python -m expkit run    --config configs/tropical.yaml --profile dry_run --cpu-workers 4   # smoke test
python -m expkit run    --config configs/tropical.yaml --profile Knapsack --gpus 0 --per-gpu 4
python -m expkit report --config configs/tropical.yaml --profile Knapsack --pdf
python -m expkit status --config configs/tropical.yaml
python -m pytest tests expkit/tests
```

The entry is `train.py:run`; tasks and length environments are in `tasks.py`, and Fishr is in `fishr.py`.

---

### Citation

If you use this repository or Tropical Attention in your research, please cite:

```bibtex
@article{hashemi2025tropical,
  title={Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms},
  author={Hashemi, Baran and Pasque, Kurt and Teska, Chris and Yoshida, Ruriko},
  journal={arXiv e-prints},
  pages={arXiv--2505},
  year={2025}
}
```







