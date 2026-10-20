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
Version **2.0.0** routes the max-plus products through
[`tropical-gemm`](https://github.com/TensorBFS/tropical-gemm) when installed,
using its SIMD CPU backend through the published native extension and its CUDA
DLPack backend when the PyTorch helper is available. If the package is not
installed or the tensor shape/dtype/device is unsupported, the code falls back
to the native PyTorch implementation.

Backend selection is automatic by default. You can force a backend with:

```bash
export TROPICAL_ATTENTION_MAXPLUS_BACKEND=auto           # default
export TROPICAL_ATTENTION_MAXPLUS_BACKEND=torch          # PyTorch fallback
export TROPICAL_ATTENTION_MAXPLUS_BACKEND=tropical_gemm  # fail if unavailable
```

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
[expkit](expkit/README.md) from one YAML, [`configs/tropical.yaml`](configs/tropical.yaml).
Its grid is model × head × objective, with 5 paired seeds per cell:

| factor | levels |
|---|---|
| `model_type` | `vanilla` (softmax), `adaptive` (adaptive-temperature softmax), `tropical` |
| `head` | `seq2seq` (a linear logit per token), `pointer` (the logits are the attention scores of a pooled query over the input positions) |
| `objective` | `erm`, `fishr` (Fishr, with one environment per training sequence length) |

Each dataset is a **campaign**, with its own directory `campaigns/tropical/<dataset>/`.
The metrics are token F1 and exact-subset match:
- in distribution (`id_*`)
- at longer lengths (`ood_len_*`, the headline; per length in `ood_len_*_L<n>`, with 64 for set tasks and 16 nodes for graph tasks as in the paper)
- at larger values (`ood_val_*`, the paper's Table 5 OOD value range per campaign)
- under perturbative noise (`ood_noise_*`: each input perturbed with probability 0.5 by a draw from Table 5's noise range)

The OOD settings are `data.ood_lengths`, `data.ood_value_range`, `data.ood_noise_range` and `data.ood_noise_prob` in `configs/tropical.yaml`.
Unlike the paper, which reports the final checkpoint, every metric is taken at the epoch with the lowest ID validation loss.

The code is split across:
- `train.py`: the expkit entry
- `tasks.py`: the tasks and per-length environments
- `models.py`: the attention modules and heads
- `fishr.py`: the Fishr penalty

#### Workflow

```bash
# 0. Setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Smoke test (minutes on a laptop CPU): tests, then the whole pipeline on a tiny profile
.venv/bin/python -m pytest tests expkit/tests
LOCAL=1 PROFILE=dry_run scripts/tune_all.sh
PROFILE=dry_run scripts/approve_all.sh
LOCAL=1 PROFILE=dry_run scripts/run_all.sh

# 2. Tune: one Slurm job per campaign. Use --plan first to see the sbatch lines.
scripts/tune_all.sh --plan
scripts/tune_all.sh

# 3. Review: every campaign shows UNAPPROVED once its search is done.
expkit/launch.sh status --config configs/tropical.yaml
#    Read campaigns/tropical/<dataset>/tune/tune.csv (one row per candidate), then approve:
scripts/approve_all.sh                 # shows each table and asks once

# 4. Run the grid of every approved campaign, then a report job for each
scripts/run_all.sh --plan
scripts/run_all.sh

# 5. Results: campaigns/tropical/<dataset>/report/ (report.tex, stats_*.csv, figures)
expkit/launch.sh report --config configs/tropical.yaml --profile Knapsack --pdf   # rebuild one
```

`CAMPAIGNS=Knapsack,SCC` limits any script to some campaigns. Re-running a script
only does what is missing: stored runs are never retrained. `run` refuses a
campaign whose `frozen.yaml` is missing or unapproved (`require_frozen: true`).

**What tune does.** The paper does no hyperparameter search. Every kernel shares
the fixed setup in `base`: 1 layer, 2 heads, width 64, lr 1e-4 (App. F and
Table 5). The only new hyperparameter is Fishr's λ. As in Fishr/DomainBed, it
is selected by **training-domain validation**:
- the score is the best ID-validation loss, `id_val_loss`
- the search runs on separate seeds (`tune.global_seed`) at a reduced budget
- one λ is chosen per campaign and shared by every cell

The test and OOD metrics never enter selection (`expkit tune` refuses them).
The search space is in the `tune:` block of the YAML.

**After a code fix** that changes results, the stored runs keep their keys, so
clear them and run again:
```bash
expkit/launch.sh clear  --config configs/tropical.yaml --profile Knapsack --only head=pointer --yes
expkit/launch.sh replay --config configs/tropical.yaml --profile Knapsack --only seed=0   # bit-identical check
```

**Commit** `campaigns/**/results/{runs,configs,registry}.jsonl`, `frozen.yaml` and
`tune/tune.csv`, which hold the record of what ran and why. Curves and
artifacts are gitignored.

#### Compute

expkit parallelizes **across runs, never within one**. Each (cell, seed) is an
independent job in a process pool. There is no data-parallel or multi-GPU
training of a single run, which these models (50k–77k parameters) do not need.

- `--gpus 0,1 --per-gpu 4` starts 8 worker processes, each pinned to one GPU,
  with the allocated CPUs split between them.
- `--cpu-workers N` starts N CPU processes, with the GPUs hidden from them.
- Apple-silicon MPS is not supported: a slot is `cpu` or `cuda:i`.
- On Slurm, each script submits **one single-node job per campaign**:
  - 13 campaigns means 13 jobs in parallel.
  - Each job runs its campaign's runs `PER_GPU` at a time on `GPUS` GPUs.
  - `--dependency=singleton` stops two jobs writing one campaign.
  - Each report job waits on its own campaign's job.
- A single run leaves a GPU mostly idle, so pack several per GPU (`PER_GPU=4`–`8`).
- The graph tasks (FloydWarshall, SCC) are the memory-heavy ones. L nodes are
  L² tokens, and tropical attention holds a B·H·S²·d_k tensor, so their batches
  are small (`batch_size: 8`, `eval_batch_size: 8`).

| variable | default | meaning |
|---|---|---|
| `GPUS` | 1 | GPUs per Slurm job (locally: GPUs to use; 0 means CPU) |
| `PER_GPU` | 4 | concurrent runs per GPU |
| `CORES` | 8 | CPU cores per Slurm job |
| `CPU_WORKERS` | 4 | concurrent CPU runs when there is no GPU |
| `LOCAL` | unset | `1`: run here, campaign after campaign, instead of submitting |
| `CAMPAIGNS` / `PROFILE` | all | which campaigns, or one profile such as `dry_run` |
| `EXPKIT_PARTITION` | unset | Slurm partition(s), e.g. `gpu` |
| `EXPKIT_SBATCH_ARGS` | unset | extra sbatch flags, e.g. `--time=12:00:00 --account=X` |
| `EXPKIT_GB_PER_CORE` | unset | `--mem` = cores × this |

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







