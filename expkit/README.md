# expkit

Experiment infrastructure with nothing about any particular experiment. It
runs a factorial grid of cells × paired seeds in parallel worker processes, into
an append-only store. Re-running is a no-op, and a changed config becomes a new
set of runs. A grid search over hyperparameters (`tune`) proposes a frozen
config, and an approval gate sits before the grid spends its budget. Runs are
reproduced by replay, never checkpointed. You get seed-paired statistics, a
LaTeX report, and one Slurm job per campaign.

You write one function (the *entry*) and one YAML file. expkit does the rest.

## Paste it in

```bash
cp -R path/to/expkit your-repo/          # the folder sits at the repo root
pip install -r your-repo/expkit/requirements.txt
python -m pytest your-repo/expkit/tests  # ~30 s; the example test needs torch
```

Commands run from the repo root as `python -m expkit ...`, or through
`expkit/launch.sh ...`, which uses `.venv/` if there is one and sets
`PYTHONPATH`. Outputs go to `campaigns/` at the repo root. Override that with
`EXPKIT_CAMPAIGNS` or `--root`.

Add to your `.gitignore`. These are bulky, and only replay and the grand
report's loss curves read them:

```gitignore
campaigns/**/results/curves.jsonl
campaigns/**/results/curves.jsonl.cleared-*
campaigns/**/results/artifacts/
campaigns/**/results/*.tmp-*
slurm/*.out
```

Track `runs.jsonl`, `configs.jsonl`, `registry.jsonl` and `frozen.yaml`. They
are what the statistics read, and they diff as appended lines.

## Try the example

```bash
python -m expkit run    --config expkit/examples/toy.yaml --profile dry_run --cpu-workers 4
python -m expkit run    --config expkit/examples/toy.yaml --profile dry_run   # 0 to run
python -m expkit replay --config expkit/examples/toy.yaml --profile dry_run --only seed=0
python -m expkit grand  --config expkit/examples/toy.yaml
python -m expkit status --config expkit/examples/toy.yaml
```

`examples/toy.py` is the whole entry contract in about 60 lines. `examples/toy.yaml`
uses every config feature.

## Your experiment

**The entry** (`expkit/api.py`) is called once per run:

```python
from expkit.api import RunContext, RunResult

def run(ctx: RunContext) -> RunResult:
    # ctx.base: the frozen config (dict); ctx.cell: {factor: level}
    # ctx.seeds: {'init', 'data', 'sampler'} -> int; ctx.device: 'cpu' / 'cuda:0'
    ...
    return RunResult(metrics={'acc': acc}, curves={'train_loss': losses}, diverged=False)
```

**The YAML** (full reference in the `expkit/config.py` docstring):

```yaml
experiment: mnist_lr
entry: myproject.train:run
seeds: 10
base: {model: {width: 256}, optim: {epochs: 20}}
factors: {lr: [1e-4, 3e-4, 1e-3], batch_size: [64, 256]}
control: {lr: 3e-4}
metrics: {acc: {direction: max, bounds: [0, 1]}}
headline: acc
analysis: {trend: {factor: lr, transform: log10}}
axes:      # campaigns, crossed: clean/plain, clean/aug, noisy/plain, noisy/aug
  data:   {clean: {}, noisy: {base: {label_noise: 0.2}}}
  method: {plain: {}, aug: {base: {augment: true}}}
profiles: {dry_run: {seeds: 2, base: {optim: {epochs: 1}}}}
```

## Commands

| command | does |
|---|---|
| `run` | trains every missing run (`--cpu-workers N`, or `--gpus 0,1 --per-gpu 2`; `--group-seeds`; `--only lr=1e-3,seed=0`; `--force`; `--plan`) |
| `status` | for every campaign: frozen and approved?, runs stored of total, failures |
| `tune [--plan]` | grid-searches the YAML's `tune:` block on its own seeds and store, writes `tune/tune.csv`, and freezes the best candidate, unapproved (see `expkit/tune.py`) |
| `freeze --set k=v` / `approve` | writes `frozen.yaml` with tuned values, unapproved; `approve` after you have read the evidence |
| `freeze --profile P --from-default` | derives a campaign from the approved default, and refuses if the profile moves a tuned value |
| `grand [--no-pdf]` | the report over every campaign (`campaigns/<experiment>/grand/`): a colored table per combination of the non-control factors, per-campaign length and loss figures, `grand_*.csv` |
| `replay --keys K` | retrains stored runs and fails unless their curves match exactly (`--tolerance`) |
| `clear [--only ...] --yes` | removes runs so they retrain after a fix that leaves their keys unchanged |
| `submit [--stage tune\|run] --cores 32 [--gpus 1 --per-gpu 4 --gpu-type T --cpus-per-gpu K] [--cpu-partition P --cpu-cores N] [--group-seeds] [--grand] [--plan]` | one `sbatch` job per campaign with runs missing, longest first; `--grand` adds one grand-report job that waits on all of them |

Pass the same `--profile` to every command. It selects the campaign directory
`campaigns/<experiment>/<campaign>/` (`frozen.yaml`, `results/`).

## The invariants (keep them, or things break silently)

- **Seeds come from `(global_seed, seed_index)` only, never from the cell.** Cells
  with the same seed index share their data and initialization. Every paired
  test in `stats.py` relies on this.
- **A run is a function of (base, cell, seeds).** Draw batch orders from a CPU
  generator seeded from `ctx.seeds`, and let hooks only observe. `replay`
  asserts that the curves are bit-identical on the same device and library
  versions.
- **The key is (experiment, frozen hash, cell, seed).** A code fix that changes
  results does not change keys, so `clear` the affected runs. Adding a knob to
  `base` *does* move the hash. Declare it under `optional:` with its default,
  and the stored runs stay valid.
- **One writer, no lock.** Workers return records. The parent process appends
  them. Never point two `run` processes at the same campaign (`submit` uses
  `--dependency=singleton` for this). Never add `flock`: on an NFS store it
  hung every job silently.
- **Threads come from the allocation, not the machine** (`threads.py`). Under
  Slurm, `os.cpu_count()` oversubscribes the cgroup. A CPU-only launch hides
  the GPUs so that torch does not open a CUDA context in every worker.
- **Never put raw Unicode math in .tex.** Use the `num`/`level` filters.

## Slurm

`submit` reads its settings from flags or the environment: `EXPKIT_PARTITION`
(e.g. `main,mixed`), `EXPKIT_GB_PER_CORE`, `EXPKIT_JOB_PREFIX` (default `xk`)
and `EXPKIT_SBATCH_ARGS` (e.g. `--time=12:00:00`). Run `--plan` first to print
the `sbatch` lines.

A campaign's `slurm:` block (top level, or in an axis level or profile) sets
`device: gpu|cpu` and `cost` (default 1, a run's relative duration). Under
`--gpus`, a `device: cpu` campaign goes to `--cpu-partition`
(`EXPKIT_CPU_PARTITION`) with `--cpu-cores` cores instead. Campaigns are
submitted in order of cost × missing runs, longest first. `--gpu-type`
(`EXPKIT_GPU_TYPE`) pins the card (`--gres=gpu:pro6000:1`). `--cpus-per-gpu`
asks `--cpus-per-gpu=K` instead of `-n --cores`, as queues that pair cores
with cards expect. `EXPKIT_MPS=1` makes `launch.sh` run a private CUDA MPS
daemon inside each GPU job.

## Seed groups

`--group-seeds` (on `run`, `tune`, `submit`) sends each cell's pending seeds to
one worker when the entry's module also defines `<name>_group(ctxs)`, which
returns one `RunResult` per context (or the exception that seed failed with).
It is for small models that leave a GPU idle: train them stacked, under
`torch.func.vmap`. Each seed is stored as its own run, with
`extra['seed_group']`. A seed that fails is retried alone, and so is every
seed of a group that raises. `replay` retrains one seed alone, so a grouped run
reproduces only to float tolerance (`--tolerance 1e-5`).

## Not included

Search strategies beyond a grid (random, Bayesian or successive-halving
searches), learning-rate range tests, a trainer, and checkpointing.
`cache.ResultCache` is the piece a custom search builds on.
