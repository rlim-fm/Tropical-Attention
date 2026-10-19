"""The contract between expkit and your experiment code.

An experiment's ``entry`` (``module:function``) is called once per run, with
a ``RunContext``, and returns a ``RunResult`` (or a dict with its fields)::

    from expkit.api import RunContext, RunResult

    def run(ctx: RunContext) -> RunResult:
        torch.manual_seed(ctx.seeds['init'])
        model = build(ctx.base, ctx.cell)
        ...
        return RunResult(metrics={'mse': mse}, curves={'train_loss': losses})

Rules the entry has to keep, or replay and the paired statistics break:

* **Randomness comes from ``ctx.seeds`` only.** Those seeds depend on the seed
  index and not on the cell (``expkit.seeding``). Never seed from a cell's
  levels, the wall clock or the process id.
* **A run is a function of (base, cell, seeds).** Then ``expkit replay`` can
  retrain a stored run and check it bit for bit. No checkpoint is ever written.
* **Hooks observe only.** ``ctx.hooks`` holds whatever callbacks the caller
  passed (``expkit replay`` passes none by default). Calling them must not
  change what the run computes.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class RunContext:
    base: dict                  # the frozen base config (plain dict)
    cell: dict                  # this run's factor levels, {name: level}
    seeds: dict                 # {stream: int}, from (global_seed, seed_index)
    seed_index: int
    device: str = 'cpu'
    experiment: str = ''
    hooks: Dict[str, Callable] = field(default_factory=dict)
    artifact_dir: Optional[str] = None   # a per-run directory for files, created on first use

    def hook(self, name, *args, **kwargs):
        """Call ``hooks[name]`` if the caller gave one; a no-op otherwise."""
        fn = self.hooks.get(name)
        return fn(*args, **kwargs) if fn else None


@dataclass
class RunResult:
    metrics: Dict[str, float] = field(default_factory=dict)
    curves: Dict[str, Any] = field(default_factory=dict)   # name -> 1-d array or list
    diverged: bool = False
    steps: Optional[int] = None
    extra: dict = field(default_factory=dict)              # anything JSON-safe beside them

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f'an entry must return a RunResult or a dict, got {type(value).__name__}')
