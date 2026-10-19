"""expkit: infrastructure for factorial empirical-ML experiments.

A grid of cells x paired seeds, run in parallel worker processes into an
append-only JSONL store, keyed so that re-running is a no-op and a changed
config is a new campaign; an approval gate before a grid spends its budget;
bit-identical replay instead of checkpoints; seed-paired statistics and a
LaTeX report; Slurm submission. See README.md.
"""
__version__ = '0.1.0'
