"""``python -m expkit <command>``: every stage of an experiment, from one YAML.

    expkit run      --config X [--profile P] [--cpu-workers N | --gpus 0,1 --per-gpu 2]
                    [--only lr=0.01,seed=0] [--keys K,...] [--force] [--plan]
    expkit status   --config X                       # every campaign: frozen? stored? failed?
    expkit freeze   --config X [--set k=v ...] [--evidence PATH] | --from-default
    expkit approve  --config X [--set k=v ...]
    expkit report   --config X [--pdf]
    expkit replay   --config X --keys K | --only ... [--tolerance 0]   # bit-identical check
    expkit clear    --config X [--only ...] [--yes]
    expkit submit   --config X [--campaigns a/b,c/d] --cores 32 [--report] [--plan]   # Slurm

Every command takes ``--config``, ``--profile`` and ``--root``, and resolves the
same campaign directory from them (``expkit.campaign``). Pass the same
``--profile`` to every stage.
"""
import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

from expkit import campaign, frozen, slurm
from expkit.config import cell_label, load_config, parse_assignments
from expkit.executor import device_slots, execute, export_worker_env, replay
from expkit.record import RunSpec, parse_selector
from expkit.store import Store
from expkit.threads import configure_threads, export_device_env


def build_specs(cfg, root=None):
    """Every cell x seed of the grid, under the base ``frozen.resolve_base`` picks."""
    base, frozen_hash, source = frozen.resolve_base(cfg, root)
    streams = cfg.extra.get('seed_streams')
    specs = [RunSpec(experiment=cfg.experiment, entry=cfg.entry, frozen_hash=frozen_hash,
                     base=base, cell=cell, seed_index=s, global_seed=cfg.global_seed,
                     seed_streams=list(streams) if streams else None)
             for cell in cfg.cells() for s in cfg.seed_indices()]
    return specs, source


def select(specs, only=None, keys=None):
    selectors = [parse_selector(o) for o in (only or [])]
    if selectors:
        specs = [s for s in specs if any(s.matches(sel) for sel in selectors)]
    if keys:
        specs = [s for s in specs if s.key in keys]
    return specs


def _keys(text):
    return set(filter(None, (text or '').split(',')))


def cmd_run(args, cfg):
    specs, source = build_specs(cfg, args.root)
    specs = select(specs, args.only, _keys(args.keys))
    store = Store(args.store or campaign.store_dir(cfg, args.root))
    todo = store.pending(specs, force=args.force or bool(args.keys))
    print(f'== {cfg.experiment} [{cfg.campaign}]: {len(specs)} runs selected, '
          f'{len(specs) - len(todo)} stored, {len(todo)} to run  (base: {source}; {store.root})')
    if args.plan:
        for label, n in sorted(Counter(cell_label(s.cell) for s in todo).items()):
            print(f'   {label}: {n} seed(s)')
        return 0
    if not todo:
        return 0
    gpus = [int(g) for g in args.gpus.split(',')] if args.gpus else None
    slots = device_slots(gpus, args.per_gpu, args.cpu_workers)
    threads = export_worker_env(slots)
    print(f'-- {len(todo)} runs on {sorted(set(slots))} ({len(slots)} concurrent, '
          f'{threads} thread(s) each)')
    out = execute(todo, store.root, slots, retries=args.retries, threads=threads)
    print(f'\ndone: {len(out["ok"])} ok, {len(out["failed"])} failed'
          + (f' ({", ".join(out["failed"])}; tracebacks in registry.jsonl)' if out['failed'] else ''))
    return 1 if out['failed'] else 0


def _status_line(cfg, root=None):
    doc = frozen.read_doc(campaign.frozen_path(cfg, root))
    state = ('no frozen file' if doc is None else 'approved' if doc.get('approved')
             else 'UNAPPROVED')
    try:
        specs, _ = build_specs(cfg, root)
    except SystemExit:
        return f'{cfg.campaign:28s} {state}'
    store = Store(campaign.store_dir(cfg, root))
    missing = len(store.pending(specs))
    return (f'{cfg.campaign:28s} {state:15s} {len(specs) - missing:6d}/{len(specs):<6d} stored  '
            f'{len(store.failures()):4d} failed')


def cmd_status(args, cfg):
    from expkit.config import campaigns
    names = [cfg.profile] if args.profile else campaigns(cfg.path)
    for name in names:
        c = load_config(cfg.path, None if name == 'default' else name)
        print(_status_line(c, args.root if args.profile else None))
    return 0


def cmd_freeze(args, cfg):
    if args.from_default:
        default = frozen.default_config(cfg)
        path, _, changed = frozen.derive(cfg, default, root=args.root)
        print(f'{"wrote" if changed else "unchanged"} {path}')
        return 0
    path, doc = frozen.freeze(cfg, tuned=parse_assignments(args.set), evidence=args.evidence,
                              root=args.root)
    print(f'wrote {path} (frozen hash {doc["frozen_hash"]}, unapproved). Review, then:\n'
          f'    expkit approve {campaign.cli_flags(cfg, args.root)}')
    return 0


def cmd_approve(args, cfg):
    path, doc = frozen.approve(cfg, parse_assignments(args.set), root=args.root)
    print(f'approved {path} (frozen hash {doc["frozen_hash"]})')
    return 0


def cmd_report(args, cfg):
    from expkit import report
    base, frozen_hash, _ = frozen.resolve_base(cfg, args.root)
    store = Store(args.store or campaign.store_dir(cfg, args.root))
    report.build(cfg, store, frozen_hash, base, args.out_dir or campaign.report_dir(cfg, args.root),
                 pdf=args.pdf, n_boot=args.n_boot)
    return 0


def curve_deviation(stored, replayed, names):
    """Largest |stored - replayed| over the named curves; NaN patterns must match."""
    worst = 0.0
    for name in names:
        if name not in stored or name not in replayed:
            return math.inf
        a, b = np.asarray(stored[name], float), np.asarray(replayed[name], float)
        if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
            return math.inf
        ok = ~np.isnan(a)
        if ok.any():
            worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]))))
    return worst


def cmd_replay(args, cfg):
    if not args.keys and not args.only:
        raise SystemExit('name the runs to replay with --keys or --only')
    export_device_env(args.device)
    configure_threads()
    store = Store(args.store or campaign.store_dir(cfg, args.root))
    _, frozen_hash, _ = frozen.resolve_base(cfg, args.root)
    selectors = [parse_selector(o) for o in (args.only or [])]
    keys = _keys(args.keys)
    chosen = [r for r in store.load_records(frozen_hash)
              if (not keys or r.key in keys)
              and (not selectors or any(r.spec().matches(s) for s in selectors))]
    if not chosen:
        raise SystemExit('no stored run matches')
    off = []
    for record in chosen:
        stored = store.load_curves([record.key]).get(record.key, {})
        result = replay(record, args.device)
        names = cfg.replay_curves or sorted(stored)
        dev = curve_deviation(stored, {k: np.asarray(v) for k, v in result.curves.items()}, names)
        print(f'  {record.key} {cell_label(record.cell)} seed {record.seed_index}: '
              f'max deviation {dev:.3g} over {", ".join(names)}')
        if not dev <= args.tolerance:
            off.append((record.key, dev))
    if off:
        raise SystemExit('replay does not reproduce the stored curves: '
                         + ', '.join(f'{k} ({d:.2e})' for k, d in off))
    return 0


def cmd_clear(args, cfg):
    from expkit.clear import clear
    store = args.store or campaign.store_dir(cfg, args.root)
    return clear(store, campaign.report_dir(cfg, args.root),
                 [parse_selector(o) for o in (args.only or [])], yes=args.yes)


def cmd_submit(args, cfg):
    from expkit.config import campaigns
    names = campaigns(cfg.path)
    if args.campaigns:
        wanted = [c for c in args.campaigns.split(',') if c]
        unknown = [c for c in wanted if c not in names]
        if unknown:
            raise SystemExit(f'unknown campaign(s) {unknown}; there are {names}')
        names = wanted
    config_arg = os.path.relpath(cfg.path, slurm.REPO)
    ids = []
    for name in names:
        c = load_config(cfg.path, None if name == 'default' else name)
        job = slurm.job_name(c.experiment, c.campaign)
        try:
            specs, _ = build_specs(c)
        except SystemExit as exc:
            print(f'skip {c.campaign}: {exc}')
            continue
        missing = len(Store(campaign.store_dir(c)).pending(specs))
        if not missing:
            print(f'skip {c.campaign}: every run is stored')
            continue
        running = slurm.queued(job)
        if running:
            print(f'skip {c.campaign}: {job} is already queued ({", ".join(running)})')
            ids.extend(running)
            continue
        devices = (['--gpus', ','.join(str(i) for i in range(args.gpus)),
                    '--per-gpu', str(args.per_gpu)] if args.gpus
                   else ['--cpu-workers', str(args.cores)])
        run_args = ['run', '--config', config_arg, *devices,
                    *(['--profile', name] if name != 'default' else [])]
        jid = slurm.submit(job, args.cores, run_args, partition=args.partition,
                           gpus=args.gpus, plan=args.plan)
        if jid:
            ids.append(jid)
    if args.report and ids:
        for name in names:
            report_args = ['report', '--config', config_arg, '--pdf']
            if name != 'default':
                report_args += ['--profile', name]
            slurm.submit(slurm.job_name(cfg.experiment, f'{name}-report'), 1, report_args,
                         dependency='afterany:' + ':'.join(str(i) for i in ids),
                         partition=args.partition, plan=args.plan)
    return 0


def parser():
    p = argparse.ArgumentParser(prog='expkit', description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        campaign.add_arguments(s)
        s.set_defaults(fn=fn)
        return s

    s = add('run', cmd_run, 'train every missing run of the grid')
    s.add_argument('--store', default=None)
    s.add_argument('--cpu-workers', type=int, default=1)
    s.add_argument('--gpus', default=None, help='comma list of GPU indices')
    s.add_argument('--per-gpu', type=int, default=1)
    s.add_argument('--retries', type=int, default=2)
    s.add_argument('--only', action='append', help="subset, e.g. 'lr=0.01,seed=0' (repeatable)")
    s.add_argument('--keys', default=None)
    s.add_argument('--force', action='store_true', help='re-run stored runs')
    s.add_argument('--plan', action='store_true', help='list what would run; train nothing')

    add('status', cmd_status, 'every campaign: frozen state, runs stored, failures')

    s = add('freeze', cmd_freeze, 'write frozen.yaml (unapproved), or derive it')
    s.add_argument('--set', action='append', metavar='KEY=VALUE', help='a tuned value')
    s.add_argument('--evidence', default=None, help='what justifies the values (a path, a note)')
    s.add_argument('--from-default', action='store_true',
                   help="derive from the default campaign's approved file")

    s = add('approve', cmd_approve, 'approve frozen.yaml')
    s.add_argument('--set', action='append', metavar='KEY=VALUE')

    s = add('report', cmd_report, 'statistics, figures and report.tex')
    s.add_argument('--store', default=None)
    s.add_argument('--out-dir', default=None)
    s.add_argument('--pdf', action='store_true')
    s.add_argument('--n-boot', type=int, default=2000)

    s = add('replay', cmd_replay, 'retrain stored runs and check their curves')
    s.add_argument('--store', default=None)
    s.add_argument('--keys', default=None)
    s.add_argument('--only', action='append')
    s.add_argument('--device', default='cpu')
    s.add_argument('--tolerance', type=float, default=0.0)

    s = add('clear', cmd_clear, 'remove stored runs so they retrain')
    s.add_argument('--store', default=None)
    s.add_argument('--only', action='append')
    s.add_argument('--yes', action='store_true')

    s = add('submit', cmd_submit, 'one Slurm job per campaign with runs missing')
    s.add_argument('--campaigns', default=None, help='comma list (default: all)')
    s.add_argument('--cores', type=int, default=8)
    s.add_argument('--gpus', type=int, default=None, help='GPUs per job')
    s.add_argument('--per-gpu', type=int, default=1)
    s.add_argument('--partition', default=None)
    s.add_argument('--report', action='store_true', help='then a report job per campaign')
    s.add_argument('--plan', action='store_true', help='print the sbatch lines; submit nothing')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = load_config(args.config, args.profile)
    return args.fn(args, cfg) or 0


if __name__ == '__main__':
    sys.exit(main())
