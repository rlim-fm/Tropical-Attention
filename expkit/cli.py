"""``python -m expkit <command>``: every stage of an experiment, from one YAML.

    expkit run      --config X [--profile P] [--cpu-workers N | --gpus 0,1 --per-gpu 2]
                    [--group-seeds] [--only lr=0.01,seed=0] [--keys K,...] [--force] [--plan]
    expkit status   --config X                       # every campaign: frozen? stored? failed?
    expkit freeze   --config X [--set k=v ...] [--evidence PATH] | --from-default
    expkit approve  --config X [--set k=v ...]
    expkit grand    --config X [--no-pdf]             # the report over every campaign
    expkit replay   --config X --keys K | --only ... [--tolerance 0]   # bit-identical check
    expkit clear    --config X [--only ...] [--yes]
    expkit submit   --config X [--campaigns a/b,c/d] --cores 32
                    [--gpus 1 --per-gpu 4 --gpu-type pro6000 --cpus-per-gpu 40]
                    [--lanes 'mixed:pro6000:2:40:120:12 main:cpu:4:32::12:0.3']
                    [--cpu-partition main --cpu-cores 32] [--group-seeds]
                    [--grand] [--plan]                                       # Slurm

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
    return 1 if _launch(args, todo, store.root)['failed'] else 0


def _launch(args, todo, store_root):
    gpus = [int(g) for g in args.gpus.split(',')] if args.gpus else None
    slots = device_slots(gpus, args.per_gpu, args.cpu_workers)
    threads = export_worker_env(slots)
    print(f'-- {len(todo)} runs on {sorted(set(slots))} ({len(slots)} concurrent'
          f'{" seed groups" if args.group_seeds else ""}, {threads} thread(s) each)')
    out = execute(todo, store_root, slots, retries=args.retries, threads=threads,
                  group_seeds=args.group_seeds)
    print(f'\ndone: {len(out["ok"])} ok, {len(out["failed"])} failed'
          + (f' ({", ".join(out["failed"])}; tracebacks in registry.jsonl)' if out['failed'] else ''))
    return out


def _status_line(cfg, root=None):
    doc = frozen.read_doc(campaign.frozen_path(cfg, root))
    state = ('no frozen file' if doc is None else 'approved' if doc.get('approved')
             else 'UNAPPROVED')
    try:
        specs, _ = build_specs(cfg, root)
        store = Store(campaign.store_dir(cfg, root))
        missing = len(store.pending(specs))
        line = (f'{cfg.campaign:28s} {state:15s} {len(specs) - missing:6d}/{len(specs):<6d} '
                f'stored  {len(store.failures()):4d} failed')
    except SystemExit:
        line = f'{cfg.campaign:28s} {state:15s} {"":>13s} stored  {"":>4s} failed'
    return line


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


def cmd_grand(args, cfg):
    from expkit import grand
    grand.build(cfg, pdf=not args.no_pdf, n_boot=args.n_boot)
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
    return clear(store, [parse_selector(o) for o in (args.only or [])], yes=args.yes)


def slurm_settings(cfg):
    """A campaign's ``slurm:`` block: ``device`` ('gpu', 'cpu' or 'any') and ``cost``.

    ``device: cpu`` sends the campaign to a CPU lane even when ``submit`` is
    given GPUs (small models can run as well on cores as on a shared card), and
    ``device: any`` to whichever lane, GPU or CPU, would finish it soonest.
    ``cost`` (default 1) is the relative time one of its runs takes; campaigns
    are submitted in order of cost x missing runs, longest first, so the long
    poles start before the short ones fill the queue.
    """
    block = cfg.extra.get('slurm') or {}
    device = str(block.get('device', 'gpu')).lower()
    if device not in ('gpu', 'cpu', 'any'):
        raise SystemExit(f'{cfg.campaign}: slurm.device must be gpu, cpu or any, not {device!r}')
    return device, float(block.get('cost', 1))


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
    ids = {}                  # campaign -> its job ids, which the grand report waits on
    cpu_partition = args.cpu_partition or os.environ.get('EXPKIT_CPU_PARTITION')
    todo = []                 # (cost, order, name, config, job name, device)
    for order, name in enumerate(names):
        c = load_config(cfg.path, None if name == 'default' else name)
        job = slurm.job_name(c.experiment, c.campaign)
        try:
            specs, _ = build_specs(c)
            missing = len(Store(campaign.store_dir(c)).pending(specs))
        except SystemExit as exc:
            print(f'skip {c.campaign}: {exc}')
            continue
        if not missing:
            print(f'skip {c.campaign}: every run is stored')
            continue
        running = slurm.queued(job)
        if running:
            print(f'skip {c.campaign}: {job} is already queued ({", ".join(running)})')
            ids[name] = running
            continue
        device, cost = slurm_settings(c)
        todo.append((-cost * missing, order, name, job, device))
    lanes = slurm.parse_lanes(args.lanes or os.environ.get('EXPKIT_LANES'))
    if all(slurm.is_cpu(lane) for lane in lanes):
        lanes.append(slurm.Lane(args.partition, args.gpu_type, 1, args.cpus_per_gpu, None,
                                args.per_gpu, 1.0))
    on = {'gpu': [i for i, lane in enumerate(lanes) if not slurm.is_cpu(lane)],
          'cpu': [i for i, lane in enumerate(lanes) if slurm.is_cpu(lane)]}
    on['any'] = on['gpu'] + on['cpu']
    if not on['cpu']:
        # Only `device: cpu` campaigns use it; `any` stays on the GPU lanes.
        cores = args.cpu_cores or args.cores
        lanes.append(slurm.Lane(cpu_partition, 'cpu', 1, cores, None, cores, 1.0))
        on['cpu'] = [len(lanes) - 1]
    filled = [[0.0] * lane.slots for lane in lanes]
    for load, _, name, job, device in sorted(todo):
        lane = slurm.assign_lane(lanes, filled, -load, on[device]) if args.gpus else None
        if lane is None:                   # no GPUs at all: every campaign on args.cores
            cores = args.cores
            devices = ['--cpu-workers', str(cores)]
            options = dict(partition=args.partition)
        elif slurm.is_cpu(lane):
            cores = lane.cpus_per_gpu
            devices = ['--cpu-workers', str(lane.per_gpu)]
            options = dict(partition=lane.partition, mem_gb=lane.mem_gb)
        else:
            cores = args.cores
            devices = ['--gpus', ','.join(str(i) for i in range(args.gpus)),
                       '--per-gpu', str(lane.per_gpu)]
            options = dict(partition=lane.partition, gpus=args.gpus, gpu_type=lane.gpu_type,
                           cpus_per_gpu=lane.cpus_per_gpu, mem_gb=lane.mem_gb)
        run_args = ['run', '--config', config_arg, *devices,
                    *(['--group-seeds'] if args.group_seeds else []),
                    *(['--profile', name] if name != 'default' else [])]
        jid = slurm.submit(job, cores, run_args, plan=args.plan, **options)
        if jid:
            ids[name] = [jid]
    waits = [str(i) for jobs in ids.values() for i in jobs]
    if args.grand:
        # after every job above, or at once
        slurm.submit(slurm.job_name(cfg.experiment, 'grand'), 1,
                     ['grand', '--config', config_arg],
                     dependency=('afterany:' + ':'.join(waits)) if waits else 'singleton',
                     partition=cpu_partition if args.gpus else args.partition, plan=args.plan)
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
    s.add_argument('--group-seeds', action='store_true',
                   help="train each cell's seeds together (the entry's <name>_group)")
    s.add_argument('--only', action='append', help="subset, e.g. 'lr=0.01,seed=0' (repeatable)")
    s.add_argument('--keys', default=None)
    s.add_argument('--force', action='store_true', help='re-run stored runs')
    s.add_argument('--plan', action='store_true', help='list what would run; train nothing')

    add('status', cmd_status, 'every campaign: frozen state, runs stored, failures')

    s = add('freeze', cmd_freeze, 'write frozen.yaml (unapproved), or derive it')
    s.add_argument('--set', action='append', metavar='KEY=VALUE', help='a value to pin into the base')
    s.add_argument('--evidence', default=None, help='what justifies the values (a path, a note)')
    s.add_argument('--from-default', action='store_true',
                   help="derive from the default campaign's approved file")

    s = add('approve', cmd_approve, 'approve frozen.yaml')
    s.add_argument('--set', action='append', metavar='KEY=VALUE')

    s = add('grand', cmd_grand, 'the grand report over every campaign (campaigns/<exp>/grand/)')
    s.add_argument('--no-pdf', action='store_true', help='write the .tex only')
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
    s.add_argument('--gpu-type', default=None, help="--gres card, e.g. 'pro6000'")
    s.add_argument('--cpus-per-gpu', type=int, default=None,
                   help='ask --cpus-per-gpu=K instead of -n --cores (GPU jobs)')
    s.add_argument('--partition', default=None)
    s.add_argument('--lanes', default=None,
                   help="job shapes to spread campaigns over, e.g. "
                        "'mixed:pro6000:2:40:120:12 main:cpu:4:32::12:0.3' (expkit.slurm.parse_lanes); "
                        'GPU lanes replace --partition, --gpu-type, --cpus-per-gpu and --per-gpu, '
                        'a CPU lane --cpu-partition and --cpu-cores')
    s.add_argument('--cpu-partition', default=None,
                   help='queue for campaigns with slurm.device: cpu, and the grand-report job')
    s.add_argument('--cpu-cores', type=int, default=None,
                   help='cores for a slurm.device: cpu campaign (default: --cores)')
    s.add_argument('--group-seeds', action='store_true',
                   help="train each cell's seeds together (the entry's <name>_group)")
    s.add_argument('--grand', action='store_true',
                   help='then one grand-report job over every campaign, after all the others')
    s.add_argument('--plan', action='store_true', help='print the sbatch lines; submit nothing')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = load_config(args.config, args.profile)
    return args.fn(args, cfg) or 0


if __name__ == '__main__':
    sys.exit(main())
