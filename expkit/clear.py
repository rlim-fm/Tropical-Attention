"""Clear stored runs so the next ``expkit run`` trains them again.

A run is complete when its key has a line in ``runs.jsonl``. A code fix that
changes what a run computes does not change its key, so after such a fix the
old runs have to be removed before the grid will retrain them. Nothing else in
expkit removes runs.

Clearing everything removes ``runs.jsonl``, ``curves.jsonl``, ``registry.jsonl``
and ``artifacts/``. It keeps ``configs.jsonl``
(keyed by frozen hash, still valid) and ``frozen.yaml``. ``--only`` selectors
(the ``run --only`` syntax, repeatable; a run matching any one goes) rewrite
the three files without the matching runs' lines.

Without ``--yes`` nothing changes. With ``--yes``, a store whose git-tracked
files have uncommitted changes is refused, because the clear must be
recoverable from the commit it prints. ``curves.jsonl`` is often untracked, so
git may not bring it back. It is set aside as ``curves.jsonl.cleared-<sha>-<time>``
instead; move it back to undo.
"""
import datetime
import os
import shutil
import subprocess

from expkit.record import REPO, selector_matches
from expkit.store import Store, read_jsonl, rewrite_jsonl

JSONL = ('runs.jsonl', 'curves.jsonl', 'registry.jsonl')
SET_ASIDE = ('curves.jsonl',)


def git_dirty(paths, repo=REPO):
    """The paths among ``paths`` that git tracks and that have uncommitted changes."""
    inside = [p for p in paths if os.path.exists(p)
              and os.path.abspath(p).startswith(os.path.abspath(repo) + os.sep)]
    if not inside:
        return []
    try:
        out = subprocess.run(['git', '-C', repo, 'status', '--porcelain', '--', *inside],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line[3:] for line in out.splitlines() if not line.startswith('??')]


def head(repo=REPO):
    out = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                         capture_output=True, text=True)
    return out.stdout.strip() or 'nogit'


def plan(store, selectors):
    """``(files to remove, dirs to remove, {file: lines to keep}, runs cleared)``."""
    runs = store.runs()
    files = [f for f in (os.path.join(store.root, n) for n in JSONL) if os.path.exists(f)]
    if not selectors:
        dirs = [store.artifacts_dir] if os.path.isdir(store.artifacts_dir) else []
        return files, dirs, {}, len(runs)
    gone = {k for k, line in runs.items()
            if any(selector_matches(s, line['cell'], line['seed_index']) for s in selectors)}
    kept = {f: [line for line in read_jsonl(f) if line['key'] not in gone] for f in files}
    dirs = [store.artifact_dir(k) for k in gone if os.path.isdir(store.artifact_dir(k))]
    return [], dirs, kept, len(gone)


def clear(store_root, selectors=(), yes=False, log=print, is_dirty=git_dirty):
    store = Store(store_root)
    files, dirs, kept, n = plan(store, list(selectors))
    log(f'{n} run(s) {"matching" if selectors else "(every file)"} in {store.root}')
    if not yes:
        log('nothing changed; pass --yes to clear')
        return 0
    tracked = [os.path.join(store.root, f) for f in JSONL if f not in SET_ASIDE]
    changed = is_dirty(tracked)
    if changed:
        log('refusing: these have uncommitted changes, so a clear could not be undone:')
        for path in changed:
            log(f'  {path}')
        return 1
    sha = head()
    suffix = f'.cleared-{sha}-{datetime.datetime.now():%Y%m%d%H%M%S}'
    for path in [*kept, *files]:
        if os.path.basename(path) in SET_ASIDE and os.path.exists(path):
            shutil.copyfile(path, path + suffix)
            log(f'set aside {path + suffix}')
    for path, lines in kept.items():
        rewrite_jsonl(path, lines)
    for path in files:
        if os.path.exists(path):
            os.remove(path)
    for path in dirs:
        shutil.rmtree(path, ignore_errors=True)
    log(f'cleared. To undo: git checkout {sha} -- <store files>, and move curves.jsonl{suffix} back')
    return 0
