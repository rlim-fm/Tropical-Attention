"""The frozen configuration: what a grid runs, and the human gate before it does.

``frozen.yaml`` holds the complete base config, the values tuned into it
(``tuned``), where it came from and an ``approved`` flag. ``expkit run`` refuses
an unapproved file. Someone has to read whatever justified the tuned values and
say yes before the grid spends its budget on them.

    expkit freeze  --config X --set optim.lr=3e-4   # write it, unapproved
    expkit approve --config X                       # after reading the evidence

Without a frozen file, ``run`` uses the YAML's ``base`` directly, unless the
experiment sets ``require_frozen: true``.

``frozen_hash`` hashes the base config alone (``config.base_hash``). Approving
a file does not change which runs it identifies; changing a value does.

**Deriving a campaign.** The other campaigns of an experiment usually share the
default campaign's tuning. ``derive`` (``expkit freeze --from-default``) writes
a campaign's file from the approved default one with the profile's ``base``
block merged over it. It inherits the approval, because the only change is the
profile, which is checked in. A profile that sets one of the *tuned* values is
refused, because that campaign needs a tuning of its own.
"""
import os

import yaml

from expkit import campaign
from expkit.config import (base_hash, flatten, load_config, plain, profile_base,
                           with_overrides, yaml_load)
from expkit.record import code_version, now


class NotApproved(SystemExit):
    pass


def write_doc(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp-{os.getpid()}'
    with open(tmp, 'w') as fh:
        yaml.safe_dump(plain(doc), fh, sort_keys=False)
    os.replace(tmp, path)
    return path


def read_doc(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return yaml_load(fh)


def freeze(cfg, *, tuned=None, evidence=None, root=None):
    """Write ``frozen.yaml``: the config's base with ``tuned`` applied, unapproved.

    ``evidence`` is free-form (a report path, a search summary) and is stored
    so the approver can find what justified the values.
    """
    base = with_overrides(cfg.base, tuned or {})
    sha, dirty = code_version()
    doc = {'experiment': cfg.experiment, 'campaign': cfg.campaign, 'profile': cfg.profile,
           'approved': False, 'frozen_hash': base_hash(base, cfg.optional),
           'tuned': dict(tuned or {}), 'base': base, 'evidence': evidence,
           'provenance': {'created': now(), 'git_sha': sha, 'git_dirty': dirty}}
    return write_doc(campaign.frozen_path(cfg, root), doc), doc


def approve(cfg, overrides=None, root=None):
    """Mark the file approved, optionally changing tuned values first."""
    path = campaign.frozen_path(cfg, root)
    doc = read_doc(path)
    if doc is None:
        raise SystemExit(f'{path} does not exist; write it first:\n'
                         f'    expkit freeze {campaign.cli_flags(cfg, root)}')
    if overrides:
        doc['base'] = with_overrides(doc['base'], overrides)
        doc['tuned'] = {**(doc.get('tuned') or {}), **overrides}
        doc['overridden_at_approval'] = sorted(overrides)
    doc['frozen_hash'] = base_hash(doc['base'], cfg.optional)
    doc.update(approved=True, approved_at=now())
    write_doc(path, doc)
    return path, doc


def resolve_base(cfg, root=None):
    """``(base, frozen_hash, source)`` for the grid: the approved frozen file, or the YAML."""
    path = campaign.frozen_path(cfg, root)
    doc = read_doc(path)
    if doc is None:
        if cfg.require_frozen:
            raise SystemExit(f'{path} does not exist, and the experiment sets '
                             f'require_frozen. Freeze and approve it first:\n'
                             f'    expkit freeze {campaign.cli_flags(cfg, root)}')
        return cfg.base, base_hash(cfg.base, cfg.optional), 'config'
    if not doc.get('approved'):
        raise NotApproved(f'{path} is not approved. Review {doc.get("evidence") or "its evidence"}, '
                          f'then run\n    expkit approve {campaign.cli_flags(cfg, root)}')
    return doc['base'], base_hash(doc['base'], cfg.optional), path


def derive(cfg, default_cfg, root=None, default_root=None, write=True):
    """A campaign's frozen file: the approved default one with the profile's base applied.

    Returns ``(path, doc, changed)``. Deriving again is a no-op while nothing
    changed. A campaign frozen on its own (no ``derived_from``) is never
    overwritten.
    """
    path = campaign.frozen_path(cfg, root)
    existing = read_doc(path)
    if existing is not None and not existing.get('derived_from'):
        raise SystemExit(f'{path} was frozen for this campaign, not derived; leaving it. '
                         f'Delete it to derive from the default campaign.')
    source = campaign.frozen_path(default_cfg, default_root)
    clean = read_doc(source)
    if clean is None or not clean.get('approved'):
        raise SystemExit(f'{source} is missing or unapproved; freeze and approve the '
                         f'default campaign first.')
    overrides = profile_base(cfg.path, cfg.profile)
    flat, current = flatten(overrides), flatten(clean['base'])
    clash = sorted(k for k in flat if k in (clean.get('tuned') or {}) and flat[k] != current.get(k))
    if clash:
        raise SystemExit(f'profile {cfg.profile!r} changes tuned values ({", ".join(clash)}), so '
                         f'the default tuning does not carry over. Freeze it on its own.')
    base = with_overrides(clean['base'], overrides)
    new_hash = base_hash(base, cfg.optional)
    if (existing and existing.get('frozen_hash') == new_hash
            and existing['derived_from'].get('frozen_hash') == clean['frozen_hash']):
        return path, existing, False
    sha, dirty = code_version()
    doc = {**clean, 'campaign': cfg.campaign, 'profile': cfg.profile, 'approved': True,
           'frozen_hash': new_hash, 'base': base, 'approved_at': now(),
           'derived_from': {'path': source, 'frozen_hash': clean['frozen_hash'],
                            'overrides': flat},
           'provenance': {'created': now(), 'git_sha': sha, 'git_dirty': dirty}}
    if write:
        write_doc(path, doc)
    return path, doc, True


def default_config(cfg):
    """The default campaign of ``cfg``'s experiment (no profile)."""
    return load_config(cfg.path)
