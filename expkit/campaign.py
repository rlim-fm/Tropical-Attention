"""Where a campaign's files live, derived from the experiment and the profile alone.

Every command resolves the same root from ``(config, root)``, so the stages
cannot disagree about where the others wrote::

    <root>/frozen.yaml     the approved base config the grid runs (expkit freeze/approve)
    <root>/results/        the run store (expkit.store)
    <root>/report/         the report (expkit.report)

The default root is ``<campaigns dir>/<experiment>/<campaign>/``. The campaigns
dir is ``$EXPKIT_CAMPAIGNS``, or else ``campaigns/`` in the repository expkit
sits in. ``--root`` overrides it. That is how a dry run writes into a scratch
directory.
"""
import os

from expkit.record import REPO


def campaigns_dir():
    return os.path.abspath(os.environ.get('EXPKIT_CAMPAIGNS') or os.path.join(REPO, 'campaigns'))


def campaign_root(cfg, root=None):
    if root:
        return os.path.abspath(root)
    return os.path.join(campaigns_dir(), cfg.experiment, *cfg.campaign.split('/'))


def frozen_path(cfg, root=None):
    return os.path.join(campaign_root(cfg, root), 'frozen.yaml')


def store_dir(cfg, root=None):
    return os.path.join(campaign_root(cfg, root), 'results')


def report_dir(cfg, root=None):
    return os.path.join(campaign_root(cfg, root), 'report')


def add_arguments(parser):
    """The flags every command resolves its paths from."""
    parser.add_argument('--config', required=True, help='the experiment YAML')
    parser.add_argument('--profile', default=None,
                        help="a campaign ('noisy/wd') or a profiles: key ('dry_run')")
    parser.add_argument('--root', default=None,
                        help='campaign root directory, overriding the one the profile implies')


def cli_flags(cfg, root=None):
    """The command-line flags that name this campaign again, for hints in messages."""
    flags = f'--config {os.path.relpath(cfg.path)}'
    if cfg.profile:
        flags += f' --profile {cfg.profile}'
    if root:
        flags += f' --root {root}'
    return flags
