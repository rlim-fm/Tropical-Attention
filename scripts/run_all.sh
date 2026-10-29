#!/bin/bash
# The full grid of every campaign, then the grand report over every campaign
# (campaigns/<experiment>/grand/). Nothing is tuned.
#
#   scripts/run_all.sh --plan                       # print the sbatch lines; submit nothing
#   scripts/run_all.sh                              # one Slurm job per campaign + the grand report
#   LOCAL=1 PROFILE=dry_run scripts/run_all.sh      # here
#
# Re-running submits only what is missing. Settings: scripts/_settings.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_settings.sh"

if [ "${LOCAL:-0}" = 1 ]; then
    for target in $(targets); do
        "$EXPKIT" run --config "$CONFIG" --profile "$target" $(local_devices) "$@" \
            || echo "$target: its runs did not all complete"
    done
    "$EXPKIT" grand --config "$CONFIG"
else
    "$EXPKIT" submit --config "$CONFIG" $(slurm_devices) $(campaign_flag) --grand "$@"
fi
