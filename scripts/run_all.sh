#!/bin/bash
# Step 3 of 3: the full grid of every approved campaign, then its report
# (campaigns/<experiment>/<campaign>/report/), then the grand report over every campaign
# (campaigns/<experiment>/grand/). Unapproved campaigns are skipped.
#
#   scripts/run_all.sh --plan                       # print the sbatch lines; submit nothing
#   scripts/run_all.sh                              # one Slurm job per campaign + a report job each
#   LOCAL=1 PROFILE=dry_run scripts/run_all.sh      # here
#
# Re-running submits only what is missing. Settings: scripts/_settings.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_settings.sh"

if [ "${LOCAL:-0}" = 1 ]; then
    for target in $(targets); do
        if "$EXPKIT" run --config "$CONFIG" --profile "$target" $(local_devices) "$@"; then
            "$EXPKIT" report --config "$CONFIG" --profile "$target" ${REPORT_PDF:+--pdf}
        else
            echo "skip report for $target: its runs did not all complete (or it is not approved)"
        fi
    done
    "$EXPKIT" grand --config "$CONFIG"
else
    "$EXPKIT" submit --config "$CONFIG" $(slurm_devices) $(campaign_flag) --report --grand "$@"
fi
