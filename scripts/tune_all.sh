#!/bin/bash
# Step 1 of 3: the search in each campaign's `tune:` block (Fishr's lambda), selected on
# in-distribution validation. Each campaign ends with frozen.yaml, UNAPPROVED, and its
# evidence in campaigns/<experiment>/<campaign>/tune/tune.csv.
#
#   scripts/tune_all.sh --plan                      # print the sbatch lines; submit nothing
#   scripts/tune_all.sh                             # one Slurm job per campaign
#   LOCAL=1 PROFILE=dry_run scripts/tune_all.sh     # here, the smoke-test profile
#
# Next: read the tune.csv tables, then scripts/approve_all.sh. Settings: scripts/_settings.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_settings.sh"

if [ "${LOCAL:-0}" = 1 ]; then
    for target in $(targets); do
        "$EXPKIT" tune --config "$CONFIG" --profile "$target" $(local_devices) "$@"
    done
else
    # A campaign whose tuning runs are all stored is frozen on the login node, without a job.
    "$EXPKIT" submit --stage tune --config "$CONFIG" $(slurm_devices) $(campaign_flag) "$@"
fi
