#!/bin/bash
# Step 2 of 3: the human gate. Shows each campaign's tuned values, the tuning report's verdict
# and warnings, and where the report is; then approves every campaign that has an unapproved
# frozen.yaml, after one confirmation.
#
#   scripts/approve_all.sh                  # asks before approving
#   scripts/approve_all.sh --yes            # no question (you have read the reports)
#   scripts/approve_all.sh --force          # also approve campaigns whose verdict is
#                                           # 'do not approve' (skipped otherwise)
#   PROFILE=dry_run scripts/approve_all.sh
#
# Nothing else calls this script: approving is your decision. Settings: scripts/_settings.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_settings.sh"

yes=0; force=0
for arg in "$@"; do
    case "$arg" in
        --yes) yes=1 ;;
        --force) force=1 ;;
        *) echo "unknown argument: $arg"; exit 2 ;;
    esac
done

pending=(); blocked=()
for target in $(targets); do
    state=$(PYTHONPATH="$REPO_ROOT" "$PY" - "$CONFIG" "$target" <<'PY'
import json, os, sys
import pandas as pd
from expkit import campaign, frozen, tune, tune_report
from expkit.config import load_config
cfg = load_config(sys.argv[1], sys.argv[2])
doc = frozen.read_doc(campaign.frozen_path(cfg))
if doc is None:
    print('missing'); sys.exit()
if doc.get('approved'):
    print('approved'); sys.exit()
err = lambda line='': print(line, file=sys.stderr)
evidence = tune.evidence_path(cfg)
verdict = doc.get('verdict') or 'no report'
err(f"  tuned: {doc.get('tuned')}   verdict: {verdict.upper()}")
err(f"  report: {doc.get('report') or os.path.relpath(tune_report.report_dir(cfg))}")
if os.path.exists(evidence):
    table = pd.read_csv(evidence).drop(columns=['frozen_hash'])
    err('    ' + table.to_string(index=False).replace('\n', '\n    '))
warnings = os.path.join(tune_report.report_dir(cfg), 'warnings.json')
if os.path.exists(warnings):
    with open(warnings) as fh:
        for w in json.load(fh)['warnings']:
            err(f"    [{w['level']}] {w['message']}")
print('blocked' if verdict == frozen.BLOCKING_VERDICT else 'pending')
PY
)
    case "$state" in
        missing)  echo "$target: no frozen.yaml yet (run scripts/tune_all.sh first)" ;;
        approved) echo "$target: already approved" ;;
        pending)  echo "^ $target: unapproved"; pending+=("$target") ;;
        blocked)
            if [ "$force" = 1 ]; then
                echo "^ $target: unapproved, report says DO NOT APPROVE (--force given)"
                pending+=("$target")
            else
                echo "^ $target: unapproved, report says DO NOT APPROVE; skipped (fix the search or pass --force)"
                blocked+=("$target")
            fi ;;
    esac
done

if [ ${#pending[@]} -eq 0 ]; then
    echo "nothing to approve"
    exit $(( ${#blocked[@]} > 0 ))
fi
if [ "$yes" != 1 ]; then
    read -r -p "Approve ${#pending[@]} campaign(s): ${pending[*]}? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { echo "nothing approved"; exit 1; }
fi
for target in "${pending[@]}"; do
    "$EXPKIT" approve --config "$CONFIG" --profile "$target" $([ "$force" = 1 ] && echo --force)
done
