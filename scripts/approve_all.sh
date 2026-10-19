#!/bin/bash
# Step 2 of 3: the human gate. Shows each campaign's tuned values and their evidence, then
# approves every campaign that has an unapproved frozen.yaml, after one confirmation.
#
#   scripts/approve_all.sh                  # asks before approving
#   scripts/approve_all.sh --yes            # no question (you have read the tables)
#   PROFILE=dry_run scripts/approve_all.sh
#
# Nothing else calls this script: approving is your decision. Settings: scripts/_settings.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_settings.sh"

pending=()
for target in $(targets); do
    state=$(PYTHONPATH="$REPO_ROOT" "$PY" - "$CONFIG" "$target" <<'PY'
import os, sys
import pandas as pd
from expkit import campaign, frozen, tune
from expkit.config import load_config
cfg = load_config(sys.argv[1], sys.argv[2])
doc = frozen.read_doc(campaign.frozen_path(cfg))
if doc is None:
    print('missing'); sys.exit()
if doc.get('approved'):
    print('approved'); sys.exit()
evidence = tune.evidence_path(cfg)
print('pending')
print(f"  tuned: {doc.get('tuned')}   evidence: {os.path.relpath(evidence)}", file=sys.stderr)
if os.path.exists(evidence):
    table = pd.read_csv(evidence).drop(columns=['frozen_hash'])
    print('    ' + table.to_string(index=False).replace('\n', '\n    '), file=sys.stderr)
PY
)
    case "$state" in
        missing)  echo "$target: no frozen.yaml yet (run scripts/tune_all.sh first)" ;;
        approved) echo "$target: already approved" ;;
        pending)  echo "^ $target: unapproved"; pending+=("$target") ;;
    esac
done

if [ ${#pending[@]} -eq 0 ]; then
    echo "nothing to approve"
    exit 0
fi
if [ "${1:-}" != "--yes" ]; then
    read -r -p "Approve ${#pending[@]} campaign(s): ${pending[*]}? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] || { echo "nothing approved"; exit 1; }
fi
for target in "${pending[@]}"; do
    "$EXPKIT" approve --config "$CONFIG" --profile "$target"
done
