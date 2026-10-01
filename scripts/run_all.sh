#!/usr/bin/env bash
# Run every dataset at its best config, for all four horizons.
# Usage:  bash scripts/run_all.sh [train|eval]        (default: eval)
#         GPU=1 bash scripts/run_all.sh eval
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-eval}"
for ds in ETTh1 ETTh2 ETTm1 ETTm2 weather ECL Traffic solar; do
  bash "scripts/$ds.sh" "$MODE"
done
