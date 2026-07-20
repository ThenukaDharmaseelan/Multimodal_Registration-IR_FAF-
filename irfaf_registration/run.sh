#!/usr/bin/env bash
set -euo pipefail

CSV="sample.csv"
OUTDIR="output"
FIXED_COL="moving"
MOVING_COL="fixed"
FIXED_VESSEL_COL="moving_vessel_mask"
MOVING_VESSEL_COL="fixed_vessel_mask"

if [ ! -d "irfaf_registration" ]; then
  echo "error: run this from the folder containing irfaf_registration/" >&2
  exit 1
fi

pip install -q torch kornia opencv-python numpy scipy scikit-image pandas

python -m fafir_registration.cli "$CSV" "$OUTDIR" \
  --fixed-col "$FIXED_COL" \
  --moving-col "$MOVING_COL" \
  --fixed-vessel-col "$FIXED_VESSEL_COL" \
  --moving-vessel-col "$MOVING_VESSEL_COL"

echo "Done. Results in $OUTDIR/results.csv"
