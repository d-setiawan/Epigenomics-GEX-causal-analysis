#!/usr/bin/env bash
# Rebuild per-mark matrices used by the integration workspace.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python reproduce/restore_inputs.py pbmc --check
data=pipelines/empirical/data
for marker in "$data"/raw/H3K*/*_ADT_counts.tsv; do
  mark="$(basename "$(dirname "$marker")")"
  prefix="$(basename "$marker" _ADT_counts.tsv)"
  input="$data/raw/$mark/$prefix"
  out="$data/outputs/$mark"
  if [[ -e "$out" ]]; then
    echo "Output already exists: $out. Run in a fresh checkout." >&2
    exit 1
  fi
  bash "$data/scripts/run_cuttag_chunks_full.sh" "$prefix" "$out" \
    "$marker" "${input}_HTO_counts.tsv" "${input}_histone_fragments.tsv" \
    "$data/raw/chromsizes/hg38.chrom.sizes" hs 10 10 100 0.01 500
done
python pipelines/empirical/integration/scripts/setup_integration_workspace.py
