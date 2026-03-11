#!/usr/bin/env bash
set -euo pipefail

# Run inside WSL/Linux
# Example: ./scripts/run_wsl_export_h2_2q.sh 2.6 angstrom

R="${1:-2.6}"
UNIT="${2:-angstrom}"

python scripts/export_h2_terms_wsl.py \
  --R "${R}" \
  --unit "${UNIT}" \
  --encoding scbk2 \
  --output "examples/data/chemistry/h2_R${R}_${UNIT}_sto-3g_scbk2.json"
