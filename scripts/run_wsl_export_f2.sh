#!/usr/bin/env bash
set -euo pipefail

# Run inside WSL/Linux
# Example: ./scripts/run_wsl_export_f2.sh 1.4 angstrom

R="${1:-1.4}"
UNIT="${2:-angstrom}"

python scripts/export_f2_terms_wsl.py \
  --R "${R}" \
  --unit "${UNIT}" \
  --reduction paper12 \
  --output "examples/data/chemistry/f2_R${R}_${UNIT}_sto-3g_12q.json"
