#!/usr/bin/env bash
set -euo pipefail

# Run inside WSL/Linux
# Example: ./scripts/run_wsl_export_lih.sh 1.6 angstrom

R="${1:-1.6}"
UNIT="${2:-angstrom}"

python scripts/export_lih_terms_wsl.py \
  --R "${R}" \
  --unit "${UNIT}" \
  --reduction paper \
  --output "examples/data/chemistry/lih_R${R}_${UNIT}_sto-3g_6q.json"
