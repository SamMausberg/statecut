#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../lean"
if ! command -v lake >/dev/null 2>&1; then
  echo "UNVERIFIED: Lean/lake not installed. No successful formal build claimed." >&2
  exit 2
fi
if grep -R -nE '\b(sorry|admit|axiom|unsafe)\b' StateCut/*.lean; then
  echo "Forbidden unchecked declaration/proof token" >&2
  exit 1
fi
lake update
lake exe cache get
lake build
lake env lean StateCut.lean
