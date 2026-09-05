#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root/lean"
if ! command -v lake >/dev/null 2>&1; then
  echo "UNVERIFIED: Lean/lake not installed. No successful formal build claimed." >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "UNVERIFIED: Python 3 is required to record the formal audit." >&2
  exit 2
fi
audit_dir="${STATECUT_FORMAL_LOG_DIR:-$repo_root/results/gh200/formal}"
mkdir -p "$audit_dir"
audit_dir="$(cd "$audit_dir" && pwd)"
# A failed rerun must not leave a stale machine-readable successful status.
rm -f "$audit_dir/summary.json"
python3 - <<'PY'
from pathlib import Path
import re

for path in [Path("StateCut.lean"), Path("Audit.lean"), *sorted(Path("StateCut").rglob("*.lean"))]:
    source = path.read_text()
    if re.search(r"\b(?:sorry|admit|unsafe)\b|^\s*(?:private\s+|protected\s+)?axiom\s", source, re.M):
        raise SystemExit(f"Forbidden unchecked declaration/proof token: {path}")
if not Path("lake-manifest.json").is_file():
    raise SystemExit("Missing pinned lake-manifest.json; restore the repository lockfile.")
PY
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  uname -srm
  lean --version
  lake --version
} | tee "$audit_dir/environment.log"
# Lake uses the committed manifest; ordinary verification does not update pins.
lake exe cache get 2>&1 | tee "$audit_dir/cache.log"
lake build 2>&1 | tee "$audit_dir/build.log"
lake env lean -DwarningAsError=true Audit.lean 2>&1 | tee "$audit_dir/dependencies.log"
python3 - "$repo_root" "$audit_dir" <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

root, audit_dir = map(Path, sys.argv[1:])
records = []
runtime_auxiliaries = []
for line in (audit_dir / "dependencies.log").read_text().splitlines():
    if "STATECUT_AUDIT " in line:
        records.append(json.loads(line.split("STATECUT_AUDIT ", 1)[1]))
    if "STATECUT_RUNTIME_AUX " in line:
        runtime_auxiliaries.append(json.loads(line.split("STATECUT_RUNTIME_AUX ", 1)[1]))
if not records:
    raise SystemExit("No compiled declaration audit was recorded.")
by_name = {record["name"]: record for record in records}
if len(by_name) != len(records):
    raise SystemExit("Duplicate declarations in audit.")
source_theorems = {}
for path in sorted((root / "lean/StateCut").rglob("*.lean")):
    for match in re.finditer(r"^theorem\s+(\w+)", path.read_text(), re.M):
        name = "StateCut." + match.group(1)
        source_theorems[name] = str(path.relative_to(root))
missing = set(source_theorems) - by_name.keys()
if missing:
    raise SystemExit(f"Unbuilt source theorems: {sorted(missing)}")
allowed = {"propext", "Classical.choice", "Quot.sound"}
for record in records:
    if set(record["dependencies"]) - allowed or record["kind"] == "axiom":
        raise SystemExit(f"Unexpected proof dependency: {record}")
paths = sorted((root / "lean/StateCut").rglob("*.lean")) + [
    root / "lean/StateCut.lean", root / "lean/Audit.lean",
    root / "lean/lean-toolchain", root / "lean/lakefile.toml",
    root / "lean/lake-manifest.json", root / "scripts/check_lean.sh"]
manifest = json.loads((root / "lean/lake-manifest.json").read_text())
dependencies = {}
for package in manifest["packages"]:
    path = root / "lean/.lake/packages" / package["name"]
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != package["rev"]:
        raise SystemExit(f"Dependency revision differs from lockfile: {package['name']}")
    if subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"]).returncode:
        raise SystemExit(f"Dependency tracked sources are modified: {package['name']}")
    dependencies[package["name"]] = actual
summary = {
    "status": "compiled_and_audited",
    "build_passed": True,
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "toolchain": (root / "lean/lean-toolchain").read_text().strip(),
    "lean_version": subprocess.check_output(["lean", "--version"], text=True).strip(),
    "mathlib_revision": dependencies["mathlib"],
    "environment": (audit_dir / "environment.log").read_text().splitlines(),
    "source_theorem_count": len(source_theorems),
    "audited_constant_count": len(records),
    "audited_theorem_count": sum(record["kind"] == "theorem" for record in records),
    "compiler_runtime_auxiliaries_outside_proof_audit": runtime_auxiliaries,
    "allowed_foundational_dependencies": sorted(allowed),
    "allowed_axioms": sorted(allowed),
    "unexpected_dependencies": [],
    "dependency_revisions": dependencies,
    "source_sha256": {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in paths},
    "log_sha256": {name: sha256((audit_dir / name).read_bytes()).hexdigest()
                   for name in ("environment.log", "cache.log", "build.log", "dependencies.log")},
    "source_theorems": source_theorems,
    "theorem_names": sorted(source_theorems),
    "declarations": records,
    "formalized_results": [
        "Signed and centered residual bounds, including the original moment chord envelope",
        "Finite-count absolute envelope bound, chord dominance, and residual containment",
        "Constructive finite-envelope attainment for k<n and extremal weight choices",
        "Mathematical floor remainder contract",
        "Translation invariance of the complete chord envelope and residual center",
        "Closed, open, and mixed-endpoint exact rounding from residual checks",
        "Tie-aware greedy equality, exact persistent writes, and all-future deterministic trace equality"
    ],
    "scope_notes": {
        "arithmetic": "Real-valued mathematics; range bounds and summary sums are premises.",
        "finite_count": "The finite envelope uses an integer k and remainder c. A separate mathematical floor theorem supplies the range and sum decomposition. Python floor/code correspondence is not formalized.",
        "sharpness": "An explicit attaining real row list is proved for k<n; endpoint choices attain both residual extrema for a fixed real row family within an independent weight box.",
        "rounding": "Cell endpoint inclusivity and the cell-to-rounding contract are explicit premises. Concrete BF16 encoding, ties, overflow and cell generation are not formalized.",
        "state": "Persistent state is abstract and must include every future-observable mutable component; the concrete forest/interpreter implementation is not refined to the theorem.",
        "backend": "No numerical bridge is discharged for a deployed attention backend. Python/CUDA binaries, extraction auxiliaries and hardware execution are outside the proof audit."
    },
    "boundary": "Real-valued mathematics and abstract state composition. No proof of BF16 cell generation, Python/CUDA refinement, hardware execution, or a backend numerical bridge."
}
(audit_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"Verified {len(source_theorems)} source theorems; audited {len(records)} compiled constants.")
PY
