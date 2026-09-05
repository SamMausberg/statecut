#!/usr/bin/env python3
"""Check consistency of recorded release evidence, not theorem correctness."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def main():
    r = ROOT/"results"
    status=json.loads((r/"STATUS.json").read_text())
    found=re.search(r"(\d+) passed",(r/"pytest.txt").read_text())
    assert found and int(found[1]) == status["python_tests_passed"]
    assert json.loads((r/"host_residuals.json").read_text())["passed"] == status["new_host_cases_passed"]
    assert json.loads((r/"host_interval_tests.json").read_text())["passed"] == status["inherited_host_cases_passed"]
    assert status["lean_compiled"] is False
    assert status["cuda_compiled"] is False and status["cuda_executed"] is False
    assert status["pretrained_backend_equivalence_proved"] is False
    assert status["gh200_speedup_measured"] is False
    experiment=json.loads((r/"v2_experiments.json").read_text())
    largest=[x for x in experiment["attention"] if x["entries"] == 8192]
    assert len(largest) == 3
    best=next(x for x in largest if x["method"] == "v2-direct-residual-tree")
    assert best["summary_records_read"] == 1 and best["raw_entries_read"] == 0
    assert all(x["raw_entries_read"] == 8192 for x in largest if x is not best)
    assert all(x["all_tokens_and_states_equal"] for x in experiment["state_trajectories"])
    assert sum(x["steps"] for x in experiment["state_trajectories"]) == 192
    assert json.loads((r/"synthetic_capture_audit.json").read_text())["backend_replacement_authorized"] is False
    # Source scan is recorded separately and never counted as a Lean build.
    bad=[]
    for path in (ROOT/"lean/StateCut").glob("*.lean"):
        if re.search(r"\b(sorry|admit|axiom|unsafe)\b",path.read_text()):
            bad.append(str(path))
    assert not bad, bad
    manifest=r/"SOURCE_MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest,relative=line.split("  ",1)
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    print("Recorded claim labels and source manifest consistent. Not a Lean/device proof.")

if __name__ == "__main__":
    main()
