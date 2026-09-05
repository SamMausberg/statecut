#!/usr/bin/env python3
"""Bind current research claims to recorded checks and their source artifacts.

--record refreshes the ledger after explicitly rerunning affected checks.
The default verifies it without changing files. Neither mode proves software.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/gh200"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def checked_sources(manifest):
    for relative, digest in manifest.items():
        assert sha(ROOT / relative) == digest, f"Stale source evidence: {relative}"


def collect():
    test_log = (EVIDENCE / "logs/pytest_final.txt").read_text()
    passed = re.search(r"(\d+) passed", test_log)
    assert passed and not re.search(r"\d+ (?:failed|error|skipped)", test_log)
    formal = read("formal/summary.json")
    assert formal["build_passed"] and not formal["unexpected_dependencies"]
    checked_sources(formal["source_sha256"])
    moment = read("integer_moment.json")
    checked_sources(moment["source_sha256"])
    assert moment["integer_moment_exhaustive"]["both_residual_extremes_match"]
    assert moment["strict_three_row"]["both_equal_e24_dense"]
    assert moment["strict_three_row"]["integer_count"]["reads"]["raw_entries"] == 0
    for name in ["device_arithmetic.json", "host_arithmetic.json"]:
        evidence = read(name)
        assert evidence["passed"] == sum(evidence[k] for k in [
            "moment_cases", "all_finite_canonical_bf16_cells", "e24_weight_cases",
            "e24_grid_rounding_cases", "exact_negative_tail_cases"])
    memcheck = (EVIDENCE / "logs/sanitizer_frontier_memcheck.txt").read_text()
    racecheck = (EVIDENCE / "logs/sanitizer_frontier_racecheck.txt").read_text()
    assert "ERROR SUMMARY: 0 errors" in memcheck
    assert "0 hazards displayed (0 errors, 0 warnings)" in racecheck
    gpu_tests = re.search(r"(\d+) passed", memcheck)
    assert gpu_tests and re.search(r"(\d+) passed", racecheck)[1] == gpu_tests[1]
    models = [read("pretrained_smollm2-135m.json"), read("pretrained_qwen2.5-0.5b.json")]
    workloads = [w for model in models for w in model["workloads"]]
    assert all(model["revision"] == model["resolved_revision"] for model in models)
    assert all(w["all_layer_kv_bit_equal"] and w["last_position_logits_equal"] for w in workloads)
    assert all(w["captures"] == len(w["captures_sha256"]) == w["layers"] * w["decode_steps"] for w in workloads)
    e24 = read("e24/summary.json")
    assert e24["all_e24_equal_dense"] and e24["all_supported_e24_equal_dense"]
    assert e24["unsupported_heads"] == 0 and not e24["backend_replacement_authorized"]
    cuda = read("capture_cuda_audit.json")
    assert not cuda["backend_replacement_authorized"] and cuda["performance_claim"] is None
    assert all(0 <= row["accepted_heads"] <= row["heads"] for row in cuda["rows"])
    aggregates = cuda["aggregates"]
    for aggregate in aggregates:
        selected = [row for row in cuda["rows"] if all(
            row[k] == aggregate[k] for k in ["model","context_tokens","block_size","proposal"])]
        for key in ["heads","accepted_heads","invalid_weight_heads","nonpositive_mass_heads"]:
            assert aggregate[key] == sum(row[key] for row in selected)
        assert aggregate["calls"] == len(selected)
        assert aggregate["accepted_calls"] == sum(row["accepted_heads"] == row["heads"] for row in selected)
    experiment = read("algorithm/v2_experiments.json")
    assert all(row["all_tokens_and_states_equal"] for row in experiment["state_trajectories"])
    benchmark = read("microbenchmark_h8_n32768_d64.json")
    assert benchmark["observed_SDPA_cut_equality"]
    assert benchmark["end_to_end_speedup"] is None and not benchmark["backend_equivalence_proved"]
    single = [row for row in aggregates if row["block_size"] == 1 and row["proposal"] == "observed_sdpa_oracle"]
    return {
        "schema":"statecut-gh200-evidence-v1", "version":"0.3.0",
        "python_tests_passed":int(passed[1]), "python_tests_skipped":0,
        "lean_compiled":True, "lean_source_theorems":formal["source_theorem_count"],
        "lean_custom_axioms":False, "cuda_compiled":True, "cuda_executed":True,
        "cuda_frontier_tests":int(gpu_tests[1]),
        "device_arithmetic_cases":read("device_arithmetic.json")["passed"],
        "host_arithmetic_cases":read("host_arithmetic.json")["passed"],
        "inherited_host_cases":read("host_intervals.json")["passed"],
        "compute_sanitizer_memcheck_errors":0, "compute_sanitizer_race_hazards":0,
        "pretrained_models":len(models), "controlled_workloads":len(workloads),
        "observer_cache_states_compared":sum(w["states_compared"] for w in workloads),
        "decode_attention_calls_captured":sum(w["captures"] for w in workloads),
        "e24_heads_checked":e24["heads_tested"], "e24_coordinates_checked":e24["coordinates"],
        "e24_sdpa_coordinate_mismatches":e24["coordinates"]-e24["observed_backend_bit_matches"],
        "cuda_single_row_oracle_heads_accepted":sum(row["accepted_heads"] for row in single),
        "cuda_single_row_oracle_heads_checked":sum(row["heads"] for row in single),
        "cuda_complete_attention_calls_accepted":sum(row["accepted_calls"] for row in aggregates),
        "cuda_compressed_heads_accepted":sum(row["accepted_heads"] for row in aggregates if row["block_size"] > 1),
        "pretrained_backend_equivalence_proved":False, "pretrained_speedup_measured":False,
        "scope":"mathematical proofs, implementation tests, and pretrained observations; no compiled-code refinement or pretrained acceleration proof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    current = collect()
    status_path = EVIDENCE / "STATUS.json"
    manifest_path = EVIDENCE / "SOURCE_MANIFEST.sha256"
    if args.record:
        current["recorded_utc"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(current,indent=2)+"\n")
        paths = []
        for folder in ["src","tests","cuda","scripts","lean/StateCut",".github"]:
            paths.extend(path for path in (ROOT/folder).rglob("*")
                         if path.is_file() and path.suffix in (".py",".cu",".cuh",".cpp",".lean",".sh",".yml"))
        paths.extend(ROOT/name for name in ["CMakeLists.txt","pyproject.toml","requirements-gh200.txt",
                     "requirements-test.txt","lean/StateCut.lean","lean/Audit.lean","lean/lean-toolchain",
                     "lean/lakefile.toml","lean/lake-manifest.json"])
        manifest_path.write_text("".join(f"{sha(path)}  {path.relative_to(ROOT)}\n" for path in sorted(set(paths))))
    else:
        recorded = read("STATUS.json")
        recorded.pop("recorded_utc")
        assert recorded == current, "Status ledger is stale; rerun affected checks before --record"
        for line in manifest_path.read_text().splitlines():
            digest, relative = line.split("  ",1)
            assert sha(ROOT/relative) == digest, f"Source changed after recorded validation: {relative}"
    print(json.dumps(current,indent=2))
    print("Evidence labels and source hashes verified; this is not a software correctness proof.")


if __name__ == "__main__":
    main()
