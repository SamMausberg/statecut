#!/usr/bin/env python3
"""Reproduce exact-rational audits on fixed layer/head samples from each workload."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--output", default="results/gh200/e24")
    args = parser.parse_args()
    if args.workers < 1 or args.heads < 1:
        parser.error("positive workers and heads required")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for manifest_path in args.manifests:
        manifest = json.loads(Path(manifest_path).read_text())
        for workload in manifest["workloads"]:
            layers = workload["layers"]
            for layer in sorted({0, layers // 2, layers - 1}):
                path = Path(workload["capture_directory"]) / f"capture-{layer:04d}.npz"
                expected_hash = workload["captures_sha256"][path.name]
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"capture hash mismatch: {path}")
                jobs.append((manifest["model"], workload["prompt"], workload["context_tokens"], layer, path))

    def run(job):
        model, prompt, context, layer, capture = job
        name = f"{model.split('/')[-1]}-{prompt}-{context}-layer{layer}"
        output = directory / (name + ".json")
        command = [sys.executable, "scripts/audit_capture.py", str(capture), "--heads", str(args.heads),
                   "--block-size", "16", "--max-expansions", "0", "--output", str(output)]
        with (directory / (name + ".txt")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        assert not result["backend_replacement_authorized"]
        assert all(row["e24_status"] in ("passed","unsupported") for row in result["rows"])
        assert result["heads_completed"] == result["heads_requested"]
        print(name, f"{result['supported_heads']} supported, {result['unsupported_heads']} unsupported", flush=True)
        return {"model":model,"prompt":prompt,"context_tokens":context,"layer":layer,
                "capture":str(capture),"result":str(output),"heads":result["rows"]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, jobs))
    heads = [head for row in rows for head in row["heads"]]
    supported = [head for head in heads if head["e24_status"] == "passed"]
    unsupported = [head for head in heads if head["e24_status"] == "unsupported"]
    report = {"scope":"exact E24 filtered-vs-dense parity and observational SDPA bit comparison",
              "sampling":"first decode step; first, middle and last layer; first requested query heads",
              "backend_replacement_authorized":False,
              "all_e24_equal_dense":True if not unsupported else None,
              "all_supported_e24_equal_dense":all(row["e24_equal_dense"] for row in supported),
              "heads_requested":len(heads),"heads_tested":len(supported),"unsupported_heads":len(unsupported),
              "coordinates":sum(row["coordinates"] for row in supported),
              "observed_backend_bit_matches":sum(row["observed_backend_bit_matches"] for row in supported),
              "accepted_without_fallback":sum(row["e24_cut_accepted_without_fallback"] for row in supported),
              "rows":rows}
    (directory / "summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
