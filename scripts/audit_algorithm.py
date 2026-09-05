#!/usr/bin/env python3
"""Reproduce the finite-count residual audit using exact rational arithmetic.

The ablation changes only the absolute-deviation envelope used by the same
tree verifier. It measures logical reads, with no GPU or backend-speed claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from statecut.arithmetic import Interval, bf16, certify_bf16, round_bf16_bits
from statecut.cache import Entry
from statecut.forest import ForestCache
from statecut.residual import (bf16_cell, chord_abs_sum, integer_abs_sum,
                               moment_residual, summary_residual, weight_box)
from statecut.tree_attention import dense_tree_attention, verify_tree_attention


def exhaustive_integer_moments() -> dict:
    grid = tuple(F(i, 4) for i in range(-4, 5))
    thresholds = (F(-2), F(-3, 4), F(-1, 8), F(0), F(1, 3), F(1), F(2))
    weights = Interval(F(2, 3), F(7, 5))
    row_multisets = evaluations = optima = 0
    strict_improvements = 0
    for n in range(1, 7):
        rows = tuple(combinations_with_replacement(grid, n))
        row_multisets += len(rows)
        for t in thresholds:
            extremes = {}
            deviations = {}
            for vs in rows:
                evaluations += 1
                total = sum(vs, F(0))
                lower = sum((weights.lo if v >= t else weights.hi)*(v-t) for v in vs)
                upper = sum((weights.hi if v >= t else weights.lo)*(v-t) for v in vs)
                old = extremes.get(total, (lower, upper))
                extremes[total] = (min(old[0], lower), max(old[1], upper))
                deviations[total] = max(deviations.get(total, F(0)), sum(abs(v-t) for v in vs))
            for total, (lower, upper) in extremes.items():
                optima += 1
                sharp = integer_abs_sum(n, total, -1, 1, t)
                chord = chord_abs_sum(n, total, -1, 1, t)
                assert sharp == deviations[total] and sharp <= chord
                assert moment_residual(n, total, -1, 1, weights, t) == Interval(lower, upper)
                strict_improvements += sharp < chord
    return {
        "arithmetic": "exact fractions; exhaustive finite grid, not a universal proof",
        "grid": list(map(str, grid)), "counts": [1, 2, 3, 4, 5, 6],
        "thresholds": list(map(str, thresholds)),
        "weight_box": [str(weights.lo), str(weights.hi)],
        "row_multisets": row_multisets,
        "row_threshold_evaluations": evaluations,
        "fixed_sum_threshold_optima": optima,
        "both_residual_extremes_match": True,
        "strictly_smaller_envelopes": strict_improvements,
        "source_test": "tests/test_residual.py::test_integer_envelope_against_exhaustive_fixed_sum_optimization",
    }


def finite_cells() -> dict:
    cells = edges = overflow_edges = 0
    for bits in range(65536):
        if bits == 0x8000 or bits & 0x7f80 == 0x7f80:
            continue
        cell = bf16_cell(bits)
        cells += 1
        assert round_bf16_bits(cell.value) == bits
        assert round_bf16_bits((cell.lo+cell.hi)/2) == bits
        for edge in (cell.lo, cell.hi):
            edges += 1
            try:
                belongs = round_bf16_bits(edge) == bits
            except OverflowError:
                belongs = False
                overflow_edges += 1
            assert belongs == cell.closed
    return {"canonical_finite_cells": cells, "boundary_decisions": edges,
            "overflow_boundaries": overflow_edges, "all_consistent": True,
            "scope": "exact cell/rounder consistency; not independent IEEE implementation verification"}


def build_family(n: int, score: F) -> ForestCache:
    assert n > 0 and n % 2 == 1
    forest = ForestCache("integer-moment-audit", n)
    for i in range(n-1):
        sign = -1 if i % 2 == 0 else 1
        forest = forest.append(Entry((sign*score,), (F(16)+sign*F(1, 4),)))
    forest = forest.append(Entry((0,), (16,)))
    forest.audit()
    return forest


def verify(forest: ForestCache):
    return verify_tree_attention(forest, (F(1),), certify_bf16,
                                 lambda a: tuple(map(bf16, a)),
                                 max_expansions=0, direct_bf16=True)


def ablate(n: int, score: F) -> dict:
    forest = build_family(n, score)
    expected = tuple(map(bf16, dense_tree_attention(forest, (F(1),))))
    # Replace just the stronger envelope by the sound original chord. All
    # proposals, summaries, signed intersections and budgets stay identical.
    with patch("statecut.residual.integer_abs_sum", chord_abs_sum):
        old = verify(forest)
    new = verify(forest)
    assert old.value == new.value == expected == (F(16),)
    return {"rows": n, "score_magnitude": str(score), "block_size": n,
            "expected_bf16": [str(x) for x in expected],
            "continuous_chord": {"accepted": old.accepted_from_bounds,
                                 "gate": old.gate, "reads": asdict(old.stats)},
            "integer_count": {"accepted": new.accepted_from_bounds,
                              "gate": new.gate, "reads": asdict(new.stats)},
            "both_equal_e24_dense": True}


def strict_three_row() -> dict:
    report = ablate(3, F(3, 20))
    forest = build_family(3, F(3, 20))
    summary = forest.roots[0].summary
    weights = weight_box((Interval.point(1),), summary)
    cell = bf16_cell(0x4180)
    report["cell"] = {"bits": "0x4180", "value": "16", "closed": True,
                      "lo": str(cell.lo), "hi": str(cell.hi)}
    report["weight_box"] = [str(weights.lo), str(weights.hi)]
    report["keys"] = ["-3/20", "3/20", "0"]
    report["values"] = ["63/4", "65/4", "16"]
    for name, t in (("lower_boundary", cell.lo), ("upper_boundary", cell.hi)):
        with patch("statecut.residual.integer_abs_sum", chord_abs_sum):
            old = summary_residual(summary, weights, 0, t)
        new = summary_residual(summary, weights, 0, t)
        report[name] = {"continuous_chord": [str(old.lo), str(old.hi)],
                        "integer_count": [str(new.lo), str(new.hi)]}
    assert not report["continuous_chord"]["accepted"]
    assert report["integer_count"]["accepted"]
    assert report["integer_count"]["reads"]["raw_entries"] == 0
    report["source_test"] = "tests/test_tree_attention.py::test_integer_count_gate_accepts_where_continuous_chord_rejects"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/gh200/integer_moment.json")
    args = parser.parse_args()
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "RATIONAL_BF16_V1 exact Python reference; logical reads, no GPU speed claim",
        "python": platform.python_version(),
        "integer_moment_exhaustive": exhaustive_integer_moments(),
        "strict_three_row": strict_three_row(),
        "ablation": [ablate(n, score) for score in (F(13, 100), F(3, 20))
                     for n in (3, 5, 9, 17, 33, 65)],
        "bf16_cells": finite_cells(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                          for path in sorted((ROOT / "src/statecut").glob("*.py"))
                          if path.name not in ("capture.py", "cuda_frontier.py")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
