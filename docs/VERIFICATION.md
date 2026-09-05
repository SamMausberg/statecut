# Verification ledger and trusted boundary

## Status vocabulary

Use **mathematical proof**, **Lean source**, **compiled Lean proof**, **host test**, **device test**, and **pretrained end-to-end result** as separate labels. A source file or a CI configuration is not evidence that its build passed. Current recorded outcomes are in `results/STATUS.json`; archived v0.1 evidence is under `results/v0.1.0/`.

The release has handwritten proofs and executed Python/C++ host tests. Lean is unavailable in this environment; CUDA compilation/execution and a real checkpoint run are unavailable. No Lean/device/pretrained exactness or speedup claim is made.

## Statement-to-source map

| Mathematical obligation | Lean source / theorem | Executable evidence | Remaining boundary |
|---|---|---|---|
| Signed sums, mass and key boxes | `Bounds.lean` | original exact-reference tests | source not compiled |
| Absolute-value chord | `Residual.abs_chord_mul`, `abs_chord_sum` | randomized exact Fraction cases | source not compiled |
| Count/sum/range residual | `moment_residual_sound` | exact rational random cases; host enclosure tests | no Python/C++ refinement proof |
| Residual-to-actual-output cut | `ResidualCut.moment_to_exact_cut` | direct-vs-divided ablation | BF16 cell implementation not formalized |
| Strict cell thresholds | `residual_lower_strict`, `residual_upper_strict` | odd-cell/midpoint tests | no full open-cell implementation theorem |
| Disjoint frontier composition | `Frontier.lean` | prefix audits; single-read fallback tests | concrete forest algorithm not formalized |
| Rounding and tie-aware argmax | `Cuts.lean`, `Composition.lean` | tie, rounding and small-model tests | full finite-format interpreter not formalized |
| Exact writes and whole future | `WriteFrontier.lean`, `State.lean` | complete K/V equality over continued decoding | all deployed persistent buffers must be inventoried |
| Backend numerical bridge | `backend_bridge_gate` | intentionally no claimed bridge | premise not discharged for PyTorch or FlashAttention |

The module names in this table identify files; theorem constants are in the `StateCut` namespace. Run `bash scripts/check_lean.sh` from the root to build the pinned Lean 4.19.0/mathlib sources and print selected theorem dependencies. Repair any elaboration/tactic failures before describing them as checked. Absence of placeholder tokens is necessary discipline, not a substitute for compilation. Lean's standard foundations may appear in printed dependencies; no application-specific unproved axiom should be accepted.

## Mathematical arithmetic contract of the CUDA draft

For `|x|<=64`, put `t=|x|/128<=1/2`. After the Taylor terms through degree 20, the tail obeys

\[
0\le e^t-\sum_{k=0}^{20}t^k/k!\le
\frac{t^{21}/21!}{1-t/22}.
\]

The ratio of each later term to its predecessor is at most `t/22`; summing the geometric majorant proves the bound. Enclose each operation outward, square seven times to enclose `exp(|x|)`, and invert with a positive denominator for negative `x`. The source uses this construction rather than trusting an empirical libdevice-error table. The domain and finiteness checks fail closed.

For any real `z`, rounding to E24 changes `exp(z)` by at most `2^-25` in absolute value. Expanding the positive-series enclosure by that amount encloses the E24 weight. The zero-score case is exactly one. Interval input scores use monotonicity. Counts through uint32 range are exactly representable in FP64; the Python/CUDA wrapper restricts to positive signed-int-compatible counts.

Directed additions, multiplications and divisions are part of the device contract [R10]. Host emulation widens ordinary binary64 operations with `nextafter`; host tests require non-fast-math compilation and do not prove the device path. The BF16 cell endpoints are exactly representable in FP64, including subnormals and the finite overflow boundary. All finite canonical cells are cross-checked on the host.

The proof-to-binary chain is not closed: compiler lowering, directed instruction selection, reduction synchronization, launch ABI, every generated instruction, and the GPU implementation have not been formally verified or executed here. Host test success must not be promoted to that claim.

## Provenance and state assumptions

Summaries are trusted products of the actual rows, not attacker-supplied claims. `audit()` rereads all rows and is permitted for ingestion/tests only. It is deliberately absent from query paths. Public Python dataclasses can be forged; their constructors are not cryptographic authenticity checks. The CUDA wrapper validates shape and layout, not the provenance of every descriptor. Real serving needs ownership/versioning and synchronization, not a token-shape hash.

Every new reference K/V write is staged once, kept private, and published only after the whole transition is accepted. Failed attempts do not mutate snapshots. No approximate persistent values, delayed repair queue, unrecorded fallback work or hidden dense proposal is permitted. The full-state theorem is conditional on including *every* future-observable state component; it does not license omitting an inconvenient buffer.

## Reproduction and evidence retention

`pytest -q` executes the reference tests. Torch/NumPy capture tests skip when those optional packages are absent, so compare both pass and skip counts. `check_host_intervals.py` and `check_host_residuals.py` invoke compiled host binaries and write machine-readable results. `run_v2_experiments.py` writes the paired logical-read and complete-state experiments. Capture-only checkpoint scripts report coverage and never authorize a backend replacement.

`check_lean.sh` exits 2 when Lake is missing, rather than succeeding with an unverified status. CUDA/benchmark attempts also return nonzero when their prerequisites are absent. `scripts/check_claims.py` checks consistency of the release's status records; it is an evidence-label guard, not a semantic proof checker. The source hash manifest identifies the tested artifact but does not establish correctness.

References: [REFERENCES.md](REFERENCES.md).
