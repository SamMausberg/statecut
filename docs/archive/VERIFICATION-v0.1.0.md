# Verification report

Version 0.1.0. Authoring date: 2026-09-05.

## Status is a property of evidence, not file names

**No end-to-end pretrained-model or GH200 acceleration claim is verified.**
The repository contains a mathematical construction, a tested numerical
reference, and uncompiled Lean/CUDA source. Calling the directory `lean/` does
not make its contents machine-checked.

| Claim | Evidence obtained | Important limitation |
|---|---|---|
| Signed block bounds, normalization, rounding-cut soundness, state preservation | Mathematical proofs in PAPER.md | Assumptions must be instantiated for each numerical backend |
| CPU interval/reference filter | 85 passing pytest tests | Tests do not verify all Python executions |
| BF16 finite encode/decode | All 65,280 finite encodings checked | Signed zero is canonicalized; infinities/NaNs unsupported |
| Exact state across decoding | 16 parametrized 12-step cases; four additional 20-step experiments | Frozen tiny fixture, not a pretrained checkpoint |
| Raw KV avoidance | Instrumented counters show 0/512 and 32/512 cases, plus full-scan cases | Summary, weight, append and independent-oracle reads excluded |
| Host directed-interval arithmetic | C++17 compiled; 1,500 arithmetic plus 1,025 exponential cases pass | Host nextafter emulation, not CUDA directed instructions |
| Lean theorem declarations | Reviewed source and an attempted build script | `lake` unavailable; **no successful compilation** |
| CUDA kernel | Source and smoke test provided | `nvcc` and device unavailable; **no compilation or execution** |
| GH200 throughput/latency | Conditional cost analysis only | **No hardware measurement** |
| Pretrained/backend bitwise equality | Integration contract and error-bridge obligation | **Not implemented or proved for a specific backend** |

The source archive includes the actual logs in `results/`. A supplied CI
configuration is not a green CI run. Rerunning tests may produce different
wall-clock times and environment strings; exact results should remain equal.

## Mathematical proof coverage

The paper proves the algorithm over a specified deterministic transition.
Its reference primitives must terminate on the accepted domain. The central
steps are:

1. Exact immutable summaries contain the currently visible KV prefix.
2. Score and monotone-weight bounds enclose all raw contributions.
3. Signed numerator bounds and positive-denominator division enclose attention.
4. Equal rounded endpoints imply the exact materialized activation.
5. Full-vocabulary inequalities imply the exact greedy decision with its tie rule.
6. Exact materialized cuts preserve all downstream state-producing operations.
7. The final token-only cut is valid only beyond the last live persistent write.
8. The accepted or fallback transition equals the reference transition.
9. Equality of the transition implies equality of every finite future trace.

For the Python profile, the paper also justifies the exact-rational interval
primitives, positive exponential series tail, BF16 midpoint comparisons, and
interval graph propagation. Their correspondence to the actual Python code
was reviewed and tested, not machine-proved.

## Lean coverage and omissions

Every row below describes a **source declaration**, not a compiled theorem in
this authoring environment.

| Module | Statements |
|---|---|
| `Bounds.lean` | Signed term/block sums, weight mass, key-box dot product, real exp monotonicity, generic monotone weights, sign-correct normalization, equal-weight family |
| `Cuts.lean` | Monotone rounding squeeze, vector cuts, tie-aware real-logit argmax, uniqueness |
| `Composition.lean` | Summary-to-rounded-output composition, checked cut-to-whole-transition composition, executable rational argmax gate and its soundness |
| `State.lean` | Sound filter equality, all future traces, exact deterministic suffixes, terminal state cut, state frontier, restricted indistinguishability |

`Composition.summary_to_exact_cut` derives rounded equality from signed bounds
and certificate cross-checks. `checked_cut_transition` then derives equality of
the whole suffix result. Neither assumes the equality it is meant to derive.

`State.all_future_equal` is a generic consequence of a **whole-transition**
soundness hypothesis. On its own it is not proof that the Python model adapter
satisfies that hypothesis. The local composition theorems and the paper explain
how a correct integration would establish it. This distinction is deliberate.

The Lean source does **not** formalize IEEE BF16, libdevice, the Taylor-series
implementation, rounding monotonicity for actual machine encodings, Python
execution, CUDA instructions, cache memory ownership, graph capture, kernel
refinement, or a checkpoint's weight loading. A generic monotonicity or enclosure
hypothesis is not a proof of those facts for a deployed kernel.

The source contains no `sorry`, `admit`, `unsafe`, newly declared `axiom`, or
`native_decide` proof escape. This is a static source observation, not an axiom
audit of a compiled environment. The root Lean file prints significant theorem
axiom dependencies. Expected foundational dependencies in mathlib include
`propext`, `Classical.choice`, and `Quot.sound`; review the actual output after a
successful build and reject unexpected dependencies such as `sorryAx`.

## Trusted computing base

For the executed CPU results, trust Python's integer/Fraction semantics,
interpreter, source execution, and the correspondence between reviewed code and
mathematical formulas. The dense comparison and certified path share arithmetic
primitives. `mpmath` provides independent high-precision numerical cross-checks;
it is not a certified real-number oracle. The all-encoding BF16 test covers the
finite representation map exhaustively, not every rational rounding input.

The C++ host tests trust the compiler, strict binary64 operations, `nextafter`,
and the host floating environment. Flags disable fast math and contraction.
The GPU prototype would additionally trust NVIDIA's documented directed
instruction semantics, compilation preserving them, correct summary construction,
stream ordering, device memory integrity, and a proved reference error bridge.

Cache identities and frozen objects defend against accidental mixing, not a
malicious in-process adversary able to fabricate dataclasses or mutate private
storage. `Cache.audit()` is a full reconstruction check for ingestion/tests and
must not be counted as a zero-scan query. A production loader must establish
summary integrity once and enforce ownership/versioning thereafter.

## Failure behavior and domain

A nonsingleton rounding interval is not success. A tie certificate must honor
the exact index rule. Nonfinite values, BF16 overflow, all-zero weight mass,
exponential resource limits, stale cache identity, and mismatched shapes do not
produce certified predictions. The toy profile may raise; a supported production
reference must provide its own appropriate fallback for such cases.

The CPU profile uses exact rational block reductions and can reuse opened
blocks. A different floating reduction schedule cannot necessarily do so.
Reconstructing partial sums with a different order is not a valid dense fallback
for a bitwise reference. The CPU and CUDA profiles are intentionally different;
see REFERENCE.md.

## Required release gates

A release claiming exact accelerated pretrained decoding needs all of these:

* Successful pinned Lean build and explicit axiom review, plus any additional
  floating-point/code-refinement proofs claimed by that release.
* A reference manifest fixing model, arithmetic, dispatch, masks, state, and tie
  semantics, with exact dense execution as fallback.
* A proved sound arithmetic bridge for every accepted kernel path. A maximum
  error observed in testing is not a proof.
* Bitwise tests of every new layer/head KV entry and greedy token over long
  trajectories, adversarial rounding, ties, branches, prefix reuse, and batching.
* GPU memory/concurrency validation, correct append/publish events, and no stale
  weight/cache or hidden dense verification work in the fast path.
* GH200 latency and bandwidth measurements counting summary creation, updates,
  metadata, candidate work, refinement, failed certificates, fallback and weights.

Any unmet gate must remain visible in the release's claim table.
