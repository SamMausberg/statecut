# Exact-reference algorithm audit

This audit strengthens the Python residual certificate and fixes reproducible
input and provenance failures while preserving `RATIONAL_BF16_V1` (E24).
It covers exact arithmetic, residual rounding predicates, forest construction
and audits, and the two complete-state decoder strategies. It does not prove
Python execution, a GPU backend, or pretrained-model acceleration.

## Sharp finite-count residual

The original chord bound permits fractional mass at the value endpoints. The
number of cache rows is an integer, which gives a stronger bound without new
summary fields. For `n > 0`, `l < u`, and `nl <= S <= nu`, define

\[
r=\frac{S-nl}{u-l},\qquad k=\lfloor r\rfloor.
\]

If `k=n`, set `C_int(t)=n|u-t|`. Otherwise let
`c=l+(r-k)(u-l)` and set

\[
C_{\rm int}(t)=k|u-t|+(n-k-1)|l-t|+|c-t|.
\]

For `l=u`, the bound is `n|l-t|`. The implementation performs every division
as an exact `Fraction`; an integer count is required.

**Why it is sound and attained.** Maximize `sum |v_i-t|` over the bounded
fixed-sum feasible set. If two coordinates are interior, vary them in opposite
directions while preserving their sum. The objective along this feasible
segment is convex, so at least one endpoint does not decrease it. That move
places another coordinate at an endpoint. Repeating leaves at most one
interior coordinate. The total then forces exactly the construction above.
It is feasible and achieves the maximum. This proves

\[
\sum_i|v_i-t|\le C_{\rm int}(t)\le C_{\rm chord}(t).
\]

For common weights `a <= w_i <= b`, put `m=(a+b)/2`, `h=(b-a)/2`.
The interval

\[
[m(S-nt)-hC_{\rm int}(t),\ m(S-nt)+hC_{\rm int}(t)]
\]

contains `sum w_i(v_i-t)`. On the maximizing value construction, assigning
`w_i=m+h sign(v_i-t)` or `m-h sign(v_i-t)` attains the respective extreme.
Thus this is sharp for the count/sum/range/common-weight relaxation. It need
not be sharp after imposing actual key/value relationships or discrete E24
score constraints. The signed-moment intersection remains enabled.

The bound is translation invariant. Only one remainder coordinate can improve
on the chord, and its chord gap is at most `(u-l)/2`. Each residual endpoint
therefore improves by at most `h(u-l)/2`. This limits the possible gain at large
nodes; the implementation does not assume broad acceptance improvements.

The CPU uses this finite-count envelope. The CUDA residual kernel retains the
sound continuous chord. Their acceptance flags can differ while targeting the
same mathematical profile.

## Reproducible strict improvement

Take `q=1`, keys `(-3/20, 3/20, 0)`, and values `(63/4, 65/4, 16)` in one
three-row block. The E24 weight box is
`[3610071/4194304, 2436543/2097152]`. The proposed BF16 result is `16`, whose
closed cell is `[511/32,257/16]`.

At the lower boundary, the chord residual lower endpoint is
`-4862889/268435456`; the finite-count lower endpoint is
`497277/33554432 > 0`. The finite-count upper residual endpoint at the upper
cell boundary is `-440073/4194304 < 0`. Therefore the new direct gate accepts.
The old chord gate rejects and falls back to all three rows. Both return the
exact dense E24 result `16`.

The paired ablation uses the same proposer, forest, signed-moment intersection,
direct gate, and zero-refinement budget; only the absolute-deviation envelope
changes. Twelve constructed cases include six strict acceptance improvements
and six cases where both methods fall back. These are logical raw-read counts,
not timing or pretrained results.

Reproduce the artifact with:

```sh
.venv/bin/python scripts/audit_algorithm.py
```

The output is [integer_moment.json](../results/gh200/integer_moment.json).
It also records an independent fixed-sum enumeration over 5,004 value
multisets, 35,028 multiset/threshold evaluations, and 1,218 moment/threshold
optima. Both residual extremes match exactly; 588 envelopes are strictly
smaller than the chord. The finite grid contains the maximizing remainder for
every enumerated moment. This is an exhaustive check of that grid, separate
from the universal mathematical argument.

## Confirmed failures and corrections

| Failure before this audit | Correction and regression evidence |
| --- | --- |
| `moment_residual(2,0,-m,m,[0,2],0)` with integer `m=2^53+1` returned an upper bound `2^54`, excluding the feasible residual `2^54+2`. Python `/` had introduced a float. | Canonicalize moment inputs to exact fractions before arithmetic. The large-integer regression checks both attained extremes. |
| For float query `(1,1,1)`, a dot with `(2^54,1,-2^54)` became zero in dense evaluation while the tree's interval path retained the exact score one. Two additional zero rows made the old dense BF16 result `171/512`, versus a falsely agreeing tree certificate of `147/256`. | Normalize query/key scalars before exact dot products and normalize flat-summary queries. Dense and both tree gates now return `147/256`, including zero-read acceptance. Float inputs denote their exact represented binary rationals. |
| A float radicand could round a squared BF16 midpoint comparison inside `bf16_rsqrt`, reversing a rounding decision. | Normalize the radicand before squared comparisons. A numerator immediately above the exact midpoint for radicand `0.1` tests both signs. |
| Frozen `Model` and `Layer` dataclasses retained caller-owned lists. Mutating those lists changed weights while `TreeModel` kept the old cached identity and accepted old cache state. | Copy all model vectors and matrices into immutable tuples of fractions, and snapshot the layer sequence. Mutation of every caller container leaves the model and its identity equal to the original fixture. |
| `ForestCache.audit()` accepted a root and tail with different value dimensions, and completed trees with incorrectly sized raw leaves. Noninteger block sizes also defeated the leaf/carry invariant. | Validate dimensions across all leaves, require exactly `block_size` rows in sealed leaves, and require an integer positive block size in both cache implementations. |
| The flat verifier raised an exponential resource error on coordinate-box scores `+/-2048`, although both correlated raw-row scores were zero and dense attention was valid. | Fall back to the exact reference when summary evaluation exhausts its domain or resources; count only attempted summaries and read each raw block once. |

The core tests also cover a failure injected after staging the second layer's
K/V during a write attempt. The dense retry produces the same complete state
and token as a direct dense step; the original state's forest objects and
prefix remain unchanged.

## Remaining trust and scope

Summary soundness relies on constructing immutable caches through append or
auditing an imported cache against its raw rows. Query evaluation deliberately
does not scan the cache to revalidate that provenance. Public Python dataclass
constructors are not an authentication boundary against forged summaries or
manually spliced states.

The audit found no counterexample on canonical exact inputs to the residual
predicate, BF16 cell gate, interval RMSNorm/SiLU propagation, or transaction
logic. The executable cell audit checks all 65,279 finite canonical cells and
130,558 boundary decisions, including both overflow edges. Agreement between
the cell and rounder implementations is a consistency test, not independent
verification of an IEEE implementation.

The full suite at the audit checkpoint completed with **154 passed, 27
skipped** in 15.79 seconds. The skipped cases were CUDA tests in that invocation;
device validation is recorded separately. Existing random exact-state
trajectories and fallback leaf-reuse tests continue to pass. The original v2
experiment is reproduced under the new core in
[algorithm/v2_experiments.json](../results/gh200/algorithm/v2_experiments.json).

Lean checks establish their stated mathematical propositions separately; they
do not by themselves verify this Python code, finite-format implementation, or
a deployed attention backend. E24 resource exhaustion remains a rejected
execution rather than an approximate answer.
