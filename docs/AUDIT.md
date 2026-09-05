# Historical audit of the supplied v0.1.0 and changes in v0.2.0

This is the imported audit. Current v0.3.0 fixes and evidence are in
[ALGORITHM_AUDIT.md](ALGORITHM_AUDIT.md), [FORMAL_AUDIT.md](FORMAL_AUDIT.md)
and [results/gh200](../results/gh200/).

The supplied v0.1.0 evidence is retained in the archive directories; the GitHub repository starts with the v0.2.0 import at `73cd266`. Its README/manuscript/verification note are archived under `docs/archive/`; its recorded outputs moved to `results/v0.1.0/`. The original reference algorithms and tests remain available as baselines, rather than silently replacing the target arithmetic.

## Findings and response

| Finding | Response |
|---|---|
| Current-token agreement can leave divergent future K/V. | Retain the full transition contract and add an implementation that checks the actual write frontier. Full state is compared in tests. |
| Independent numerator/denominator bounds lose cancellation. | Add the centered count/sum/range residual and test it directly at rounding-cell endpoints. Keep a centered-quotient ablation to isolate this difference. |
| v0.1 queries scan every block summary. | Add persistent binary-counter roots and a disjoint refinement frontier. Root hits do not audit/hash/traverse raw history. |
| v0.1 model identity is recomputed from weights on each check. | Cache the immutable model identity in `TreeModel` construction. Do not count a weight hash as free per-token work. |
| Whole-step interval propagation can be very loose. | Keep two strategies. Exact activation cuts are the practical conservative baseline; root-only write-frontier attempts fail closed. Limited acceptance results are retained. |
| Rational and deployed floating-point reductions are different. | Preserve E24 as an explicit reference. Keep real-softmax CUDA code separately labelled. Do not use empirical epsilons to claim PyTorch equivalence. |
| Formal source was not compiled. | Add concrete residual-to-cut and write-frontier sources, but keep formal status false. Provide strict build and dependency-audit commands. |
| No measured GH200/pretrained result exists. | Supply a local CUDA gate, device microbenchmark and capture-only pretrained audit. Do not call the microbenchmark an LLM speedup. |
| Public summary objects are forgeable. | State a trusted-builder model; retain expensive ingestion audits and require serving version/owner provenance. Shape checks are not authenticity proofs. |

## Tests that prevent misleading success

The expanded suite covers negative values, canonical zero, BF16 subnormals/overflow edges, midpoint parity, positive-denominator requirements, random exact residual containment, every finite canonical BF16 center, immutable forest coverage, fallback raw-leaf reuse, foreign-state rejection and rollback. A loose summary box outside the oracle domain falls back to valid actual row scores rather than incorrectly rejecting the target. It includes a direct-gate case that the divided gate cannot certify, a write-frontier case with unresolved transient attention, and direct comparisons of all new strategies with the **original v0.1 dense decoder** under arbitrary supplied continuations.

The complete future-state comparisons include each raw K/V entry, layer/prefix identity and position. Matching generated text alone is not used as state evidence. Synthetic capture tests verify that the observer preserves returned output bits, restores the original function on exceptions, and refuses unsupported masking/length semantics.

## Explicit unresolved work

A successful pinned Lean build; a concrete proof of the finite-format interpreter and forest implementation; a sound numerical bridge to one pinned pretrained backend; GPU summary maintenance/refinement and transactional serving integration; device arithmetic adversarial tests; checkpoint coverage; held-out all-layer/full-state parity; and same-contract end-to-end GH200 benchmarks remain open engineering/research tasks for this artifact.

No guarantee of pretrained acceptance, unconditional sublinear inference, novelty priority, or commercial readiness is inferred from the supplied tests.
