# Changelog

## 0.2.0

Adds denominator-free first-moment residual predicates, exact BF16 cell boundaries, a persistent binary-counter forest, bounded refinement with single-read rational fallback, and write-frontier interval execution with transactional dense fallback. Retains v0.1 as a numerical baseline and checks new implementations against its dense decoder.

Adds residual/frontier/write-frontier Lean sources and a concrete moment-to-cut composition theorem. Formal compilation remains unverified. Adds outward host/device residual arithmetic, an SM90 local-gate ABI and microbenchmark, plus capture-only pretrained observation tools that do not authorize a backend replacement.

Expands proofs, audit notes, related work, reproduction instructions, boundary/adversarial tests and paired experiments. Archives original evidence without relabelling it as new results. No measured GPU or pretrained speedup is claimed.

## 0.1.0

Original user-provided StateCut archive: exact-rational attention and frozen decoder, flat signed-moment summaries, numerical cuts, future-state contract, initial Lean sources and real-softmax CUDA prototype.
