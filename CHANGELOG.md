# Changelog

## 0.3.0 — September 5, 2026

Author and project owner: Samuel Mausberg.

Adds a sharp finite-count residual envelope with exact-reference ablations and
compiled Lean proofs of its bound, dominance, attainment and rounding cut.
Repairs scalar coercion, mutable model inputs, forest invariants and fallback
handling; extends the rigorous E24 oracle to the recorded large Qwen scores.

Validates SM90 code on the GH200, tightens E24 weight bounds by monotone lattice
rounding, and adds exhaustive device-cell checks, rational frontier tests and
sanitizer evidence. Pins an isolated ARM64 environment and preserves imported
results with explicit provenance.

Adds complete observer K/V comparisons on two pinned pretrained checkpoints,
972 captured attention calls and 72 exact-rational head audits. Retains zero
compressed-summary acceptance and 501 E24/SDPA coordinate differences as
negative results. No pretrained acceleration or backend bridge is claimed.

Adds an ICML 2026 LaTeX manuscript, full proofs, native TikZ/PGFPlots figures,
generated result tables, PDF, citation metadata, reproduction commands and CI.

## 0.2.0

Adds denominator-free first-moment residual predicates, exact BF16 cell boundaries, a persistent binary-counter forest, bounded refinement with single-read rational fallback, and write-frontier interval execution with transactional dense fallback. Retains v0.1 as a numerical baseline and checks new implementations against its dense decoder.

Adds residual/frontier/write-frontier Lean sources and a concrete moment-to-cut composition theorem. Formal compilation remains unverified. Adds outward host/device residual arithmetic, an SM90 local-gate ABI and microbenchmark, plus capture-only pretrained observation tools that do not authorize a backend replacement.

Expands proofs, audit notes, related work, reproduction instructions, boundary/adversarial tests and paired experiments. Archives original evidence without relabelling it as new results. No measured GPU or pretrained speedup is claimed.

## 0.1.0

Original user-provided StateCut archive: exact-rational attention and frozen decoder, flat signed-moment summaries, numerical cuts, future-state contract, initial Lean sources and real-softmax CUDA prototype.
