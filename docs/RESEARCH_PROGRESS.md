# Research progress and open obligations

Owner and manuscript author: **Samuel Mausberg**.

The September 5, 2026 revision completes a reproducible research milestone:
an isolated GH200 setup, audited exact-reference implementation, a sharper
finite-count algorithm, compiled mathematical Lean proofs, device arithmetic
validation, two real pretrained checkpoints, and an illustrated ICML-style
LaTeX manuscript. Commits are pushed incrementally to the private repository.

## Established results

The finite-count envelope is sharp for its count/sum/range/weight-box
relaxation and needs no extra summary fields. The code repairs exact-input
coercion, immutable model provenance, cache dimension/leaf invariants, and
faithful fallback behavior. The E24 evaluator now covers the recorded large
positive Qwen scores with rigorous magnitude-aware precision. CUDA uses a
tighter monotone-grid weight enclosure and a proved zero tail.

Current verification is bound by `results/gh200/STATUS.json` and the source
manifest. The main results are 185 passing Python tests, compiled Lean with
only standard foundations, 68,851 host and device arithmetic checks, clean
GPU sanitizer runs, and exact observer-state comparisons on both checkpoints.
The manuscript contains full mathematical arguments and native TikZ figures.

## What the real models show

The controlled workload is twelve combinations of checkpoint, prompt pattern,
and context length. It is not a representative production benchmark. Local
E24 and SDPA disagree on 501 of 4,608 sampled coordinates, despite complete
filtered-vs-dense E24 agreement. All sampled exact-reference evaluations fall
back. Compressed CUDA summaries accept no heads even with the observed SDPA
answer supplied as a proposal. Oracle-proposal acceptance improves from 254 to
280 heads only with one row per summary; that layout has no cache compression
and still accepts no complete attention call. Uniform-mean proposals accept
zero heads at every tested block size.

These results establish implementation and mathematical progress, and expose
the present approach's practical limits. They do not establish an LLM speedup.

## Remaining research

1. A deployed numerical target must be fixed and a sound bridge proved before
   any StateCut output can replace that backend. Empirical discrepancy bounds
   and the current E24 proofs do not meet that condition.
2. Better key/query and value summaries or refinement scheduling must produce
   useful certified acceptance on diverse frozen evaluation workloads. The
   finite-count envelope is already optimal within its stated relaxation;
   broader gains require additional information or a different decomposition.
3. A serving implementation needs incremental GPU summaries, ownership and
   generation binding, masks/positions/GQA support, transactional publication,
   and an exact fallback scheduler. These are absent from the observer.
4. End-to-end cost must include indexing, maintenance, proposals, rejected
   attempts, complete decoder work and all launches. Primitive event times or
   per-head acceptance rates do not prove a token-level benefit.
5. The Lean theorems concern mathematical objects and abstract state. A
   refinement proof for the concrete finite-format interpreter, forest, CUDA
   instructions and actual state inventory remains a separate undertaking.

No universal claim of repository perfection, implementation verification,
novelty priority, conference acceptance or pretrained acceleration is made.
