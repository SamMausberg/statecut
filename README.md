# StateCut

**Author and project owner: Samuel Mausberg.**

StateCut studies exact attention certificates and exact persistent decoder
writes. Version 0.3.0 adds a sharp finite-count moment bound, compiled Lean
proofs, validated GH200 kernels, real-checkpoint observations, and an illustrated
ICML-style manuscript.

**Paper:** [LaTeX source](paper/main.tex) · [PDF](paper/main.pdf) ·
[complete proofs](paper/appendix.tex) · [citation metadata](CITATION.cff).
The figures use native TikZ/PGFPlots and recorded experiment data.

## What is established

- The finite-count residual envelope is no weaker than the classical chord
  envelope and is sharp for the count/sum/range/common-weight relaxation.
  A three-row example certifies its exact BF16 output from one summary with
  zero raw reads, where the previous envelope falls back to every row.
- Lean 4.19 with pinned mathlib compiles the mathematical certificates,
  including finite-count sharpness, rounding conditions, and abstract
  persistent-state composition. [Formal audit](docs/FORMAL_AUDIT.md).
- The GH200 suite passes 185 Python tests, 68,851 device arithmetic checks,
  68,851 matching host checks, and 2,525 inherited host checks. Device tests
  include every finite canonical BF16 cell; Compute Sanitizer reports zero
  memory errors and race hazards. [Current evidence](results/gh200/).
- SmolLM2-135M and Qwen2.5-0.5B observer runs compare 48 full cache states
  across 12 controlled workloads and preserve all K/V bits, logits, and
  selected tokens. The runs capture 972 decode attention calls.
- On 72 sampled pretrained heads, filtered evaluation equals dense E24 in
  all 4,608 coordinates. E24 differs from the observed SDPA result in **501
  coordinates**. The numerical profiles are not interchangeable.

The tested compressed summaries accept no complete heads at block sizes
16, 64, or 128 on these model workloads, including when given the observed
SDPA answer as an oracle proposal. The improved CUDA enclosure accepts
280/10,908 individual heads at block size one using the observed SDPA oracle
proposal, but no complete attention calls. The uniform-mean summary proposal
accepts zero heads at every tested block size. This is a research artifact
with explicit negative results;
**pretrained acceleration and a deployed-backend equivalence proof remain open**.

## Reproduce on this GH200

```bash
bash scripts/setup_gh200.sh
STATECUT_CUDA_LIBRARY=build-gh200/libstatecut_residual_cuda.so .venv/bin/python -m pytest -q
bash scripts/check_lean.sh
.venv/bin/python scripts/audit_algorithm.py
.venv/bin/python scripts/check_host_residuals.py \
  --binary build-gh200/statecut_device_residual --device \
  --output results/gh200/device_arithmetic.json
.venv/bin/python scripts/check_claims.py
make -C paper
```

The setup creates a repository-local environment while using the machine's
CUDA-enabled ARM64 PyTorch. CPU-only development needs
`pip install -e '.[test]'` and `pytest -q`.

[GH200 instructions](docs/GH200.md) cover model downloads, complete capture
audits, sanitizers, and performance measurements. Model weights and raw
captures remain local; pinned revisions, exact input tokens, capture hashes,
aggregate results, and reproduction scripts are tracked.

## Algorithm and scope

The filter evaluates `sum_i w_i (v_i - t)` directly at rounding-cell
boundaries. Mergeable count, first-moment, and range summaries preserve
normalization dependence. A persistent binary-counter forest supports
refinement, and a transactional interpreter checks every future-visible write.
Rejected attempts use the unchanged exact reference state.

The Python target is the explicit rational `RATIONAL_BF16_V1` profile with
fixed-grid E24 exponentials. CUDA certifies that profile locally using directed
enclosures; it retains the continuous chord envelope while the Python/Lean
path includes the finite-count strengthening. Mathematical proofs do not
constitute a refinement proof for Python, generated CUDA instructions, or
PyTorch. [Reference contract](docs/REFERENCE.md) ·
[verification ledger](docs/VERIFICATION.md) ·
[algorithm audit](docs/ALGORITHM_AUDIT.md) ·
[literature audit](docs/LITERATURE_AUDIT.md).

Imported v0.2.0 measurements at the top of `results/` describe the original
import (`73cd266`); current results are under `results/gh200/`. Earlier v0.1
evidence remains archived. See [result provenance](results/README.md).
