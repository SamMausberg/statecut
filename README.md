# StateCut

Exact-state attention filters. Version 0.2.0 extends the supplied v0.1.0 repository.

**New result:** evaluate `sum w_i(v_i - t)` directly at rounding-cell boundaries, using mergeable count/first-moment/range summaries. A persistent forest avoids a compulsory linear metadata scan. A write-frontier interpreter can preserve exact K/V even when transient attention activations remain uncertain.

**Status:** tested exact-rational reference and host arithmetic. **Lean sources are uncompiled. CUDA is uncompiled/unexecuted here. No pretrained-backend bridge or measured GH200 speedup is claimed.** This is a research artifact, not a drop-in LLM accelerator.

## Evidence

| Check | Recorded result |
|---|---|
| Python regression suite | 135 passed |
| New C++ host checks | 68,292 passed, including all 65,279 finite canonical BF16 cells |
| Inherited C++ host checks | 2,525 passed |
| Nonconstant 8,192-row ablation | Direct gate: 1 summary, 0 raw rows. Both divided variants: all 8,192 raw rows. |
| Random frozen-decoder trajectories | 192 steps across two strategies, identical tokens and complete states |
| Write-frontier separating example | Exact writes/token despite a non-singleton attention cut; 0 raw reads |
| Lean / GPU / pretrained acceleration | Not validated |

The zero-read result is synthetic and follows preprocessing. It is not an LLM acceptance rate or latency measurement. Unstructured inputs can require the full cache. See [status](results/STATUS.json), [experiments](results/v2_experiments.json) and [verification boundaries](docs/VERIFICATION.md).

## Run

```bash
python -m pip install -r requirements-test.txt
python -m pip install -e .
pytest -q
python scripts/run_v2_experiments.py
cmake -S . -B build && cmake --build build --parallel
python scripts/check_host_intervals.py
python scripts/check_host_residuals.py
bash scripts/check_lean.sh  # requires the pinned Lean/mathlib toolchain
```

[GH200 build, microbenchmark and checkpoint observation](docs/GH200.md) includes exact commands and rejection criteria. The observer never replaces SDPA outputs. Its offline cross-profile matches do not authorize a backend replacement.

## Structure

```text
src/statecut/residual.py        centered residuals and exact BF16 cells
src/statecut/forest.py          persistent binary-counter summaries
src/statecut/tree_attention.py  frontier refinement and exact fallback
src/statecut/tree_model.py      exact cuts and persistent-write gates
cuda/residual_frontier.cu       SM90 local E24 cut prototype
lean/StateCut/                  mathematical theorem sources
scripts/                       reproductions, host checks, GH200 and capture tools
tests/                         regression, adversarial and state-equality checks
docs/PAPER.md                  definitions and mathematical proofs
```

Read the [paper](docs/PAPER.md), [audit](docs/AUDIT.md), [changelog](CHANGELOG.md) and [primary references](docs/REFERENCES.md). Original algorithms remain as paired baselines; original evidence is archived under `results/v0.1.0/`. MIT license retained.
