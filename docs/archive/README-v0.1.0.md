# StateCut

**Exact-decision certificates with exact future state.**

Research prototype. Summarize KV blocks, bound attention, and refine only what
is needed to certify either an exact materialized activation or a terminal
argmax. Keep exact raw KV for fallback. Never persist an approximate state just
because the current token agrees.

**This is not a completed pretrained-model accelerator.** The CPU reference
and host arithmetic tests pass. Lean source is provided but was not compiled
in the authoring environment. CUDA was not compiled or executed. No pretrained
checkpoint, FlashAttention equivalence, or GH200 speedup has been validated.

## Evidence

| Component | Recorded status |
|---|---|
| Mathematical proof | [Full derivation](docs/PAPER.md), with explicit numerical and state assumptions |
| Python reference | **85 tests passed**, including exact KV-state trajectories |
| BF16 arithmetic | Every finite encoding round-tripped; midpoint, subnormal, and double-rounding cases tested |
| C++ host intervals | **2,525 cases passed**; directed CUDA instructions were not executed |
| Lean | Source without proof placeholders; **uncompiled**, not machine-verified |
| CUDA / GH200 | Kernel draft and smoke test; **uncompiled, unmeasured** |
| Pretrained checkpoint | **No adapter or checkpoint validation** |

The formalization does not verify the Python/CUDA implementations. Successful
compilation would establish the stated abstract Lean theorems, not close the
floating-point backend or code-refinement obligations. See
[verification boundaries](docs/VERIFICATION.md).

## Method

For each immutable KV block, store a key box, count, and separate positive and
negative value sums. An exact query gives lower/upper weight bounds `a,b`.
For value coordinate `j`, with signed moments `P[j] >= 0`, `M[j] <= 0`:

```
D_block in [count*a, count*b]
N_block[j] in [a*P[j] + b*M[j], b*P[j] + a*M[j]]
```

Enclose `N/D`, with sign-correct division. At an intermediate attention cut,
accept only when both endpoints round to the same **reference** BF16 value in
every coordinate. Resume the deterministic suffix with that exact vector.
After the final KV writes, a sound full-vocabulary argmax certificate can be
weaker. Ambiguity triggers refinement or reference fallback, never an
uncertified prediction.

The local proof implies equality of the entire one-step transition. Equality
of all future token/state traces follows. [Reference semantics](docs/REFERENCE.md)
explain why this does not automatically mean bitwise equality to a deployed
floating-point runtime.

## Reproduce the executed checks

Python 3.11+ and a C++17 compiler are required. Commands run from the repository
root. Recorded execution used Python 3.13.5 and GCC 14.2.0. Exact test dependency
versions are in `requirements-test.txt`; the tested build backend is pinned in
`pyproject.toml`.

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/run_cpu_experiments.py
cmake -S . -B build
cmake --build build --parallel 2
python scripts/check_host_intervals.py
```

The last script uses `mpmath` as an independent high-precision numerical
cross-check, not as a theorem prover. The interval construction itself has a
mathematical derivation. The dense and certified CPU paths share specified
arithmetic primitives, but only the certified path uses summaries.

## Lean and CUDA checks that still need execution

```bash
# Requires elan / Lean; dependencies need network access.
bash scripts/check_lean.sh

# Requires a CUDA development environment and an SM90 device, such as GH200.
cmake -S . -B build-gh200 -DSTATECUT_CUDA=ON
cmake --build build-gh200 --parallel 2
./build-gh200/statecut_cuda_smoke
```

Lean is pinned to 4.19.0 and mathlib to
`c44e0c8ee63ca166450922a373c7409c5d26b00b`. Compilation was attempted but `lake`
was absent. A CI workflow is included; it has **not** been run. The CUDA smoke
test is synthetic and cannot validate a pretrained backend.

## Recorded raw-KV experiments

Each attention case has 512 rows and 16 summaries, block size 32. Counts are
logical raw rows consumed by the filter, not HBM byte or latency measurements.
The separate dense oracle is outside that counter.

| Synthetic case | Raw rows read | Exact reference output? |
|---|---:|---|
| Constant keys, variable signed values | 0 / 512 | Yes |
| Dominant block, nonconstant keys | 32 / 512 | Yes |
| Clustered keys | 512 / 512 | Yes, fallback |
| Unstructured keys | 512 / 512 | Yes, fallback |

All four 20-step toy trajectories matched every token and the complete KV
state. These examples are not acceptance-rate estimates for pretrained models.
No claim of uniformly sublinear time is made: fixed-size flat summaries still
require a linear metadata scan. The Python oracle also has deliberately simple
scheduling and is not an optimized inference implementation.

## Repository

- [Paper and proofs](docs/PAPER.md)
- [Verification report and trusted base](docs/VERIFICATION.md)
- [GH200 implementation and measurement plan](docs/GH200.md)
- [Reference and integration contract](docs/REFERENCE.md)
- [Prior work and primary sources](docs/RELATED_WORK.md)
- `src/statecut/`: exact rational reference, summaries, filter, toy decoder
- `lean/StateCut/`: bounds, rounding/argmax gates, state preservation, composition
- `cuda/`: directed-interval prototype and host/device smoke checks
- `tests/`, `results/`: executed tests and raw results

## What remains before a deployment claim

Bind one actual checkpoint and one deterministic backend. Prove its arithmetic
error bridge, compile and audit Lean, execute CUDA tests, implement the GPU
refinement/snapshot lifecycle, and measure end-to-end latency with all costs
included. Full raw scans remain possible. A certificate hit is useful only when
its total cost is lower than the work it avoids.

MIT license. No first-in-literature or unconditional speedup claim.
