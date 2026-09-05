# Reproducing the GH200 validation

The recorded environment is Python 3.10, machine-provided ARM64 PyTorch 2.7.0,
CUDA toolkit 12.8 and a GH200 with 97,871 MiB reported device memory. Exact
package/device details are in `results/gh200/environment.json`.

## Setup and mathematical checks

```bash
bash scripts/setup_gh200.sh
STATECUT_CUDA_LIBRARY=build-gh200/libstatecut_residual_cuda.so .venv/bin/python -m pytest -q
bash scripts/check_lean.sh
.venv/bin/python scripts/audit_algorithm.py
.venv/bin/python scripts/run_v2_experiments.py --output results/gh200/algorithm/v2_experiments.json
.venv/bin/python scripts/check_host_residuals.py --binary build-gh200/statecut_device_residual --device --output results/gh200/device_arithmetic.json
.venv/bin/python scripts/check_host_residuals.py --binary build-gh200/statecut_host_residual --output results/gh200/host_arithmetic.json
.venv/bin/python scripts/check_host_intervals.py build-gh200/statecut_host_interval results/gh200/host_intervals.json
```

Setup creates a local venv with system-site-packages to retain the CUDA-enabled
ARM64 Torch build. It pins compatible NumPy, SciPy, Pillow and Transformers;
the initial image's old Pillow and a NumPy 2 wheel were incompatible with this
stack. Other repositories' environments are not modified.

## Device checks and measurements

```bash
STATECUT_CUDA_LIBRARY=build-gh200/libstatecut_residual_cuda.so compute-sanitizer --tool memcheck --error-exitcode 99 .venv/bin/python -m pytest tests/test_cuda_frontier.py -q
STATECUT_CUDA_LIBRARY=build-gh200/libstatecut_residual_cuda.so compute-sanitizer --tool racecheck --error-exitcode 99 .venv/bin/python -m pytest tests/test_cuda_frontier.py -q
.venv/bin/python scripts/bench_gh200.py --out results/gh200/microbenchmark_h8_n32768_d64.json
.venv/bin/python results/gh200/formal/review_cuda_numerics.py
```

Run measurements without concurrent GPU tests. Single-call event timing
includes host launch gaps. Graph-batch timing amortizes submission over 100
sequential fixed-input invocations whose allocations remain owned by the
benchmark. It measures primitive throughput, not token latency. E24 gate and
SDPA timings have different numerical contracts; their ratio is not a
certified model speedup. Initial tensor construction is charged separately.

## Pinned pretrained observations

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from scripts.validate_pretrained import MODELS
for model, revision in MODELS.values():
    snapshot_download(model, revision=revision, cache_dir='.cache/huggingface',
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE','README.md'])
PY
.venv/bin/python scripts/validate_pretrained.py --model smollm2-135m
.venv/bin/python scripts/validate_pretrained.py --model qwen2.5-0.5b
.venv/bin/python scripts/audit_capture_cuda.py results/gh200/pretrained_smollm2-135m.json results/gh200/pretrained_qwen2.5-0.5b.json
.venv/bin/python scripts/audit_pretrained_e24.py results/gh200/pretrained_smollm2-135m.json results/gh200/pretrained_qwen2.5-0.5b.json
```

Capture directories must be empty. For another run, pass a fresh `--output`
directory to validation and audit the resulting manifests. Model identifiers,
immutable revisions and prompts are defined in `validate_pretrained.py`;
remote custom model code is disabled. Exact input tokens and capture hashes
are stored in the two model manifests.

The workloads use three controlled prompt patterns repeated to 128 or 512
tokens and three greedy decode steps. They are not a held-out language
benchmark. The observer compares all layers' K/V bits and last-position logits
at prefill and each decode step against unchanged SDPA execution. All supported
decode calls are captured; the prefill skips remain visible in the ledger.

The CUDA audit checks an integer-grid sufficient condition for exact FP64
value sums and supports only the observed scale `1/8`. It reads all raw rows
offline. Its uniform-mean proposal uses summaries; its observed-SDPA proposal
is an oracle diagnostic. Score-domain and denominator failures are counted
separately from remaining cell failures.

The exact E24 audit samples the first two query heads at the first, middle and
last layers of the first decode step. Domain failures are explicit unsupported
results; internal arithmetic defects abort. The bounded evaluator uses
magnitude-aware precision and a score resource domain through ±2048, sufficient
for the recorded positive Qwen scores around 1,280. It applies no score shift.

## Paper and evidence

```bash
make -C paper
.venv/bin/python scripts/check_claims.py
```

The paper uses the unmodified ICML 2026 style and native TikZ/PGFPlots figures.
Generated tables bind their experiment inputs with hashes. It is a private
research draft authored by Samuel Mausberg, with no submission or acceptance
claim. Read `docs/VERIFICATION.md` for the proof-to-implementation boundary.
