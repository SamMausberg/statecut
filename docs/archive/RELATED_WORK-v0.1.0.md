# Prior work and primary sources

Checked 2026-09-05. This is a focused related-work review, not an exhaustive
novelty or patent search. StateCut makes no first-in-literature claim.

## Distinctions that matter

**IO-aware exact attention.** FlashAttention reduces memory transfers with
attention tiling and online normalization [R1]. “Exact attention” distinguishes
its mathematical computation from approximate attention; it does not establish
bitwise equivalence to every other floating-point reduction [R5]. StateCut seeks
to skip reading some raw KV blocks by certifying a consumer result. It does not
replace FlashAttention's throughput engineering or supply a proven arithmetic
bridge to it.

**Speculative decoding.** Speculative decoding uses a draft and target
verification to preserve the target distribution [R2]. That distributional
statement is different from pathwise equality of a specified greedy trajectory
and every persistent state bit. StateCut addresses attention-consumer checks and
state boundaries, not a substitute proof of stochastic rejection sampling.

**Runtime-certified attention.** Calver's bounded-error quantized attention
paper gives runtime local error certificates relative to a specified attention
reference [R8]. The paper's scope distinction between a local error budget and
full downstream equivalence is important here. A local nonzero attention error
bound alone is not the StateCut activation-equality theorem.

**KV-quantization witnesses.** WitCert develops sound runtime risk observability
and gating for KV-cache quantization, including a formally supported witness
construction [R9]. This is relevant prior art for runtime attention certification.
A risk/TV bound is not automatically a proof of a particular greedy argmax or
future KV-state equality. The actual numerical and probabilistic assumptions
must be compared rather than treating all “certified” methods as equivalent.

The contribution developed in this repository is the explicit composition of
block enclosures, exact materialized-activation cuts, a weaker state-free final
argmax cut, immutable append state, and a whole-transition correctness contract.
Its components draw on standard interval reasoning and deterministic state
simulation. Whether this composition is novel requires a broader search.

## Sources

**[R1]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré.
*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
arXiv:2205.14135, 2022.
<https://arxiv.org/abs/2205.14135>

**[R2]** Yaniv Leviathan, Matan Kalman, Yossi Matias.
*Fast Inference from Transformers via Speculative Decoding*.
arXiv:2211.17192, 2022.
<https://arxiv.org/abs/2211.17192>

**[R3]** NVIDIA. *Grace Performance Tuning Guide*.
Official hardware/memory/NUMA documentation.
<https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html>

**[R4]** NVIDIA. *CUDA Math API: Double Precision Intrinsics*.
Official directed-rounding instruction semantics.
<https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html>

**[R5]** PyTorch. *Numerical accuracy*.
Official discussion of nonassociativity and numerical differences across
implementations, platforms, and batched versus sliced computations.
<https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html>

**[R6]** NVIDIA. *CUDA Math API: Bfloat16 Precision Conversion and Data Movement*.
<https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html>

**[R7]** Lean. *Installation* and *The Lean Language Reference: Lake*.
The repository pins an older reproducible toolchain rather than claiming to
use the latest release.
<https://lean-lang.org/install/manual/>
<https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/>

**[R8]** Dean Calver. *Runtime-Certified Bounded-Error Quantized Attention*.
arXiv:2605.20868, 2026.
<https://arxiv.org/abs/2605.20868>
<https://arxiv.org/html/2605.20868>

**[R9]** *WitCert: Sound Runtime Risk Observability and Gating for KV-Cache
Quantization*. arXiv:2607.28699, 2026. Author and revision metadata are available
at the primary paper record.
<https://arxiv.org/abs/2607.28699>
<https://arxiv.org/html/2607.28699>

## Dependency provenance

Lean 4.19.0 and mathlib's matching v4.19.0 commit:
`c44e0c8ee63ca166450922a373c7409c5d26b00b`.
The tag-to-commit mapping was read from the official mathlib GitHub repository.
No copied mathlib or Lean binaries are bundled in this archive. They retain
their own upstream licenses; the repository's original source is MIT licensed.
