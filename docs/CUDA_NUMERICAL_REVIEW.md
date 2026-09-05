# Independent review of the E24 enclosure

The revised E24 grid rounding, monotone endpoint transport, negative-score shortcut and BF16 sum guard are mathematically sound under the stated binary64 operation contract. The review found one insufficient precondition in the capture audit: a power-of-two scale can still underflow. The audit now accepts exactly the observed scale `1/8`, for which every finite BF16 query component scales exactly.

[cuda_numerical_review.json](../results/gh200/formal/cuda_numerical_review.json) records source and binary hashes, 20,117 accepted grid cases across all 2,047 finite binary64 exponent fields, 238 intended scaling-overflow rejections, five invalid-input rejections, 45 difficult exponential-score cases on each of host and device, and 2,916 exact partial additions in accepted sum-guard cases. These are implementation tests alongside the following mathematical argument. They are not formal verification of CUDA or PyTorch.

Reproduce the probes after building the current host and device arithmetic binaries:

```bash
.venv/bin/python results/gh200/formal/review_cuda_numerics.py
```

The probe invokes `rne_e24_grid` through raw binary64 bits to include subnormals without decimal parser limitations. Its grid oracle uses independent exact rational division, remainder comparison and parity. It selects scores adjacent to logarithms of E24 rounding midpoints, then checks weights using the repository's exact rational exponential oracle. Decimal arithmetic selects test inputs; it never decides an expected acceptance result.

## Exact lattice rounding

Let `x` be the nonnegative finite binary64 input to `rne_e24_grid`. Multiplication by `2^24` is exact whenever its finite result exists: scaling upward cannot lose subnormal bits. Detected scaling overflow rejects. Put `y = 2^24 x` and `k = floor(y)`.

If `y < 1`, then `k = 0`, so computing `y-k` is exact. If `1 <= y < 2^53`, both `y` and `k` are binary64 values and `k <= y < k+1 <= 2k`; their subtraction is exact. The division `k/2`, its floor and multiplication by two are exact, so `k-2 floor(k/2)` computes parity exactly. The selected integer is representable, including a possible result of `2^53`.

If `y >= 2^53`, binary64 spacing is at least two. Thus `y` is already an even integer. The computed fraction and parity are zero and the integer is unchanged. This also covers the large exponential values near the supported score-domain maximum.

Finally, multiplying the selected integer by `2^-24` is exact. A nonzero result is at least `2^-24`, so it cannot underflow; its exponent is smaller than that of the already finite scaled value. The function therefore returns precisely `2^-24 RNE(2^24 x)` for every accepted input, including midpoint ties. At large scores the true E24 weight need not itself be binary64-representable; soundness requires only the enclosing endpoint values to be representable.

## Transporting exponential endpoints

Nearest-integer rounding with ties to even is monotone. Therefore, if `a <= exp(z) <= b`, applying the E24 lattice quantizer to both endpoints preserves enclosure. For a score interval `[lo,hi]`, real exponential monotonicity allows the lower bound from `exp_real(lo)` and the upper bound from `exp_real(hi)` to be transported separately. This uses the existing conditional real-exponential enclosure contract in `interval.cuh`; this review does not establish a proof of its compiled instructions.

The existing compiled Lean theorem `StateCut.monotone_weight_bounds` is exactly the generic endpoint-transport rule. Its monotonicity premise is explicit. A new abstract theorem would not verify the concrete C++ grid implementation and is unnecessary.

For `z <= -25`, positivity and `e > 2` give `exp(z) <= e^-25 < 2^-25`. Its E24 value is exactly zero. The shortcut represents this already-quantized value; it does not claim that the real exponential is zero. If only the lower score endpoint uses the shortcut, zero remains a valid lower E24 bound. The change supports arbitrarily negative finite score endpoints, while positive scores above the exponential routine's domain still reject. The whole attention gate continues to require strictly positive total mass.

## Exact BF16 first moments

A finite nonzero BF16 value has the form `sign * coefficient * 2^g`, with `coefficient <= 255` and `g = max(exponent_field-134, -133)`. Let `gmin` and `gmax` be the extrema among the observed nonzero values. Every input is an integer multiple of `2^gmin`, with absolute integer coefficient at most `255 * 2^(gmax-gmin)`.

The guard requires `n * 255 * 2^(gmax-gmin) <= 2^53`. Consequently, every partial sum of any subset of at most `n` inputs has an integer coefficient of magnitude at most `2^53`. It is exactly representable in binary64. Here `gmin >= -133` prevents underflow, and `gmin <= 120` together with the coefficient bound prevents overflow. Induction over any ordinary addition reduction tree proves exact summation, including cancellation. The argument assumes that the reduction includes each intended input once and uses binary64 addition; those PyTorch implementation obligations are not formally proved.

The call site supplies the actual row count as a Python integer, so the guard's integer shift and multiplication cannot overflow. Its exponent span is taken over all captured values, which is conservative for the individual column reductions. All-zero inputs are exact; nonfinite BF16 encodings reject. The probe also checks that `[1, 2^-133, -1]`, whose ordinary ordered binary64 sum can lose the small term, fails the guard.

The subsequent floating division used to generate the uniform-mean proposal need not be exact. A proposal is only a candidate; the residual gate must independently certify its rounding cell.

## Query-scaling correction and boundary

The original power-of-two-only precondition admitted `2^-133 * 2^-1000`, whose exact value is `2^-1133` but whose binary64 result is zero. This refutes the general claim that any power-of-two multiplier preserves a finite BF16 input exactly. It does not affect the captured `d=64` workloads, whose scale is `1/8`.

The revised audit explicitly requires `scale == 0.125`. Scaling a nonzero finite BF16 value then produces a binary64 number of magnitude between `2^-136` and `2^125`, with at most eight significant binary digits. The operation is exact and cannot underflow or overflow. More general scales should use a proved representability guard or outward query intervals.

No numerical backend bridge follows from this review. The CUDA gate still certifies the named E24 profile from trusted summaries; matching a captured SDPA proposal does not establish equality to a deployed backend on other inputs, nor an accelerated model transition.
