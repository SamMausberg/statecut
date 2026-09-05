# Reference semantics and scope

## What “exact” means

Exactness is equality to a specified deterministic reference transition, not small error, unchanged perplexity, or agreement on a test set. The transition returns both a next-token decision and all persistent decoding state. Processing token `x[t]` stores the keys and values **of x[t]**, then predicts `x[t+1]`.

A real-valued transformer and a particular floating-point implementation are different references. A proof about real softmax does not automatically certify the output bits of FlashAttention, cuDNN, SDPA, vLLM, or SGLang. Even mathematically equivalent floating-point algorithms can disagree [R4].

## Implemented CPU profile: RATIONAL_BF16_V1

This profile is an explicitly defined numerical research reference. It is not an assertion about an existing pretrained checkpoint's original runtime.

* All inputs and frozen weights are exact rationals. Values materialized as BF16 use exact nearest-even rounding of the rational, not an intermediate binary32 conversion.
* BF16 infinities and NaNs are rejected. Zero is canonical positive zero. The reference does not distinguish signed zeros.
* Each linear layer computes its rational dot products exactly and then rounds each result to BF16. Residual additions are likewise exactly formed and BF16-rounded.
* Attention scores are exact rational dot products of the supplied query and key. The fixture's query scale is exactly 1/2, corresponding to head dimension four. No hidden default scaling is applied inside attention.
* The exponential primitive is `E_p(s) = 2^-p RNE(2^p exp(s))`, with `p=24`. This is rounding on a fixed absolute dyadic grid, NOT an FP32 exponential. It is monotone and nonnegative. It can round to zero for sufficiently negative scores.
* Weights are evaluated without a max-subtraction shift. Such a shift would change this fixed-grid reference. A zero total weight is rejected. Do not add numerical stabilization silently.
* Attention forms `sum E_24(score)*value / sum E_24(score)` in exact rational arithmetic, then rounds the materialized vector to BF16.
* RMSNorm uses epsilon 1/256. Each `gamma*x/sqrt(mean(x*x)+epsilon)` is correctly BF16-rounded by exact rational squared-midpoint comparisons. This avoids nontermination at an exact rounding midpoint.
* The fixture's SiLU-like gate computes `RNE_BF16(x/(1+E_24(-x)))`. Its formula is part of this profile; it is not libdevice SiLU.
* Greedy decoding selects the smallest vocabulary index among equal maximal materialized logits. There is no stochastic sampling.

The bounded exp evaluator supports scores through ±2048 and budgets initial
precision according to positive score magnitude. It may raise a resource-limit error. It never substitutes an approximate value when rounding is unresolved. The correctness claim is partial correctness of successful executions; a resource failure is not a valid prediction. The mathematical filter's termination theorem assumes its exact-reference primitive/fallback calls terminate. The Python fixture uses finite, modest-sized inputs for which the recorded tests completed.

## Inherited real-softmax CUDA draft

`cuda/summary_attention.cu` encloses **mathematical real softmax** using binary64 directed operations and a Taylor-series exponential enclosure. Its numerical profile is not the CPU profile above. The host-emulation tests check the shared interval arithmetic; they do not establish CPU/CUDA output identity.

CUDA score endpoints outside [-64,64], nonfinite metadata, invalid moments, and nonpositive denominator lower bounds are rejected by returning invalid intervals. The smoke test's all-zero keys and all-1/2 values give the same exact answer in both mathematical and CPU profiles, but this special coincidence does not equate the profiles generally.

A CUDA BF16 certificate for a deployed backend requires a sound bound `epsilon[j]` on the difference between its reference pre-round attention output and mathematical attention. The code expands the enclosure by that bound. `epsilon=0` is valid for the correctly-rounded mathematical reference, **not automatically for any deployed attention kernel**. No supported pretrained backend or proven FlashAttention error bridge is provided in this repository.

## New E24 CUDA local-gate draft

`cuda/residual_frontier.cu` is different from the inherited real-softmax code.
It encloses E24 weights by monotonically rounding outward real-exponential
endpoints onto the E24 lattice, then evaluates count/sum/range residual bounds
at exact BF16 cell endpoints. Scores at most -25 have exactly zero E24 weight;
positive endpoints above 64 still fail closed. It supports dimensions 1 through 128 and requires trusted
outward summaries and a positive mass bound. It returns local head flags, not
full token or state certificates. The current code is compiled and tested on
the GH200; see `CUDA_NUMERICAL_REVIEW.md` for its arithmetic proof and tests.
No deployed-backend error bridge is supplied by this gate.

The Python `TreeModel` preserves the inherited E24 target while adding a forest
and write-frontier strategy. Its equivalence to the original dense decoder is
covered by regression tests. See `PAPER.md` for exact algorithm/proof scope.

## What a production reference identifier must bind

At minimum: model and adapter weight hashes; tokenizer and logit-processor configuration; attention and projection kernel implementations; reduction schedule or allowed arithmetic semantics; operand/accumulator types; TF32 and reduced-precision settings; rounding/flush-to-zero rules; RoPE parameters and positions; masks and visible prefix; KV layout, quantization and dtype; deterministic argmax tie rule; persistent state schema; and any routing or random state. Batch shape can affect dispatch and must be bound when relevant.

A cache keyed only by shape is invalid. Model weights must not be silently reused after modification. A measured error tolerance is not an admissible bridge theorem.

## Architectural scope

The state-cut theorem applies to a deterministic graph whose live state has been identified. The local-cut specialization fits a conventional causal decoder with K/V formed before that layer's attention and a state-free final suffix. GQA/MQA requires one append per KV head, not one per query head. Summaries must be built over the exact rotated keys and exact visible tokens.

Sliding windows, prefix masks and speculative branches require separate correctly scoped summaries. A summary containing future or masked tokens cannot be used as though those tokens were visible. Hybrid recurrent models, persistent final hidden-state caches, cross-layer attention, nondeterministic MoE routing, and stochastic decoding need a new liveness/state analysis. They are not covered by the toy integration merely because they are called LLMs.

Primary sources are indexed in [REFERENCES.md](REFERENCES.md).
