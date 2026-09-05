# StateCut: exact-state certificates for adaptive attention evaluation

Research note, version 0.1.0, 5 September 2026.

## Abstract

We give a deterministic filtering algorithm that can decide a decoder's next token without rereading every raw key/value entry while preserving its exact future decoding state. The essential acceptance predicate is not approximate attention accuracy and not merely equality of the current token. It is equality of the live state-producing computation, established at materialization cuts; a weaker argmax certificate is allowed only after the last persistent-state write. Immutable block summaries enclose attention. Unresolved blocks are read and evaluated exactly, with an exact-reference fallback. We prove the enclosure, acceptance, state-preservation and finite-refinement results below. A nontrivial family needs no raw cache reads after preprocessing, while general inputs can force dense work for this filter. This is a conditional exactness construction, not a universal sublinear-time theorem or a measured GH200 speedup.

The accompanying Python implementation has an explicit finite numerical reference and a small frozen decoder. CUDA source implements mathematical-softmax enclosures. Lean source covers the core algebra and transition argument, but was not compiled in the authoring environment. These boundaries are material parts of the result.

## 1. Problem and exactness contract

Let `S` be all persistent state, including token prefix, layer/head K/V, positions, visibility, and any other information read by later steps. Fix a deterministic reference

$$
 F_R:S\to\mathcal V\times S.
$$

Model parameters and reference numerical semantics are fixed. A successful exact verifier must return **both fields** of `F_R(S)`. Its auxiliary metadata can differ from the reference representation, provided it describes the same state and cannot affect the reference decision except through sound certificates.

The operational task processes the current input token, forms that token's K/V, and returns the decision for the next token. More explicitly, the implementation exposes `step_R(C,x)=(y,C')`, where `C` is the committed cache/prefix and `x` is the pending input. To obtain the autonomous transition above, take `S=(C,x)` and `F_R(C,x)=(y,(C',y))`. The predicted token is not inserted into K/V until it is processed on the next invocation. The tests compare `y` and every component of `C'`, which together compare this complete autonomous state. We restrict the main result to greedy decoding with smallest-index tie breaking. Every theorem is relative to `R`. For actual hardware, proving a bound on mathematical attention is not enough unless that bound also encloses `R`'s pre-round value. Section 8 states this missing bridge precisely.

### Why a current-token certificate alone is insufficient

Consider the deterministic scalar-state transition

$$
 F(s)=(\mathbf 1_{s<0},s-1/2).
$$

Starting from `s=1/4` and `s'=3/4`, both current decisions are zero. Their next states are `-1/4` and `1/4`, and their subsequent decisions differ. Thus

$$
 \pi_1F(s)=\pi_1F(s')\not\Rightarrow\pi_2F(s)=\pi_2F(s')
$$

and does not imply equal future decoding. This counterexample disproves the logical shortcut independently of any transformer-specific implementation. In a decoder, approximate intermediate states can change later-layer K/V even when the current argmax is unchanged.

## 2. Persistent-state cuts

A **live-state cut** is a collection of values through which every not-yet-completed persistent-state write depends. Recovering those cut values exactly permits the downstream deterministic state-producing graph to be evaluated exactly. It is unnecessary to recover every transient activation, but it is necessary to cover every live state dependency.

For a conventional decoder layer, a sufficient local scheme is:

1. Enter the layer with exactly the reference hidden vector.
2. Execute normalization and Q/K/V projections using the pinned reference operations.
3. Stage, but do not yet publish, the new exact K/V.
4. Enclose attention and recover its **reference materialized output vector exactly**.
5. Run the reference output projection, residual and feed-forward suffix.

At the next layer, the input is again exact. This prevents approximation error from propagating into later-layer K/V. The scheme is sufficient, not a claim that this is the minimal cut for every graph.

In the final layer, if all persistent K/V writes have already been exactly determined and the remaining suffix has no state writes, it is enough to enclose that suffix's logits and certify the token. An approximate final hidden vector must not be written into some additional cache merely because ordinary transformers usually do not need it.

## 3. Attention and immutable block witnesses

For one visible query/head, write

$$
 s_i=q^\top k_i,\quad w_i=E(s_i),\quad
 D=\sum_iw_i,\quad N_j=\sum_iw_i v_{ij},\quad a_j=N_j/D.
$$

Any scaling is included in `q`. The mathematical specialization uses `E=exp`; the CPU reference uses the explicit monotone dyadic function `E_24`. Assume the actual reference denominator `D>0`. Retain all original exact K/V for refinement and fallback.

Partition visible entries into disjoint blocks. A block `b` stores count `n_b`, exact coordinate extrema

$$
 k^-_{br}\le k_{ir}\le k^+_{br},
$$

and signed value moments

$$
 P_{bj}=\sum_{i\in b}\max(v_{ij},0),\qquad
 M_{bj}=\sum_{i\in b}\min(v_{ij},0).
$$

Thus `P>=0`, `M<=0`. The CPU builder computes these in exact rational arithmetic. A GPU builder must instead store sound directed enclosures of the moments or prove its sums exact. An ordinary floating-point sum is not automatically an exact moment.

The summaries are constructed once from actual cache contents. New entries update a tail's extrema and signed sums in `O(d_k+d_v)` arithmetic; full blocks are immutable. Summary validation at ingestion is real work, not a free proof. This saves query-path raw reads, not raw-cache storage. The Python persistent containers also copy some tuple handles on append; they are not an optimized serving allocator.

### Lemma 1: score enclosure

Define

$$
 \ell_b=\sum_r q_r
 \begin{cases}k^-_{br}&q_r\ge0\\k^+_{br}&q_r<0,\end{cases}
 \qquad
 u_b=\sum_r q_r
 \begin{cases}k^+_{br}&q_r\ge0\\k^-_{br}&q_r<0.\end{cases}
$$

Then `ell_b <= s_i <= u_b` for every `i` in the block.

**Proof.** Multiplication by a nonnegative query coordinate preserves its key-coordinate inequalities; multiplication by a negative coordinate reverses them. Sum over coordinates. No norm-distribution or independence hypothesis is used. The Lean statement is `dot_box`.

Let `alpha_b=E(ell_b)` and `beta_b=E(u_b)`. Monotonicity gives

$$
 0\le\alpha_b\le w_i\le\beta_b.
$$

For mathematical softmax this follows from monotonicity and positivity of exp. For `E_p`, nearest-even rounding on an ordered uniform grid is monotone, so composition with exp is monotone and nonnegative. Finite-grid underflow can make the lower endpoint zero; division is not permitted until the aggregate denominator lower bound is strictly positive.

### Lemma 2: block mass and numerator enclosure

$$
 n_b\alpha_b\le D_b\le n_b\beta_b,
$$

$$
 \alpha_bP_{bj}+\beta_bM_{bj}
 \le N_{bj}\le
 \beta_bP_{bj}+\alpha_bM_{bj}.
$$

**Proof.** Sum the weight bounds for mass. If `v>=0`, `alpha*v <= w*v <= beta*v`. If `v<=0`, `beta*v <= w*v <= alpha*v`. Decompose each value into `max(v,0)+min(v,0)` and sum. Using the same endpoint on both the positive and negative terms is unsound. Lean statements: `signed_term_bounds`, `signed_block_bounds`, `mass_bounds`.

When signed moments themselves have enclosures, replace `P` and `M` by their relevant directed endpoints. All nonnegative-weight multiplications and additions must round outward. The shared CUDA/host header implements these operations, but the device refinement to IEEE/PTX remains outside the Lean model.

### Lemma 3: normalization

Summing all block enclosures gives `D in [D_L,D_U]` and `N_j in [N_Lj,N_Uj]`. When `D_L>0`, let

$$
 L_j=\min\{N_{Lj}/D_L,N_{Lj}/D_U,N_{Uj}/D_L,N_{Uj}/D_U\},
$$

$$
 U_j=\max\{N_{Lj}/D_L,N_{Lj}/D_U,N_{Uj}/D_L,N_{Uj}/D_U\}.
$$

Then `a_j in [L_j,U_j]`.

**Proof.** For fixed positive `d`, `n/d` is increasing in `n`, so extremizers in the rectangular superset occur at a numerator endpoint. For either fixed numerator endpoint the derivative with respect to `d` has a constant sign, possibly zero, so its extrema occur at denominator endpoints. The true `(N,D)` lies in the rectangle, even though its two components are correlated. Ignoring correlation can loosen a bound but cannot invalidate it. In particular, simply dividing both lower endpoints and both upper endpoints is incorrect for negative numerators.

The Lean theorem `ratio_from_cross_checks` proves the equivalent division-free certificate: check `L*D_L <= N_L`, `L*D_U <= N_L`, `N_U <= U*D_L`, `N_U <= U*D_U`. A sign split on `L` and `U` gives `L*D <= N <= U*D`, then division by positive `D` proves the claim.

## 4. Exact acceptance predicates

### Theorem 1: a rounding-cell certificate recovers exact state-producing values

Let `Q` be the reference monotone materialization map, for example BF16 nearest-even on the specified finite domain. Suppose the **reference pre-round value** `y_j` lies in `[L_j,U_j]`. If

$$
 Q(L_j)=Q(U_j)=c_j
$$

for every live coordinate, then `Q(y_j)=c_j` for every coordinate.

**Proof.** Monotonicity gives `Q(L_j) <= Q(y_j) <= Q(U_j)`. Equal endpoints squeeze the middle value to `c_j`. Apply this separately to every coordinate. No probabilistic error allowance remains. Lean: `rounding_cut`, `vector_rounding_cut`.

For nearest-even, the exact midpoint belongs to one neighboring cell, determined by parity. The Python implementation rounds rational endpoints exactly and accepts endpoint equality, thereby handling ties. The CUDA proposal converts an interval midpoint to a candidate BF16 value, then independently checks strict membership of both endpoints between the candidate's exact binary64-representable BF16 midpoints. Double rounding in the proposal cannot create a false acceptance, because the proposal is not trusted. The CUDA path rejects midpoint boundary cases and the extreme finite overflow cell, and canonicalizes zero in its declared numerical profile.

The map must actually be the materialization used by the reference graph. Inserting a new BF16 rounding point into an FP32-fused reference computation changes the reference. Such an insertion is not licensed by this theorem.

### Theorem 2: terminal tie-aware argmax certificate

Let the true reference logits satisfy `z_j in [l_j,u_j]`. Candidate `c` is the smallest-index reference maximizer if

$$
 l_c>u_j\quad(j<c),\qquad l_c\ge u_j\quad(j>c).
$$

**Proof.** For every smaller index, `z_c >= l_c > u_j >= z_j`, ruling out even a tie. Every larger index satisfies `z_c >= l_c >= u_j >= z_j`. Thus `c` maximizes the logits and no smaller index ties it. This is exactly the specified tie rule. Lean: `argmax_certificate`, `greedy_unique`, `certified_greedy_equals_reference`.

The implementation selects the largest lower endpoint, breaking lower-endpoint ties toward smaller indices, then applies the complete check against **every other vocabulary entry**. This does not avoid a vocabulary scan. A vocabulary hierarchy could be an additional optimization, but is not implemented.

### Sound terminal interval propagation

The finite fixture propagates boxes through the remaining exact-reference operators. Linear maps sum interval-scaled weights, then apply endpoint rounding. Residuals use interval addition and rounding. For RMSNorm, squared intervals enclose each square, the mean plus positive epsilon has a positive interval, sqrt is monotone, and reciprocal division encloses every normalized coordinate even though numerator and denominator are dependent. BF16 endpoint rounding is monotone. The SiLU-like function is **not assumed monotone**: the code encloses its monotone sigmoid factor and uses interval multiplication with the input. Products and subsequent linear maps are enclosed similarly. Composition preserves enclosure by induction over this operation graph.

For actual pretrained suffixes, every reference operation needs a sound interval extension. Calibration data and unverified local Lipschitz estimates cannot replace this obligation.

## 5. Algorithm and refinement

For a state-producing attention cut:

```text
start from exact reference hidden input and cache snapshot
compute reference Q/K/V; stage the current token's exact K/V
build a contribution enclosure for every visible summary block
repeat:
    aggregate and normalize, only with a positive denominator lower bound
    if every materialized coordinate has a valid rounding-cell certificate:
        recover the exact reference vector and execute the ordinary suffix
        finish this layer
    if the refinement budget is exhausted:
        evaluate every unopened raw block by the exact reference
        combine with already opened exact contributions
        execute the exact reference materialization and suffix
        finish this layer
    open one unresolved block and replace its enclosure by its exact contribution
```

For a state-free terminal suffix, replace the rounding-vector consumer by interval suffix propagation plus Theorem 2. Its fallback evaluates the exact terminal suffix. Only after the entire step succeeds are staged caches published. An error leaves the previous immutable snapshot unchanged.

A scheduler may use draft attention, retrieval, approximate scores or arbitrary priorities to choose the next block. The choice need not be correct: certificates, not the scheduler, authorize acceptance. The current CPU heuristic prioritizes a sum of numerator widths plus a denominator-width/value-scale term. It does not use randomness or inspect unopened raw entries.

### Theorem 3: soundness is preserved by each refinement

**Proof.** An exact contribution is a singleton interval containing the same block's true numerator and denominator. Replacing a containing block enclosure by this singleton preserves all summed containments. The original blocks still form a disjoint, complete visible partition. Therefore Lemmas 1-3 and Theorems 1-2 remain applicable after any sequence of chosen refinements. In exact interval arithmetic the replacement can only narrow the contribution. In directed floating arithmetic, use a fixed reduction tree or a sound rebuilt enclosure; changing reduction order must not be assumed bitwise neutral for a floating reference.

### Theorem 4: finite refinement and no duplicate raw-block reads

Assume a finite partition with `B` blocks, a terminating exact-reference block evaluator, and a positive reference denominator. Each refinement removes one block from a finite unresolved set and never reinserts it. Thus after at most `B` openings, all contributions are exact. The exact-reference fallback is then available. Previously opened contributions are reused, so each block is read at most once **in this CPU reference algorithm**.

This reuse relies on the CPU reference's exact rational reduction: its contribution combination order has no rounding effect. A FlashAttention fallback with a different required online reduction schedule cannot automatically reuse these contributions. It may have to reread the full cache. Production performance accounting must charge that extra read unless a backend-specific refinement theorem proves reuse legal.

If a resource failure, invalid metadata, unsupported numeric value or unavailable fallback occurs, the implementation must return an error, not a draft token. The theorem does not turn resource failure into successful decoding.

## 6. Exact future-state theorem

### Theorem 5: exact state cuts plus a terminal certificate imply the entire reference transition

Assume: (a) initial persistent K/V and state equal the reference; (b) each staged new K/V is computed from an exact input using the reference operations; (c) every nonterminal approximation is replaced only after an exact cut certificate or exact fallback; (d) a terminal argmax shortcut is used only after all persistent state writes have been exactly determined; (e) the step publishes its staged state atomically after success. Then the returned token and complete committed state equal `F_R(S)`.

**Proof.** Induct over layers. The input to the first layer is exact. Equal inputs and persistent caches give equal reference normalization, Q/K/V and staged new K/V. The local acceptance predicate recovers the exact reference materialized attention vector by Theorem 1, or the fallback evaluates it directly. The deterministic suffix therefore produces the exact next-layer input. This establishes the induction for every state-producing layer. At the final layer, the same reasoning already establishes its new K/V. Either execute the exact suffix or use Theorem 2 to recover exactly its reference token while making no additional state writes. All staged state components equal the reference components; atomic publication commits precisely that state. The token also agrees. QED.

The more general state-frontier version only requires equality of all values on which future state depends plus equality of the terminal decision; it need not materialize every intermediate tensor. Lean statements `exact_cut_suffix`, `terminal_cut`, `state_frontier` capture the corresponding substitution steps. `checked_cut_transition` composes coordinatewise interval/rounding checks with whole-suffix equality, and `summary_to_exact_cut` connects signed summary bounds to the rounding result.

### Corollary: all future greedy decisions and states agree

For every number `t` of successful steps from the same initial state, the filtered and reference traces and final states are identical.

**Proof.** Theorem 5 establishes transition equality. Induct over steps: equal state yields equal next token and equal next state. The next invocation therefore satisfies the same hypotheses. No unbounded error debt accumulates. Lean's `filteredStep_eq` and `all_future_equal` formalize this consequence, with a clearly explicit whole-transition soundness premise.

The Lean transition theorem does not prove the Python program satisfies its premises. Nor does a theorem whose premise already asserts whole-transition equality establish a missing numerical enclosure. The algebraic/cut proofs and the concrete implementation evidence must be assessed separately.

## 7. How this avoids a full raw KV scan, and what it does not prove

### Proposition: a genuine zero-raw-read family

Suppose all keys within every block are equal. For each block the lower and upper scores coincide, so its weight bounds coincide for the finite reference. Count and signed moments then give exact block mass and exact numerator. All attention coordinates are exact from metadata alone. No raw K/V query read is required after preprocessing, even if values within the blocks differ and different blocks have different keys. A single block with one common key reduces exactly to its mean value. The scalar cancellation is formalized as `constant_weight_attention`.

This is not a claim that a large pretrained model commonly produces identical-key blocks. The repository also tests a nonconstant-key, dominant-block case: one block is opened and the other blocks are certified from summaries.

### Restricted lower bound, not an overclaim

In a raw-read-only query model with no preprocessed information about values, use zero keys and a nonempty context, so attention is the value average. Along an all-zero observation trace, suppose an algorithm leaves position `i` unread. Two value arrays that agree on all read positions but put `+1` or `-1` at `i` have opposite-sign attention and different greedy decisions for logits `(attention,0)`. The algorithm has the same transcript on both and cannot be correct on both. Therefore a universally correct algorithm in this restricted model must read every position in some execution.

**This is not a lower bound against arbitrary summaries or preprocessing.** A stored sum defeats this particular equal-key example. Lean's `indistinguishable_observations` proves only the stated indistinguishability principle.

### Finite moments alone do not make exponential attention exact

For any degree `m`, take scalar keys `k_i=i`, `0<=i<=m+1`, and values

$$
 v_i=(-1)^{m+1-i}\binom{m+1}{i}.
$$

Their polynomial moments `sum i^r v_i` vanish for all `0<=r<=m`, by the `(m+1)`st finite difference of a degree-at-most-`m` polynomial. Yet

$$
 \sum_i e^{qi}v_i=(e^q-1)^{m+1},
$$

by the binomial theorem, which is nonzero for `q!=0`. Negating the values preserves those zero moments and reverses the numerator, with the same positive denominator. Thus those fixed-degree signed moments alone do not identify exact attention for all queries. This proposition concerns that specified summary family, not all possible data structures. Remainder bounds, refinement or an exactness-enabling consumer are essential.

### Complexity

With context length `n`, block size `b`, and key/value dimensions `d_k,d_v`, the flat summary pass uses

$$
 O((n/b)(d_k+d_v))
$$

metadata arithmetic and reads. Opening `r` full blocks adds `O(rb(d_k+d_v))` raw work. Summary construction and raw storage remain linear in context length. With fixed block size the flat metadata pass is still asymptotically linear in `n`; the claim is reduced **raw KV traffic**, not general sublinear complexity. The Python implementation rebuilds aggregates and priorities between refinements and can add quadratic-in-block-count scheduling work; it is an oracle, not a latency-optimized implementation. A reduction tree could improve this without altering the proof.

## 8. Floating-point reference bridge

Let `[L_j,U_j]` enclose mathematical attention. Let `r_j` denote a pinned reference's actual pre-round attention value. A sufficient bridge is a proved input-dependent bound

$$
 |r_j-a_j|\le\epsilon_j.
$$

Then `[L_j-epsilon_j,U_j+epsilon_j]` encloses `r_j`, and Theorem 1 applies to its reference rounding map. The bridge must account for score products/sums, scale multiplication, exp implementation, max subtraction, softmax reduction order, numerator accumulation, division, cast points, subnormals, overflow and fusion. It may instead be obtained by sound interval interpretation of the pinned reference instruction graph.

A norm estimate learned from test inputs, “one ULP should be enough”, comparing a few outputs, or treating FP64 as exact is not a proof. This repository does **not** provide the requisite bridge for FlashAttention or PyTorch. Without it, their fast path must remain disabled. The fallback can preserve their exact decisions, but that alone establishes no acceleration.

### Constructive real-exp enclosure used here

For rational `x`, reduce `t=|x|/2^s<=1/2`. Let `S_m=sum_{k=0}^m t^k/k!`. The first omitted term is `a=t^(m+1)/(m+1)!`. Every subsequent ratio is at most `t/(m+2)<1`, so

$$
 S_m\le e^t\le S_m+\frac{a}{1-t/(m+2)}.
$$

Every term is nonnegative. Repeated squaring preserves containment of these nonnegative intervals, and reciprocal reverses positive endpoints for negative `x`. Outward dyadic rounding preserves containment. This proves the Python real-exp interval formula. For a fixed-grid exp reference, endpoint nearest-even agreement certifies the exact rounded exponential by Theorem 1; unresolved rounding raises an error.

The CUDA implementation uses `s=7`, `m=20` and `|x|<=64`. It performs the same algebra with directed binary64 operations, rejects invalid intermediate results and uses no libm approximation for exp. Its real-analysis formula is justified above, but its complete compilation to GPU instructions is not Lean-verified.

## 9. GH200 performance hypothesis and falsifiable acceptance criteria

A summary method can accelerate only when the raw work avoided exceeds summary reads, bound evaluation, refinement, launch, synchronization and state-commit costs. For a particular query define baseline time `T_D` and measured components

$$
 T_C=T_{summary}+T_{check}+T_{opened}+T_{fallback-extra}+T_{launch}+T_{state}.
$$

The algorithm accelerates that query **if and only if** its actual `T_C<T_D`. Algebraic correctness does not establish this inequality. Average speedup requires `E[T_C]<E[T_D]`, with the same workload and matched reference semantics. A fallback-heavy filter can be slower than dense attention.

GH200's GPU/CPU memory tiers and coherent C2C interconnect offer placement options [R3], but coherency does not make CPU memory HBM-speed. Begin with retained raw K/V and summaries in HBM. Do not offload the fallback copy until measured capacity needs justify the bandwidth cost. GQA must share raw reads and summaries where legal; measure actual memory transactions instead of multiplying per-query-head logical reads.

Rounded internal cuts can be much harder to certify than an argmax margin, because every state-producing coordinate must agree, including near-zero outputs and values close to a rounding midpoint. The tests include full-scan cases. High-dimensional pretrained tensors may therefore erase the gain. If only the final layer succeeds, its benefit is limited by the fraction of runtime spent in that final attention. Under the artificial equal-cost, attention-only `L`-layer model, eliminating that attention completely gives at most `L/(L-1)` for `L>1`, before checker costs. Real end-to-end gains can be smaller.

There are no GH200 latency measurements in this release. Successful CUDA compilation, sanitizers, reference identity, all-state equality, and end-to-end latency measurements on pinned pretrained checkpoints are release gates, not completed results.

## 10. Evidence and limitations

The repository records CPU tests, C++ host-emulation checks, and small-model trajectories. They are reproducible implementation evidence, not proof by testing. Raw-read counters exclude metadata, model weights, summary construction, allocation and writes. The pretrained-model indicator is false. Lean compilation is explicitly false. CUDA execution is explicitly false.

The mathematical contribution is the precise state-frontier acceptance contract and its integration with adaptive attention enclosures, not a claim to have invented interval arithmetic, moment summaries, adaptive precision, attention sparsity or speculative decoding. Related work and current implementation gaps are listed in [RELATED_WORK.md](RELATED_WORK.md), [VERIFICATION.md](VERIFICATION.md) and [GH200.md](GH200.md).
