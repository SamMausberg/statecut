> Historical v0.2.0 manuscript. The current manuscript, authored by Samuel Mausberg, is [ICML-style LaTeX](../paper/main.tex) with [PDF](../paper/main.pdf), TikZ figures, updated proofs and GH200/pretrained results. The uncompiled/unexecuted status statements below describe the imported version.

# StateCut v0.2: denominator-free attention predicates and exact write frontiers

Research manuscript, 5 September 2026. Extends the supplied StateCut v0.1.0.

**Evidence boundary.** The mathematical arguments below are proofs over the stated objects. The Python implementation is an executable exact-rational reference with tests. Lean theorem sources are supplied but were not compiled in this environment. The SM90 CUDA implementation is uncompiled and unexecuted here. No pretrained-checkpoint exactness or GH200 speedup is established by this release. These are different obligations, not interchangeable meanings of "verified."

## Abstract

We investigate deterministic greedy decoding in which a verifier need not reread every historical key/value row. The verifier must preserve both the current token and all future-observable state. We strengthen StateCut's numerical predicate by testing a rounding-cell boundary through the signed residual `N - tD`, rather than separately enclosing and dividing an attention numerator and denominator. Count, first moment, and value extrema give a translation-invariant residual enclosure. A persistent binary-counter forest avoids the previous compulsory linear scan of block summaries. Finally, an interval interpreter checks actual persistent writes instead of demanding that every transient attention activation be exact. A successful write-frontier check commits concrete reference-equal K/V, not approximate state or deferred correction debt.

We prove the residual enclosure, rounding implication, frontier invariants, and whole-transition/future-trace implication. A nonconstant family admits one-summary, zero-raw-read verification after preprocessing, independent of its length. In a paired synthetic experiment at 8,192 rows, both the old independent-ratio gate and a centered-but-divided gate fall back to all rows, while the direct predicate accepts from one root. These are logical-read results for a named numerical profile, not pretrained acceptance rates or GPU acceleration measurements.

## 1. What must be equal

Fix the model parameters, tokenizer/policy, numerical operator semantics, visible-prefix rule, and deterministic tie rule. A decoding invocation processes a pending token `x`:

\[
 F_R(C,x)=(y,C').
\]

Here `C` contains every state component that can influence any future invocation. In the reference prototype these are the model identity, exact per-layer K/V, prefix tokens, and position. In a deployed system they also include masks, RoPE conventions, sequence/branch identity, cache ownership and generations, recurrent state if present, and any other mutable buffers read later. The newly predicted token `y` is not in the cache until the next invocation processes it.

An exact replacement returns the same pair, or a representation with an explicitly proved equivalent denotation. Token agreement alone is insufficient. For example, `F(s)=(1[s<0],s-1/2)` gives the same current token at `s=1/4` and `s=3/4`, but the next states have different signs. Their subsequent tokens differ.

For autonomous generation use the state `(C,x)` and transition `(C,x) -> (y,(C',y))`. For externally supplied continuations, prove one-step equivalence for every legal supplied `x` and induct over that input sequence. The prototype tests both generated continuations and supplied tokens.

The scope here is greedy selection with the smallest vocabulary index winning a tie. Sampling requires a separate coupling proof and inclusion of RNG state. Equality in distribution is not pathwise token/state equality; speculative decoding addresses a related but different contract [R3].

## 2. Reference numerical profile

The inherited executable target is named `RATIONAL_BF16_V1` (abbreviated E24 below):

\[
 E_{24}(z)=2^{-24}\operatorname{RNE}_{\mathbb Z}(2^{24}e^z),\quad
 w_i=E_{24}(q^\top k_i),\quad
 A_j=\frac{\sum_iw_iv_{ij}}{\sum_iw_i}.
\]

The query includes the profile's exact scale. Inputs and all algebraic reductions are rationals. The exponential is enclosed using convergent rational bounds until its integer rounding is decided. Its implementation has explicit resource/domain guards; exhaustion rejects rather than guesses. `E24(0)=1`. E24 weights can be zero, so nonempty attention does not alone imply positive denominator. A zero reference denominator is an error, not a certificate.

BF16 materializations use exact round-to-nearest, ties-to-even; nonfinite results reject, and signed zeros are canonicalized to positive zero. Matrix products sum exactly before rounding at the specified cut. RMSNorm computes the specified square-root expression with exact rounding decisions. The gate is `BF16(x/(1+E24(-x)))`, not an unqualified claim about a library SiLU implementation. `model.py` fixes these choices.

This is **not** bitwise PyTorch SDPA, FlashAttention, tensor-core GEMM, or real-exponential softmax. E24 is not invariant under adding a common score shift: rounding the exponentials changes the ratios. The implementation never imports real-softmax shift invariance into E24. The original `summary_attention.cu` is separately a real-softmax enclosure draft; the new `residual_frontier.cu` encloses E24. They must not be substituted for one another without a bridge.

The algebra below only requires valid nonnegative weight bounds. It can support other profiles once their weight/operator contract is proved. Merely fixing the same pretrained weights does not fix floating-point decisions across different implementations [R4].

## 3. Mergeable information without a query-time raw scan

For a block or tree node with `n > 0` visible entries, store for each key coordinate its minimum and maximum. For each value coordinate store:

\[
 S=\sum_i v_i,\qquad \ell=\min_i v_i,\qquad u=\max_i v_i.
\]

The Python implementation retains the old positive and negative moments `P=sum max(v_i,0)` and `M=sum min(v_i,0)`, so `S=P+M`. Counts and moments add; extrema take min/max. Thus adjacent summaries merge without reading their raw entries. The CUDA gate uses an outward interval for `S` rather than assuming a hardware sum is exact.

For a known query, interval dot products give `q^T k_i in [L,U]` for every row. Monotonicity of E24 gives:

\[
 a=E_{24}(L)\le w_i\le E_{24}(U)=b.
\]

The same construction works when each query coordinate is an interval containing its reference value. This is essential in the write-frontier interpreter. Different heads and prefixes require distinct, correctly attributed summaries. A general bias or mask requires its own sound visible-set/score bounds. The reference code supports the complete causal prefix, not arbitrary masks by omission.

## 4. The centered residual theorem

Fix a value coordinate and a threshold `t`. Define:

\[
 D=\sum_iw_i,\quad N=\sum_iw_iv_i,\quad
 R(t)=N-tD=\sum_iw_i(v_i-t),
\]

\[
 m=(a+b)/2,\qquad h=(b-a)/2.
\]

For `ell < u`, define the count/sum/range envelope

\[
 C(t)=\frac{(nu-S)|\ell-t|+(S-n\ell)|u-t|}{u-\ell}.
\tag{1}
\]

For `ell=u`, put `C(t)=n|ell-t|`.

### Theorem 1. Residual enclosure

If `a <= w_i <= b`, `ell <= v_i <= u`, and the summary has the actual count and first moment, then

\[
 \boxed{m(S-nt)-hC(t)\le R(t)\le m(S-nt)+hC(t).}
\tag{2}
\]

**Proof.** Since `|w_i-m| <= h`,

\[
 R(t)=m(S-nt)+\sum_i(w_i-m)(v_i-t)
\]

and the last sum has absolute value at most `h sum_i |v_i-t|` by the triangle inequality. For any `v in [ell,u]`, write

\[
 v-t=\frac{u-v}{u-\ell}(\ell-t)+\frac{v-\ell}{u-\ell}(u-t).
\]

The coefficients are nonnegative. Taking absolute values and summing gives

\[
 \sum_i|v_i-t|\le
 \frac{(nu-S)|\ell-t|+(S-n\ell)|u-t|}{u-\ell}=C(t).
\]

Substitution proves (2). When `ell=u`, every value equals `ell`, and the stated constant-value expression is exact before bounding the weights. QED.

This proof uses no probabilistic assumption, smoothness, norm heuristic, independence between keys and values, or observed error tolerance. The input correlations can be arbitrary.

**Tightness scope.** For the relaxation that allows nonnegative mass at the two endpoints with total mass `n` and moment `S`, the chord envelope is attained. Choosing weights `m+h sign(v-t)` or `m-h sign(v-t)` attains the corresponding residual extreme. The endpoint masses are `(nu-S)/(u-ell)` and `(S-nell)/(u-ell)`. They need not be integers. We therefore do not claim this is the tightest possible enclosure for every finite integer-row realization. It is sharp for endpoint-valued realizations with the required integer counts, and for the continuous-mass moment relaxation.

**Translation invariance.** Replacing each `v_i` by `v_i+c` and `t` by `t+c` leaves `S-nt`, every absolute difference in (1), and `R(t)` unchanged. An independent enclosure of `N` and `D` generally loses this cancellation. In particular, a large value offset does not itself enlarge the residual radius.

**Safe intersection.** The old signed-moment interval is

\[
 N\in[aP+bM,bP+aM],\quad D\in[na,nb].
\]

Subtracting `t[na,nb]` with interval arithmetic gives another sound residual interval. Intersecting it with (2) is sound. A disjoint intersection is an arithmetic/provenance failure, not permission to select the more convenient result.

## 5. Test the decision boundary, not a loose quotient

Let `y` be a proposed finite BF16 value and `J_y` its exact RNE cell. For an ordinary even-significand value its adjacent midpoints are included. For an odd-significand value they are excluded. The positive-zero cell under this profile is `[-2^-134,2^-134]`; negative zero is not a separate output. Negative values reverse the neighboring endpoints. Maximum finite BF16 values have a finite overflow midpoint and an open edge. These cases are explicit in `bf16_cell` and the host/device cell implementation.

### Theorem 2. Denominator-free rounding gate

Suppose `D>0`. Let `[L_lo,U_lo]` enclose `R(t_lo)` and `[L_hi,U_hi]` enclose `R(t_hi)`, where `t_lo,t_hi` are the cell endpoints. For a closed cell, the checks

\[
 L_{lo}\ge0,\qquad U_{hi}\le0
\tag{3}
\]

imply `BF16(A)=y`. For an open cell replace both comparisons by strict inequalities.

**Proof.** `R(t)=D(A-t)`. Positivity of `D` makes `R(t_lo)>=0` equivalent to `A>=t_lo`, and `R(t_hi)<=0` equivalent to `A<=t_hi`. Strict versions follow identically. The exact RNE cell definition gives the result. QED.

For several disjoint nodes, sum their residual lower and upper bounds at the same threshold. There is no division in the acceptance predicate. An exact opened leaf contributes its exact `N-tD`.

The proposal itself is untrusted. The CPU implementation uses midpoint weights and block means; the CUDA microbenchmark uses its synthetic construction's known proposal, not a dense-oracle output. A production proposer must be counted in latency. A bad proposal can cause rejection but cannot establish (3) falsely when bounds and metadata are sound.

### Why even a centered quotient can lose

An interval for `A` can be made as `t + R(t)/D`, then intersected with both the old quotient interval and the global value hull. This is sound and useful to propagate transient uncertainty. But the quotient can still be too wide because the residual and denominator extrema may be mutually incompatible. Tests (3) only need their signs and preserve more of the relevant information.

The repository's paired ablation uses the same exact rows and expansion budget for three variants: original independent quotient, centered quotient, and direct threshold residual. The direct variant accepts cases where both quotient variants fail. This is not a comparison against an intentionally incorrect baseline.

## 6. A length-independent nonconstant family

Take any positive even `n`, with half the values `63/4` and half `65/4`, so `S=16n`, `ell=16-1/4`, `u=16+1/4`. Allow arbitrary correlations with any weights in `[7/8,9/8]`. Then `m=1`, `h=1/8`.

The BF16 cell of `16` is the **asymmetric** closed interval

\[
 [511/32,257/16]=[16-1/32,16+1/16].
\]

Both endpoints lie inside the value range. Equation (1) therefore gives `C(t)=n/4` at either endpoint. At the lower boundary,

\[
 m(S-nt)-hC(t)=n/32-n/32=0.
\]

At the upper boundary,

\[
 m(S-nt)+hC(t)=-n/16+n/32=-n/32<0.
\]

The denominator is at least `7n/8>0`, so Theorem 2 proves the BF16 result is exactly 16 for every allowed pairing. Keys need not be equal and values are not constant. One root summary suffices if the forest has one root. There are zero query-time raw-row reads after summary construction.

The executable stricter ablation uses keys `+/-1/8` and the same balanced values with E24 weights. Its actual endpoint weights, rather than the preceding illustrative `[7/8,9/8]` box, are used in the check. Four-way repeated key/value pairs give an exact mean of 16. At lengths 128, 512, 2048 and 8192 the direct gate accepts from one summary; both divided variants fall back to the full cache. The looser analytic family and the actual E24 experiment are not silently identified.

This is an existence and sufficient-condition result. It gives no lower bound on acceptance for trained LLM activations. Coordinates near zero have extremely small BF16 cells; large key boxes and mixed value signs can also make these certificates unhelpful. High-dimensional multihead cuts require all needed coordinates to pass, not an average error metric.

## 7. Persistent forest and honest complexity

Seal exact raw leaves of at most `B` rows. Full leaves are combined as in a binary counter: adjacent equal-sized completed roots merge. A partial tail is never merged. Completed roots cover consecutive disjoint portions of the visible prefix, in descending ranks. If `m=floor(N/B)`, there are `popcount(m)` completed roots and at most one partial tail, hence `O(log(1+N/B))` initial root summaries.

### Theorem 3. Frontier partition and no duplicate raw reads

The initial roots and tail partition the visible rows. Replacing an internal node by its two children preserves the partition. Replacing a leaf summary by its exact contribution preserves both the partition and its exact total.

**Proof.** Initial coverage follows inductively from contiguous append and equal-sized adjacent carries. An internal split replaces `[a,c)` by `[a,b)` and `[b,c)`, which are disjoint with the same union. Exactification changes only the representation of one member of the partition. Therefore the frontier continues to be a disjoint cover. The scheduler never reopens an exact member. On fallback it traverses only unresolved members; their descendants cannot overlap any exact member. Each raw leaf is consequently read at most once during that query, including exploration and fallback. QED.

Only this exact-rational target permits arbitrary contribution reassociation. A floating-point backend may require its original reduction tree and can need a fresh dense read after a failed filter; that extra read must be charged.

Root-only certificate cost is `O((d_k+d_v) log(1+N/B))` scalar summary work, or `O(d_k+d_v)` for one root. This is a logical operation count, not a bit-complexity or GPU latency bound. Fraction arithmetic can grow in bit length; Python overhead and allocation are substantial. The actual Python scheduler scans the current frontier for priority on each action. It is not a constant-time priority queue.

An append updates the partial tail and performs at most `O(log(1+N/B))` summary merges. Over `m` completed leaves, a binary counter performs fewer than `m` merges, so merges are amortized constant per completed leaf. This does not erase the per-token summary-update cost, Python root-tuple copying, or the reference model's token-prefix tuple copy. The latter is `O(T)` in the toy implementation and is not on a claimed high-performance path.

The data structure retains all exact raw rows plus metadata; it is not a compressed-memory theorem. Initial index construction reads the prompt once and must be charged. Unlimited exploration can visit the full tree and all raw leaves. A fixed exploration budget limits wasted refinement, not fallback work or total latency.

## 8. Exact writes are enough; exact transients are not necessary

A deterministic computation can contain transient uncertainty without future state uncertainty. Let `W(h)` be all persistent writes from a transient activation `h`, and `G(h)` its terminal decision. If `W(h_hat)=W(h_ref)` and `G(h_hat)=G(h_ref)`, the transition is equal even when `h_hat != h_ref`. The statement is a liveness/property-of-consumers observation, not a claim that arbitrary activation errors are harmless.

### Theorem 4. Write-frontier acceptance

Assume a sound interval interpreter contains the reference value of every intermediate that it represents. Stage writes privately. Accept only if each newly persistent scalar is contained in a singleton interval and the final full-vocabulary greedy decision passes its interval gate. Then the committed token and persistent state equal the reference transition.

**Proof.** If an actual write `w` is enclosed by `[a,a]`, antisymmetry gives `w=a`. This holds coordinatewise for every persistent write, including all K/V coordinates at every layer. Previously committed state is unchanged. For logits `z_j in [L_j,U_j]`, a candidate `c` is the smallest-index maximizer when

\[
 U_j<L_c\text{ for }j<c,\qquad U_j\le L_c\text{ for }j>c.
\tag{4}
\]

Indeed, earlier competitors are strictly smaller and later competitors are no greater. Equality with the reference tie rule follows. Thus both fields of the committed transition are equal. QED.

The Python `writes` strategy starts from exact input embeddings, encloses normalization and projections, and checks the rounded K and V coordinates **before** appending them to a private forest. Queries may remain intervals. It reads root summaries for attention, propagates the enclosed rounded attention through the residual/FFN, and repeats for later layers. Any non-singleton write, unresolved denominator, resource/domain failure, or ambiguous final decision discards the entire attempt and runs the dense step from the original committed state. There is no partial commit and no approximate persistent cache.

A constructed two-layer decoder demonstrates strict separation: the first attention's materialized output remains non-singleton, but all actual later writes and the terminal decision are exact. The successful invocation reads no raw historical rows. This deliberately uses zero second-layer K/V projections; it is a valid separating example, **not evidence of typical pretrained behavior**. In the four random small-model trajectories, root-only whole-step write acceptance is only 7/96 steps. Those negative/limited outcomes are retained in the results.

The `cuts` strategy is less speculative in its execution: recover exact intermediate attention materializations, then run the exact suffix; only the final state-free suffix uses an argmax certificate. It often succeeds where interval uncertainty across a whole step becomes too wide. Neither strategy is universally better.

### Soundness of the reference interval interpreter

Exact point inputs use the exact reference operation. Otherwise, affine maps use signed endpoint multiplication and addition; interval squaring treats intervals containing zero correctly; the RMS denominator includes positive `EPS=1/256` and uses positive square-root enclosures. Monotonic BF16 rounding maps an enclosure to an enclosure of its materialized reference value. The E24 sigmoid factor is monotone, but SiLU itself is not globally monotone: the code multiplies an interval for `x` by an interval for its sigmoid factor rather than evaluating SiLU only at endpoints. Standard containment induction over this finite graph then establishes the interpreter premise. The supplied Lean development does not formalize this entire Python interpreter or BF16 implementation.

### Corollary 4.1. Arbitrarily long future equality

For each valid state and input, an accepted step equals the reference by Theorem 4; a rejected step returns the reference by construction. Induction on the number of transitions gives equality of the full token trace and final persistent state for every finite continuation within the reference domain. The statement has no accumulated error term or failure probability.

This is not circular reliance on token equality. The numerical gate, exact-write checks, and complete state inventory are the obligations that establish the one-step premise. Omitting a live write invalidates the proof's application.

## 9. The deployed-backend bridge is a separate theorem

A proof about E24 or real attention is not automatically a proof about a pinned GPU instruction graph. PyTorch explicitly documents nonassociativity and lack of general bitwise equivalence across mathematically identical operations, releases and platforms [R4]. A dense fallback must execute the actual chosen reference, not a faster mathematically equivalent reordering.

One possible bridge form is a proved enclosure of the backend's pre-materialization value `B` around the certified mathematical value `A`:

\[
 |B-A|\le\varepsilon.
\]

To place `B` in a closed cell `[l,u]`, it suffices to prove `A>=l+epsilon` and `A<=u-epsilon`. Equivalently, use residual thresholds shifted **inward** by the proved bound. Open cells need strict inequalities. The Lean `backend_bridge_gate` states this implication. It does not invent or discharge the `epsilon` premise.

The bridge must cover the complete implementation: scaled Q/K products, accumulation order, score shifts, exponential approximation, normalization, value accumulation, fused epilogues, conversion, exceptional values, and any reduced precision. The CUDA mathematical-function accuracy tables explicitly describe observed errors from non-exhaustive tests and do not guarantee their stated bounds [R5]. Treating such a table, or a measured maximum discrepancy, as a universal proof premise would be unsound.

The new CUDA checker instead uses outward double operations and a positive-series exponential enclosure, then adds the global E24 quantization uncertainty `2^-25`. Its claim is local E24 cut certification, conditional on device/compiler rounding semantics and sound summaries. This is not a deployed-backend bridge. The capture-only pretrained harness records actual BF16 tensors and untouched SDPA outputs, and the offline audit labels all cross-profile matches as observations. It never authorizes replacement based on those matches.

## 10. What would establish acceleration

Let `D` be dense step cost, `F` the filter/proposal cost paid on every attempt, `M` additional maintenance cost, `p` the probability of full valid acceptance under a specified workload, and `A` the remaining accepted-path work. Under this simplified serial cost model,

\[
 E[T]=F+M+pA+(1-p)D.
\]

Thus a speedup requires

\[
 \boxed{F+M<p(D-A).}
\tag{5}
\]

This is an algebraic break-even condition, not a predicted `p`. Initial indexing must be amortized over the actual number of generated tokens. Unchanged weight/MLP/logit work remains in `A`. Branching, retries, transfers, launches and synchronization can dominate. Per-head acceptance percentages do not determine full-step acceptance without their dependence structure and actual write-frontier conditions.

For GH200, first keep raw KV in HBM and measure an exact pinned backend. Test Grace-memory placement separately with page migration and interconnect traffic visible; the hardware's topology and memory configuration must be recorded rather than inferred from a product-family headline [R6]. The current CUDA microbenchmark evaluates a root gate and a separate SDPA primitive but deliberately does not call their latency ratio an LLM speedup. See `GH200.md` for the bring-up and rejection criteria.

## 11. Formal and experimental scope

`Residual.lean` states the chord, centered residual, threshold and bridge results. `ResidualCut.lean` composes actual moment inequalities with rounding to derive an exact cut. `Frontier.lean` covers disjoint-sum/count preservation and enclosure composition. `WriteFrontier.lean` covers singleton writes, tie-aware terminal decisions, and the future-trace implication. Original bounds/cuts/state/composition modules are retained.

None of these sources was accepted by a Lean compiler in this environment. The release must be described as **Lean source supplied, formal build unverified**. No `sorry`, `admit`, custom `axiom`, or `unsafe` proof declaration is intentionally used; a textual scan is not a proof check. Even a successful build would not yet verify Python, the full finite-format profile, CUDA source, generated PTX/SASS, or a pretrained adapter.

Recorded executable evidence is in `results/STATUS.json` and companion logs. It includes exhaustive finite canonical BF16 cell checks, exact-rational randomized residual tests, forest audits, fallback no-duplicate-read checks, and complete small-model state comparisons. Floating-point host tests are tests, not proofs of device behavior. The synthetic zero-read family is not held-out LLM evidence.

## 12. Research claim and limitations

The strongest supported contribution is the combination of **direct residual decision predicates, hierarchical summary refinement, and exact persistent-write acceptance**. Exact filtered predicates have substantial prior history [R1]. FlashAttention improves attention IO without this particular summary-skip contract [R2]. Recent certified-attention and attention-memory work already studies local bounds, runtime risk, and correct composition boundaries [R7-R9]. This manuscript does not claim that runtime certification, attention certificates, or state-aware reasoning are first inventions.

A defensible novelty claim needs a more exhaustive comparison of the specific residual/moment construction and write-frontier algorithm against primary literature and released code. A defensible systems claim additionally needs a proved backend bridge, compiled Lean results, device arithmetic validation, real checkpoint coverage, complete-state equivalence tests and measured end-to-end performance. The repository supplies the mathematical construction, a reference implementation and falsifiable tests to pursue that claim; it does not label the remaining obligations as completed.

References are in [REFERENCES.md](REFERENCES.md).
