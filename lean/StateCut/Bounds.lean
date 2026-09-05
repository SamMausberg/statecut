import Mathlib

open scoped BigOperators

/-!
Algebraic certificate theorems, over arbitrary finite sets and real inputs.
No floating-point error estimate is smuggled in as a theorem. A deployed
backend must establish the score/weight and arithmetic enclosures.

STATUS: Source provided; not compiler-checked in the authoring environment.
-/
namespace StateCut

/-- Signed values require opposite weight endpoints in the negative term. -/
theorem signed_term_bounds {a b w v : ℝ} (ha : a ≤ w) (hb : w ≤ b) :
    a * max v 0 + b * min v 0 ≤ w * v ∧
    w * v ≤ b * max v 0 + a * min v 0 := by
  by_cases hv : 0 ≤ v
  · simpa [max_eq_left hv, min_eq_right hv] using
      And.intro (mul_le_mul_of_nonneg_right ha hv)
        (mul_le_mul_of_nonneg_right hb hv)
  · have hv' : v ≤ 0 := le_of_not_ge hv
    simpa [max_eq_right hv', min_eq_left hv'] using
      And.intro (mul_le_mul_of_nonpos_right hb hv')
        (mul_le_mul_of_nonpos_right ha hv')

/-- Query-independent positive/negative value sums enclose block numerators. -/
theorem signed_block_bounds {ι : Type*} (s : Finset ι)
    (w v : ι → ℝ) {a b : ℝ}
    (ha : ∀ i ∈ s, a ≤ w i) (hb : ∀ i ∈ s, w i ≤ b) :
    a * (∑ i ∈ s, max (v i) 0) + b * (∑ i ∈ s, min (v i) 0)
      ≤ ∑ i ∈ s, w i * v i ∧
    (∑ i ∈ s, w i * v i) ≤
      b * (∑ i ∈ s, max (v i) 0) + a * (∑ i ∈ s, min (v i) 0) := by
  constructor
  · have h := Finset.sum_le_sum (fun i hi =>
        (signed_term_bounds (v := v i) (ha i hi) (hb i hi)).1)
    simpa only [Finset.sum_add_distrib, ← Finset.mul_sum] using h
  · have h := Finset.sum_le_sum (fun i hi =>
        (signed_term_bounds (v := v i) (ha i hi) (hb i hi)).2)
    simpa only [Finset.sum_add_distrib, ← Finset.mul_sum] using h

/-- Block denominator bounds. -/
theorem mass_bounds {ι : Type*} (s : Finset ι) (w : ι → ℝ)
    {a b : ℝ} (ha : ∀ i ∈ s, a ≤ w i) (hb : ∀ i ∈ s, w i ≤ b) :
    (s.card : ℝ) * a ≤ ∑ i ∈ s, w i ∧
    (∑ i ∈ s, w i) ≤ (s.card : ℝ) * b := by
  constructor
  · simpa [nsmul_eq_mul] using Finset.sum_le_sum ha
  · simpa [nsmul_eq_mul] using Finset.sum_le_sum hb

/-- Exact-query dot-product enclosure from a coordinatewise key box. -/
theorem dot_box {ι : Type*} (s : Finset ι) (q k kl ku : ι → ℝ)
    (hlo : ∀ i ∈ s, kl i ≤ k i) (hhi : ∀ i ∈ s, k i ≤ ku i) :
    (∑ i ∈ s, if 0 ≤ q i then q i * kl i else q i * ku i)
      ≤ (∑ i ∈ s, q i * k i) ∧
    (∑ i ∈ s, q i * k i) ≤
      (∑ i ∈ s, if 0 ≤ q i then q i * ku i else q i * kl i) := by
  constructor
  · apply Finset.sum_le_sum
    intro i hi
    by_cases hq : 0 ≤ q i
    · simpa [hq] using mul_le_mul_of_nonneg_left (hlo i hi) hq
    · simpa [hq] using mul_le_mul_of_nonpos_left (hhi i hi) (le_of_not_ge hq)
  · apply Finset.sum_le_sum
    intro i hi
    by_cases hq : 0 ≤ q i
    · simpa [hq] using mul_le_mul_of_nonneg_left (hhi i hi) hq
    · simpa [hq] using mul_le_mul_of_nonpos_left (hlo i hi) (le_of_not_ge hq)

/-- This is an actual real-exponential bound, not an empirical error budget. -/
theorem exp_weight_bounds {l s u : ℝ} (hl : l ≤ s) (hu : s ≤ u) :
    Real.exp l ≤ Real.exp s ∧ Real.exp s ≤ Real.exp u := by
  exact ⟨Real.exp_le_exp.mpr hl, Real.exp_le_exp.mpr hu⟩

/-- The finite reference uses a monotone weight quantizer. Its implementation
and monotonicity-to-code correspondence are outside this formalization. -/
theorem monotone_weight_bounds (E : ℝ → ℝ) (hm : Monotone E)
    {l s u : ℝ} (hl : l ≤ s) (hu : s ≤ u) : E l ≤ E s ∧ E s ≤ E u := by
  exact ⟨hm hl, hm hu⟩

/-- Verify a normalized interval by four cross-multiplication checks.
Unlike a naive Nlo/Dlo formula this works with negative numerators. -/
theorem ratio_from_cross_checks {dl d du nl n nu l u : ℝ}
    (hd : 0 < d) (hdl : dl ≤ d) (hdu : d ≤ du)
    (hnl : nl ≤ n) (hnu : n ≤ nu)
    (hll : l * dl ≤ nl) (hlu : l * du ≤ nl)
    (hul : nu ≤ u * dl) (huu : nu ≤ u * du) :
    l ≤ n / d ∧ n / d ≤ u := by
  have hl : l * d ≤ n := by
    by_cases hs : 0 ≤ l
    · have h := mul_le_mul_of_nonneg_left hdu hs
      linarith
    · have h := mul_le_mul_of_nonpos_left hdl (le_of_not_ge hs)
      linarith
  have hu : n ≤ u * d := by
    by_cases hs : 0 ≤ u
    · have h := mul_le_mul_of_nonneg_left hdl hs
      linarith
    · have h := mul_le_mul_of_nonpos_left hdu (le_of_not_ge hs)
      linarith
  exact ⟨(le_div_iff₀ hd).2 hl, (div_le_iff₀ hd).2 hu⟩

/-- A non-vacuous no-raw-scan family: equal keys imply equal weights, and
an exact count/value-sum summary determines attention for every query. -/
theorem constant_weight_attention {w n s : ℝ} (hw : w ≠ 0) (hn : n ≠ 0) :
    (w * s) / (n * w) = s / n := by
  field_simp [hw, hn] <;> ring

end StateCut
