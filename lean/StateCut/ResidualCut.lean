import StateCut.Residual
import StateCut.Cuts
open scoped BigOperators
namespace StateCut

/-- The sufficient-statistic absolute-deviation envelope. -/
noncomputable def chordEnvelope {ι : Type*} (s : Finset ι) (v : ι → ℝ) (l u t : ℝ) : ℝ :=
  (((s.card : ℝ)*u-(∑ i ∈ s, v i))*|l-t| +
    ((∑ i ∈ s, v i)-(s.card : ℝ)*l)*|u-t|)/(u-l)

noncomputable def residualCenter {ι : Type*} (s : Finset ι) (v : ι → ℝ) (a b t : ℝ) : ℝ :=
  ((a+b)/2)*((∑ i ∈ s, v i)-(s.card : ℝ)*t)

/-- Translation of values and thresholds leaves the complete summary envelope
unchanged, including its extrema-dependent absolute deviations. -/
theorem chordEnvelope_shift {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (l u t c : ℝ) :
    chordEnvelope s (fun i => v i+c) (l+c) (u+c) (t+c) =
      chordEnvelope s v l u t := by
  simp only [chordEnvelope, Finset.sum_add_distrib, Finset.sum_const,
    nsmul_eq_mul, add_sub_add_right_eq_sub]
  congr 1; ring

/-- The residual center has the same value/threshold translation symmetry. -/
theorem residualCenter_shift {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (a b t c : ℝ) :
    residualCenter s (fun i => v i+c) a b (t+c) = residualCenter s v a b t := by
  simp only [residualCenter, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul]
  ring

/-- The absolute residual theorem supplies both endpoints needed by a checker. -/
theorem moment_residual_bounds {ι : Type*} (s : Finset ι) (w v : ι → ℝ)
    {a b l u t : ℝ} (hab : a ≤ b) (hlu : l < u)
    (hwa : ∀ i ∈ s, a ≤ w i) (hwb : ∀ i ∈ s, w i ≤ b)
    (hvl : ∀ i ∈ s, l ≤ v i) (hvu : ∀ i ∈ s, v i ≤ u) :
    residualCenter s v a b t - ((b-a)/2)*chordEnvelope s v l u t ≤
        (∑ i ∈ s, w i*(v i-t)) ∧
    (∑ i ∈ s, w i*(v i-t)) ≤
        residualCenter s v a b t + ((b-a)/2)*chordEnvelope s v l u t := by
  have h := moment_residual_sound s w v hab hlu hwa hwb hvl hvu (t := t)
  change |(∑ i ∈ s, w i*(v i-t))-residualCenter s v a b t| ≤
    ((b-a)/2)*chordEnvelope s v l u t at h
  rw [abs_le] at h
  constructor <;> linarith

/-- Link threshold residuals to the actual attention numerator and mass. -/
theorem residual_sum_identity {ι : Type*} (s : Finset ι) (w v : ι → ℝ) (t : ℝ) :
    (∑ i ∈ s, w i*(v i-t)) =
      (∑ i ∈ s, w i*v i)-t*(∑ i ∈ s, w i) := by
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- Concrete summary checks imply equality of a closed rounding cut.
The exact output is a conclusion. The hypotheses are range provenance,
positive mass, computed summary inequalities, and the rounding-cell contract.
Open cells use the strict residual lemmas in Residual.lean instead. -/
theorem moment_to_exact_cut {ι β : Type*} [PartialOrder β]
    (s : Finset ι) (w v : ι → ℝ) (round : ℝ → β) (hm : Monotone round)
    {a b l u low high : ℝ} (hab : a ≤ b) (hlu : l < u)
    (hwa : ∀ i ∈ s, a ≤ w i) (hwb : ∀ i ∈ s, w i ≤ b)
    (hvl : ∀ i ∈ s, l ≤ v i) (hvu : ∀ i ∈ s, v i ≤ u)
    (hd : 0 < ∑ i ∈ s, w i)
    (hgateLow : 0 ≤ residualCenter s v a b low - ((b-a)/2)*chordEnvelope s v l u low)
    (hgateHigh : residualCenter s v a b high + ((b-a)/2)*chordEnvelope s v l u high ≤ 0)
    (hcell : round low = round high) :
    round ((∑ i ∈ s, w i*v i)/(∑ i ∈ s, w i)) = round low := by
  have hl := moment_residual_sound s w v hab hlu hwa hwb hvl hvu (t := low)
  have hh := moment_residual_sound s w v hab hlu hwa hwb hvl hvu (t := high)
  change |(∑ i ∈ s, w i*(v i-low))-residualCenter s v a b low| ≤
    ((b-a)/2)*chordEnvelope s v l u low at hl
  change |(∑ i ∈ s, w i*(v i-high))-residualCenter s v a b high| ≤
    ((b-a)/2)*chordEnvelope s v l u high at hh
  rw [abs_le] at hl hh
  have rl : 0 ≤ ∑ i ∈ s, w i*(v i-low) := by linarith
  have rh : (∑ i ∈ s, w i*(v i-high)) ≤ 0 := by linarith
  rw [residual_sum_identity] at rl rh
  exact rounding_cut round hm ⟨residual_lower hd rl, residual_upper hd rh⟩ hcell

end StateCut
