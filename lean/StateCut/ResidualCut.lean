import StateCut.Residual
import StateCut.Cuts
open scoped BigOperators
namespace StateCut

/-- The sufficient-statistic absolute-deviation envelope. -/
def chordEnvelope {ι : Type*} (s : Finset ι) (v : ι → ℝ) (l u t : ℝ) : ℝ :=
  (((s.card : ℝ)*u-(∑ i ∈ s, v i))*|l-t| +
    ((∑ i ∈ s, v i)-(s.card : ℝ)*l)*|u-t|)/(u-l)

def residualCenter {ι : Type*} (s : Finset ι) (v : ι → ℝ) (a b t : ℝ) : ℝ :=
  ((a+b)/2)*((∑ i ∈ s, v i)-(s.card : ℝ)*t)

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
