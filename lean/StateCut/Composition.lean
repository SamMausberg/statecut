import StateCut.Bounds
import StateCut.Cuts
import StateCut.State

open scoped BigOperators

namespace StateCut

/-- Compose signed summary inequalities, positive normalization, and a
rounding certificate. The endpoint cross-checks are certificate obligations;
equality of the rounded output is a conclusion, not a hypothesis. -/
theorem summary_to_exact_cut {ι β : Type*} [PartialOrder β]
    (s : Finset ι) (w v : ι → ℝ) (round : ℝ → β) (hm : Monotone round)
    {a b lo hi : ℝ}
    (ha : ∀ i ∈ s, a ≤ w i) (hb : ∀ i ∈ s, w i ≤ b)
    (hd : 0 < ∑ i ∈ s, w i)
    (hll : lo * ((s.card : ℝ) * a) ≤
      a * (∑ i ∈ s, max (v i) 0) + b * (∑ i ∈ s, min (v i) 0))
    (hlu : lo * ((s.card : ℝ) * b) ≤
      a * (∑ i ∈ s, max (v i) 0) + b * (∑ i ∈ s, min (v i) 0))
    (hul : b * (∑ i ∈ s, max (v i) 0) + a * (∑ i ∈ s, min (v i) 0)
      ≤ hi * ((s.card : ℝ) * a))
    (huu : b * (∑ i ∈ s, max (v i) 0) + a * (∑ i ∈ s, min (v i) 0)
      ≤ hi * ((s.card : ℝ) * b))
    (he : round lo = round hi) :
    round ((∑ i ∈ s, w i * v i) / (∑ i ∈ s, w i)) = round lo := by
  have hden := mass_bounds s w ha hb
  have hnum := signed_block_bounds s w v ha hb
  have hratio := ratio_from_cross_checks hd hden.1 hden.2 hnum.1 hnum.2
      hll hlu hul huu
  exact rounding_cut round hm hratio he

/-- A recovered vector cut is sufficient for the WHOLE suffix transition,
including all persistent writes. Its equality is derived from the interval
and rounding checks, not posited as an acceptance assumption. -/
theorem checked_cut_transition {ι α β S T : Type*}
    [Preorder α] [PartialOrder β]
    (round : α → β) (hm : Monotone round)
    (lo x hi : ι → α) (hx : ∀ i, lo i ≤ x i ∧ x i ≤ hi i)
    (he : ∀ i, round (lo i) = round (hi i))
    (suffix : (ι → β) → T × S) :
    suffix (fun i => round (lo i)) = suffix (fun i => round (x i)) := by
  exact congrArg suffix (vector_rounding_cut round hm lo x hi hx he).symm

/-- An executable rational logit gate, matching the CPU tie convention.
This is not an executable formalization of BF16, exp, or the Python program. -/
def rationalArgmaxGate {n : ℕ} (lo hi : Fin n → ℚ) (c : Fin n) : Bool :=
  decide ((∀ j, lo j ≤ hi j) ∧
    (∀ j, j.val < c.val → hi j < lo c) ∧
    (∀ j, c.val < j.val → hi j ≤ lo c))

theorem rationalArgmaxGate_sound {n : ℕ} (z lo hi : Fin n → ℚ) (c : Fin n)
    (hbox : ∀ j, lo j ≤ z j ∧ z j ≤ hi j)
    (haccept : rationalArgmaxGate lo hi c = true) :
    (∀ j, z j ≤ z c) ∧ (∀ j, j.val < c.val → z j < z c) := by
  have hc : (∀ j, lo j ≤ hi j) ∧
      (∀ j, j.val < c.val → hi j < lo c) ∧
      (∀ j, c.val < j.val → hi j ≤ lo c) := by
    exact of_decide_eq_true haccept
  constructor
  · intro j
    by_cases he : j = c
    · subst j
      exact le_refl _
    · have hv : j.val ≠ c.val := fun h => he (Fin.ext h)
      by_cases hj : j.val < c.val
      · exact le_of_lt (lt_of_le_of_lt (hbox j).2
          (lt_of_lt_of_le (hc.2.1 j hj) (hbox c).1))
      · have hj' : c.val < j.val := by omega
        exact le_trans (hbox j).2 (le_trans (hc.2.2 j hj') (hbox c).1)
  · intro j hj
    exact lt_of_le_of_lt (hbox j).2 (lt_of_lt_of_le (hc.2.1 j hj) (hbox c).1)

end StateCut
