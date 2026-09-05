import Mathlib

namespace StateCut

/-- Monotonic rounding is constant on an interval whose endpoints agree. -/
theorem rounding_cut {α β : Type*} [Preorder α] [PartialOrder β]
    (round : α → β) (hm : Monotone round)
    {lo x hi : α} (hx : lo ≤ x ∧ x ≤ hi)
    (he : round lo = round hi) : round x = round lo := by
  apply le_antisymm
  · calc
      round x ≤ round hi := hm hx.2
      _ = round lo := he.symm
  · exact hm hx.1

/-- Apply the exact cut coordinatewise, not merely to the token or a norm. -/
theorem vector_rounding_cut {ι α β : Type*} [Preorder α] [PartialOrder β]
    (round : α → β) (hm : Monotone round)
    (lo x hi : ι → α) (hx : ∀ i, lo i ≤ x i ∧ x i ≤ hi i)
    (he : ∀ i, round (lo i) = round (hi i)) :
    (fun i => round (x i)) = (fun i => round (lo i)) := by
  funext i
  exact rounding_cut round hm (hx i) (he i)

/-- Tie rule: choose the smallest vocabulary index among maximizers. -/
def IsGreedy {n : ℕ} (z : Fin n → ℝ) (c : Fin n) : Prop :=
  (∀ j, z j ≤ z c) ∧ (∀ j, j.val < c.val → z j < z c)

theorem argmax_certificate {n : ℕ} (z lo hi : Fin n → ℝ) (c : Fin n)
    (hbox : ∀ j, lo j ≤ z j ∧ z j ≤ hi j)
    (hbefore : ∀ j, j.val < c.val → hi j < lo c)
    (hafter : ∀ j, c.val < j.val → hi j ≤ lo c) : IsGreedy z c := by
  constructor
  · intro j
    by_cases he : j = c
    · subst j
      exact le_refl _
    · have hv : j.val ≠ c.val := fun h => he (Fin.ext h)
      by_cases hj : j.val < c.val
      · have h := hbefore j hj
        have hjb := hbox j
        have hcb := hbox c
        linarith
      · have hj' : c.val < j.val := by omega
        exact le_trans (hbox j).2 (le_trans (hafter j hj') (hbox c).1)
  · intro j hj
    exact lt_of_le_of_lt (hbox j).2 (lt_of_lt_of_le (hbefore j hj) (hbox c).1)

theorem greedy_unique {n : ℕ} {z : Fin n → ℝ} {c d : Fin n}
    (hc : IsGreedy z c) (hd : IsGreedy z d) : c = d := by
  apply Fin.ext
  rcases lt_trichotomy c.val d.val with h | h | h
  · have h1 := hd.2 c h
    have h2 := hc.1 d
    linarith
  · exact h
  · have h1 := hc.2 d h
    have h2 := hd.1 c
    linarith

/-- The acceptance theorem also handles exact logit ties. -/
theorem certified_greedy_equals_reference {n : ℕ} (z lo hi : Fin n → ℝ)
    (c reference : Fin n) (href : IsGreedy z reference)
    (hbox : ∀ j, lo j ≤ z j ∧ z j ≤ hi j)
    (hbefore : ∀ j, j.val < c.val → hi j < lo c)
    (hafter : ∀ j, c.val < j.val → hi j ≤ lo c) : c = reference := by
  exact greedy_unique (argmax_certificate z lo hi c hbox hbefore hafter) href

end StateCut
