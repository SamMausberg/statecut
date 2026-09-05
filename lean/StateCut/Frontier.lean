import Mathlib
open scoped BigOperators
namespace StateCut

/-- A disjoint split preserves the underlying exact aggregate. -/
theorem disjoint_split_sum {ι : Type*} [DecidableEq ι] (left right : Finset ι) (f : ι → ℝ)
    (hd : Disjoint left right) :
    (∑ i ∈ left ∪ right, f i) = (∑ i ∈ left, f i)+(∑ i ∈ right, f i) := by
  exact Finset.sum_union hd

/-- A split also preserves the count needed by the moment certificate. -/
theorem disjoint_split_count {ι : Type*} [DecidableEq ι] (left right : Finset ι)
    (hd : Disjoint left right) : (left ∪ right).card = left.card+right.card := by
  exact Finset.card_union_of_disjoint hd

/-- Independent scheduling cannot damage enclosure soundness at a split. -/
theorem split_preserves_bounds {x y lx ux ly uy : ℝ}
    (hx : lx ≤ x ∧ x ≤ ux) (hy : ly ≤ y ∧ y ≤ uy) :
    lx+ly ≤ x+y ∧ x+y ≤ ux+uy := by
  exact ⟨add_le_add hx.1 hy.1, add_le_add hx.2 hy.2⟩

/-- Intersecting old and new enclosures cannot remove the true value. -/
theorem intersect_preserves {x l u a b : ℝ}
    (h : l ≤ x ∧ x ≤ u) (k : a ≤ x ∧ x ≤ b) :
    max l a ≤ x ∧ x ≤ min u b := by
  exact ⟨max_le h.1 k.1, le_min h.2 k.2⟩

/-- A frontier of sound per-node residual bounds gives a sound global bound. -/
theorem frontier_sum_bounds {ι : Type*} (s : Finset ι) (r lo hi : ι → ℝ)
    (h : ∀ i ∈ s, lo i ≤ r i ∧ r i ≤ hi i) :
    (∑ i ∈ s, lo i) ≤ (∑ i ∈ s, r i) ∧
    (∑ i ∈ s, r i) ≤ (∑ i ∈ s, hi i) := by
  exact ⟨Finset.sum_le_sum (fun i hi => (h i hi).1),
    Finset.sum_le_sum (fun i hi => (h i hi).2)⟩

/-- An eager current token can agree while future observations differ.
This explicit counterexample rules out token-only state commits. -/
theorem token_only_is_insufficient :
    (0, false).1 = (0, true).1 ∧ (0, false).2 ≠ (0, true).2 := by
  decide

end StateCut
