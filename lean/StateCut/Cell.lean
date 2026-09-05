import StateCut.ResidualCut

open scoped BigOperators
namespace StateCut

/-- The lower endpoint can independently include or exclude its midpoint. -/
def cellLower (closed : Bool) (boundary value : ℝ) : Prop :=
  if closed then boundary ≤ value else boundary < value

/-- The upper endpoint can independently include or exclude its midpoint. -/
def cellUpper (closed : Bool) (boundary value : ℝ) : Prop :=
  if closed then value ≤ boundary else value < boundary

/-- Residual sign checks establish membership in any of the four interval
types. In particular, an overflow edge can be open while the other edge is
closed. Endpoint inclusion is part of the supplied rounding-cell contract. -/
theorem residual_cell_membership {N D low high lowerResidual upperResidual : ℝ}
    (closedLow closedHigh : Bool) (hD : 0 < D)
    (hlo : lowerResidual ≤ N-low*D) (hhi : N-high*D ≤ upperResidual)
    (hgateLow : cellLower closedLow 0 lowerResidual)
    (hgateHigh : cellUpper closedHigh 0 upperResidual) :
    cellLower closedLow low (N/D) ∧ cellUpper closedHigh high (N/D) := by
  constructor
  · cases closedLow with
    | false =>
      exact residual_lower_strict hD (lt_of_lt_of_le hgateLow hlo)
    | true =>
      exact residual_lower hD (le_trans hgateLow hlo)
  · cases closedHigh with
    | false =>
      exact residual_upper_strict hD (lt_of_le_of_lt hhi hgateHigh)
    | true =>
      exact residual_upper hD (le_trans hhi hgateHigh)

/-- A rounding function need only obey the chosen cell contract. The proof
covers open endpoints without incorrectly rounding the endpoints themselves. -/
theorem residual_to_cell {β : Type*} (round : ℝ → β) (value : β)
    {N D low high lowerResidual upperResidual : ℝ}
    (closedLow closedHigh : Bool) (hD : 0 < D)
    (hlo : lowerResidual ≤ N-low*D) (hhi : N-high*D ≤ upperResidual)
    (hgateLow : cellLower closedLow 0 lowerResidual)
    (hgateHigh : cellUpper closedHigh 0 upperResidual)
    (hcell : ∀ x, cellLower closedLow low x → cellUpper closedHigh high x →
      round x = value) : round (N/D) = value := by
  have h := residual_cell_membership closedLow closedHigh hD hlo hhi hgateLow hgateHigh
  exact hcell (N/D) h.1 h.2

/-- Count, sum and range summary checks imply exact rounding for closed,
open, or mixed cells. Concrete BF16 cell generation remains an obligation. -/
theorem moment_to_cell {ι β : Type*}
    (s : Finset ι) (w v : ι → ℝ) (round : ℝ → β) (value : β)
    {a b l u low high : ℝ} (closedLow closedHigh : Bool)
    (hab : a ≤ b) (hlu : l < u)
    (hwa : ∀ i ∈ s, a ≤ w i) (hwb : ∀ i ∈ s, w i ≤ b)
    (hvl : ∀ i ∈ s, l ≤ v i) (hvu : ∀ i ∈ s, v i ≤ u)
    (hd : 0 < ∑ i ∈ s, w i)
    (hgateLow : cellLower closedLow 0
      (residualCenter s v a b low - ((b-a)/2)*chordEnvelope s v l u low))
    (hgateHigh : cellUpper closedHigh 0
      (residualCenter s v a b high + ((b-a)/2)*chordEnvelope s v l u high))
    (hcell : ∀ x, cellLower closedLow low x → cellUpper closedHigh high x →
      round x = value) :
    round ((∑ i ∈ s, w i*v i)/(∑ i ∈ s, w i)) = value := by
  have hl := (moment_residual_bounds s w v hab hlu hwa hwb hvl hvu (t := low)).1
  have hh := (moment_residual_bounds s w v hab hlu hwa hwb hvl hvu (t := high)).2
  rw [residual_sum_identity] at hl hh
  exact residual_to_cell round value closedLow closedHigh hd hl hh hgateLow hgateHigh hcell

/-- A proved backend error bound can be combined with either endpoint rule.
The numerical bridge itself is a premise, as in backend_bridge_gate. -/
theorem backend_bridge_cell {A backend low high error : ℝ}
    (closedLow closedHigh : Bool) (hbridge : |backend-A| ≤ error)
    (hl : cellLower closedLow (low+error) A)
    (hh : cellUpper closedHigh (high-error) A) :
    cellLower closedLow low backend ∧ cellUpper closedHigh high backend := by
  rw [abs_le] at hbridge
  constructor
  · cases closedLow <;> simp_all only [cellLower, Bool.false_eq_true, if_false, if_true]
    all_goals linarith
  · cases closedHigh <;> simp_all only [cellUpper, Bool.false_eq_true, if_false, if_true]
    all_goals linarith

end StateCut
