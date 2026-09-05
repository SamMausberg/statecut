import Mathlib

open scoped BigOperators
namespace StateCut

/-- A chord of the convex absolute-value function, in denominator-free form. -/
theorem abs_chord_mul {l v u t : ℝ} (hlu : l < u)
    (hlv : l ≤ v) (hvu : v ≤ u) :
    (u-l)*|v-t| ≤ (u-v)*|l-t| + (v-l)*|u-t| := by
  have hid : (u-l)*(v-t) = (u-v)*(l-t)+(v-l)*(u-t) := by ring
  calc
    (u-l)*|v-t| = |(u-l)*(v-t)| := by
      rw [abs_mul, abs_of_pos (sub_pos.mpr hlu)]
    _ = |(u-v)*(l-t)+(v-l)*(u-t)| := by rw [hid]
    _ ≤ |(u-v)*(l-t)|+|(v-l)*(u-t)| := abs_add _ _
    _ = (u-v)*|l-t|+(v-l)*|u-t| := by
      rw [abs_mul, abs_mul, abs_of_nonneg (sub_nonneg.mpr hvu),
        abs_of_nonneg (sub_nonneg.mpr hlv)]

/-- Only count, sum and extrema appear on the right. -/
theorem abs_chord_sum {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    {l u t : ℝ} (hlu : l < u)
    (hl : ∀ i ∈ s, l ≤ v i) (hu : ∀ i ∈ s, v i ≤ u) :
    (u-l)*(∑ i ∈ s, |v i-t|) ≤
      ((s.card : ℝ)*u-(∑ i ∈ s, v i))*|l-t| +
      ((∑ i ∈ s, v i)-(s.card : ℝ)*l)*|u-t| := by
  have h := Finset.sum_le_sum (fun i hi => abs_chord_mul hlu (hl i hi) (hu i hi) (t := t))
  simpa only [← Finset.mul_sum, Finset.sum_add_distrib, ← Finset.sum_mul,
    Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul] using h

/-- Weight-box centering needs no differentiability or exponential assumption. -/
theorem half_width_deviation {a b w : ℝ} (ha : a ≤ w) (hb : w ≤ b) :
    |w-(a+b)/2| ≤ (b-a)/2 := by
  rw [abs_le]
  constructor <;> linarith

/-- The correlation-preserving residual identity. -/
theorem residual_center_identity {ι : Type*} (s : Finset ι) (w v : ι → ℝ)
    (m t : ℝ) :
    (∑ i ∈ s, w i*(v i-t)) - m*(∑ i ∈ s, (v i-t)) =
      ∑ i ∈ s, (w i-m)*(v i-t) := by
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- Sum of deviations is computed from a mergeable first moment. -/
theorem centered_first_moment {ι : Type*} (s : Finset ι) (v : ι → ℝ) (t : ℝ) :
    (∑ i ∈ s, (v i-t)) = (∑ i ∈ s, v i) - (s.card : ℝ)*t := by
  simp [Finset.sum_sub_distrib, nsmul_eq_mul]

/-- A finite-sum residual enclosure. The chord theorem supplies hA. -/
theorem centered_residual_bound {ι : Type*} (s : Finset ι) (w v : ι → ℝ)
    {m h t A : ℝ} (hh : 0 ≤ h)
    (hdev : ∀ i ∈ s, |w i-m| ≤ h)
    (hA : (∑ i ∈ s, |v i-t|) ≤ A) :
    |(∑ i ∈ s, w i*(v i-t)) - m*((∑ i ∈ s, v i)-(s.card : ℝ)*t)| ≤ h*A := by
  rw [← centered_first_moment s v t, residual_center_identity]
  calc
    |∑ i ∈ s, (w i-m)*(v i-t)| ≤ ∑ i ∈ s, |(w i-m)*(v i-t)| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ s, h*|v i-t| := by
      apply Finset.sum_le_sum
      intro i hi
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_right (hdev i hi) (abs_nonneg _)
    _ = h*(∑ i ∈ s, |v i-t|) := by rw [Finset.mul_sum]
    _ ≤ h*A := mul_le_mul_of_nonneg_left hA hh

/-- The actual count/sum/range certificate used by the implementation. -/
theorem moment_residual_sound {ι : Type*} (s : Finset ι) (w v : ι → ℝ)
    {a b l u t : ℝ} (hab : a ≤ b) (hlu : l < u)
    (hwa : ∀ i ∈ s, a ≤ w i) (hwb : ∀ i ∈ s, w i ≤ b)
    (hvl : ∀ i ∈ s, l ≤ v i) (hvu : ∀ i ∈ s, v i ≤ u) :
    |(∑ i ∈ s, w i*(v i-t)) -
        ((a+b)/2)*((∑ i ∈ s, v i)-(s.card : ℝ)*t)| ≤
      ((b-a)/2)*
        ((((s.card : ℝ)*u-(∑ i ∈ s, v i))*|l-t| +
          ((∑ i ∈ s, v i)-(s.card : ℝ)*l)*|u-t|)/(u-l)) := by
  apply centered_residual_bound s w v
  · linarith
  · intro i hi
    exact half_width_deviation (hwa i hi) (hwb i hi)
  · apply (le_div_iff₀ (sub_pos.mpr hlu)).2
    have h := abs_chord_sum s v hlu hvl hvu (t := t)
    nlinarith

/-- Constant-value blocks cancel their denominator exactly, regardless of keys. -/
theorem constant_value_residual {ι : Type*} (s : Finset ι) (w : ι → ℝ) (c t : ℝ) :
    (∑ i ∈ s, w i*(c-t)) = (∑ i ∈ s, w i)*(c-t) := by
  rw [Finset.sum_mul]

/-- Lower and upper cell tests require no division in the checker. -/
theorem residual_lower {N D t : ℝ} (hD : 0 < D) (h : 0 ≤ N-t*D) : t ≤ N/D := by
  apply (le_div_iff₀ hD).2
  linarith

theorem residual_upper {N D t : ℝ} (hD : 0 < D) (h : N-t*D ≤ 0) : N/D ≤ t := by
  apply (div_le_iff₀ hD).2
  linarith

theorem residual_lower_strict {N D t : ℝ} (hD : 0 < D) (h : 0 < N-t*D) : t < N/D := by
  apply (lt_div_iff₀ hD).2
  linarith

theorem residual_upper_strict {N D t : ℝ} (hD : 0 < D) (h : N-t*D < 0) : N/D < t := by
  apply (div_lt_iff₀ hD).2
  linarith

/-- Value shifts do not change the residual's centered first moment. -/
theorem first_moment_shift (S n t c : ℝ) :
    (S+n*c)-n*(t+c) = S-n*t := by ring

/-- A bridge is an explicit mathematical hypothesis, not a guessed tolerance. -/
theorem backend_bridge_gate {A backend lo hi e : ℝ}
    (hbridge : |backend-A| ≤ e) (hl : lo+e ≤ A) (hu : A ≤ hi-e) :
    lo ≤ backend ∧ backend ≤ hi := by
  rw [abs_le] at hbridge
  constructor <;> linarith

/-- Translation-independent sufficient condition for a balanced value range.
This algebraic corollary is the margin condition; it does not assert that any
particular checkpoint satisfies the condition. -/
theorem balanced_margin {n m h r delta : ℝ}
    (hn : 0 < n) (hmargin : h*r < m*delta) :
    0 < m*(n*delta)-h*(n*r) := by
  have h := mul_pos hn (sub_pos.mpr hmargin)
  nlinarith

end StateCut
