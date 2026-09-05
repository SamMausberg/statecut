import StateCut.Cell

open scoped BigOperators
namespace StateCut

/-- The sharp absolute-deviation envelope for an integer count decomposition. -/
noncomputable def finiteEnvelope (n k : ℕ) (l u t c : ℝ) : ℝ :=
  (k : ℝ)*|u-t|+((n : ℝ)-(k : ℝ)-1)*|l-t|+|c-t|

/-- The absolute value is a linear expression plus twice its positive part. -/
theorem abs_eq_two_max_sub (x : ℝ) : |x| = 2 * max x 0 - x := by
  by_cases hx : 0 ≤ x
  · rw [abs_of_nonneg hx, max_eq_left hx]
    ring
  · have hx' : x ≤ 0 := le_of_not_ge hx
    rw [abs_of_nonpos hx', max_eq_right hx']
    ring

/-- Integer row counts sharpen a positive-part envelope. The sum condition
describes k upper endpoints, n-k-1 lower endpoints, and one residual value c.
The proof partitions the actual rows above t and uses their integer count. -/
theorem finite_positive_part_bound {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (k : ℕ) {l u t c : ℝ}
    (hl : ∀ i ∈ s, l ≤ v i) (hu : ∀ i ∈ s, v i ≤ u)
    (hlt : l ≤ t) (htu : t ≤ u)
    (hsum : (∑ i ∈ s, v i) =
      (k : ℝ)*u + ((s.card : ℝ)-(k : ℝ)-1)*l+c) :
    (∑ i ∈ s, max (v i-t) 0) ≤ (k : ℝ)*(u-t)+max (c-t) 0 := by
  classical
  let p := s.filter (fun i => t < v i)
  have hp : (∑ i ∈ s, if t < v i then t-l else 0) = (p.card : ℝ)*(t-l) := by
    rw [← Finset.sum_filter]
    simp only [p, Finset.sum_const, nsmul_eq_mul]
  have hrepr : (∑ i ∈ s, max (v i-t) 0) = (∑ i ∈ p, (v i-t)) := by
    rw [Finset.sum_filter]
    apply Finset.sum_congr rfl
    intro i hi
    by_cases h : t < v i
    · simp only [h, if_true, max_eq_left (sub_nonneg.mpr (le_of_lt h))]
    · simp only [h, if_false, max_eq_right (sub_nonpos.mpr (le_of_not_gt h))]
  have hupper : (∑ i ∈ s, max (v i-t) 0) ≤ (p.card : ℝ)*(u-t) := by
    rw [hrepr]
    have h : (∑ i ∈ p, (v i-t)) ≤ ∑ _i ∈ p, (u-t) :=
      Finset.sum_le_sum (fun i hi =>
        sub_le_sub_right (hu i (Finset.mem_filter.mp hi).1) t)
    simpa only [Finset.sum_const, nsmul_eq_mul] using h
  have htotal : (∑ i ∈ s, max (v i-t) 0) ≤
      (∑ i ∈ s, v i)-(s.card : ℝ)*l-(p.card : ℝ)*(t-l) := by
    have hpoint : ∀ i ∈ s, max (v i-t) 0 ≤
        (v i-l)-(if t < v i then t-l else 0) := by
      intro i hi
      by_cases h : t < v i
      · simp only [h, if_true, max_eq_left (sub_nonneg.mpr (le_of_lt h))]
        linarith
      · simp only [h, if_false, max_eq_right (sub_nonpos.mpr (le_of_not_gt h)), sub_zero]
        exact sub_nonneg.mpr (hl i hi)
    have h := Finset.sum_le_sum hpoint
    simpa only [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, hp] using h
  by_cases hpk : p.card ≤ k
  · have hcast : (p.card : ℝ) ≤ (k : ℝ) := by exact_mod_cast hpk
    have h := mul_le_mul_of_nonneg_right hcast (sub_nonneg.mpr htu)
    have hm := le_max_right (c-t) (0 : ℝ)
    linarith
  · have hkp : k+1 ≤ p.card := by omega
    have hcast : (k : ℝ)+1 ≤ (p.card : ℝ) := by exact_mod_cast hkp
    have h := mul_le_mul_of_nonneg_right hcast (sub_nonneg.mpr hlt)
    have hm := le_max_left (c-t) (0 : ℝ)
    rw [hsum] at htotal
    nlinarith

/-- The finite-count envelope bounds every real row family with the given
count, sum and range. The floor operation is unnecessary in this statement:
the implementation must supply an integer k and residual c with this sum. -/
theorem finite_abs_sum_bound {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (k : ℕ) {l u t c : ℝ}
    (hl : ∀ i ∈ s, l ≤ v i) (hu : ∀ i ∈ s, v i ≤ u)
    (hlu : l ≤ u) (hcl : l ≤ c) (hcu : c ≤ u)
    (hsum : (∑ i ∈ s, v i) =
      (k : ℝ)*u + ((s.card : ℝ)-(k : ℝ)-1)*l+c) :
    (∑ i ∈ s, |v i-t|) ≤
      (k : ℝ)*|u-t|+((s.card : ℝ)-(k : ℝ)-1)*|l-t|+|c-t| := by
  by_cases htl : t ≤ l
  · have hpoint : ∀ i ∈ s, |v i-t| = v i-t := by
      intro i hi
      exact abs_of_nonneg (sub_nonneg.mpr (le_trans htl (hl i hi)))
    rw [Finset.sum_congr rfl hpoint]
    simp only [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
      abs_of_nonneg (sub_nonneg.mpr (le_trans htl hlu)),
      abs_of_nonneg (sub_nonneg.mpr htl),
      abs_of_nonneg (sub_nonneg.mpr (le_trans htl hcl))]
    rw [hsum]
    nlinarith
  · by_cases hut : u ≤ t
    · have hpoint : ∀ i ∈ s, |v i-t| = -(v i-t) := by
        intro i hi
        exact abs_of_nonpos (sub_nonpos.mpr (le_trans (hu i hi) hut))
      rw [Finset.sum_congr rfl hpoint]
      simp only [Finset.sum_neg_distrib, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
        abs_of_nonpos (sub_nonpos.mpr hut),
        abs_of_nonpos (sub_nonpos.mpr (le_trans hlu hut)),
        abs_of_nonpos (sub_nonpos.mpr (le_trans hcu hut))]
      rw [hsum]
      nlinarith
    · have hlt : l ≤ t := le_of_not_ge htl
      have htu : t ≤ u := le_of_not_ge hut
      have h := finite_positive_part_bound s v k hl hu hlt htu hsum
      have hid : (∑ i ∈ s, |v i-t|) =
          2*(∑ i ∈ s, max (v i-t) 0) - (∑ i ∈ s, v i)+(s.card : ℝ)*t := by
        simp only [abs_eq_two_max_sub, Finset.sum_sub_distrib, ← Finset.mul_sum,
          Finset.sum_const, nsmul_eq_mul]
        ring
      rw [hid, abs_of_nonneg (sub_nonneg.mpr htu),
        abs_of_nonpos (sub_nonpos.mpr hlt), abs_eq_two_max_sub, hsum]
      nlinarith

/-- The sharpened finite-count envelope composes with the proved centered
residual inequality, without any assumption about value/weight correlation. -/
theorem finite_residual_bound {ι : Type*} (s : Finset ι) (w v : ι → ℝ)
    (k : ℕ) {a b l u t c : ℝ} (hab : a ≤ b)
    (hwa : ∀ i ∈ s, a ≤ w i) (hwb : ∀ i ∈ s, w i ≤ b)
    (hvl : ∀ i ∈ s, l ≤ v i) (hvu : ∀ i ∈ s, v i ≤ u)
    (hlu : l ≤ u) (hcl : l ≤ c) (hcu : c ≤ u)
    (hsum : (∑ i ∈ s, v i) =
      (k : ℝ)*u+((s.card : ℝ)-(k : ℝ)-1)*l+c) :
    |(∑ i ∈ s, w i*(v i-t))-residualCenter s v a b t| ≤
      ((b-a)/2)*((k : ℝ)*|u-t|+((s.card : ℝ)-(k : ℝ)-1)*|l-t|+|c-t|) := by
  apply centered_residual_bound s w v
  · linarith
  · intro i hi
    exact half_width_deviation (hwa i hi) (hwb i hi)
  · exact finite_abs_sum_bound s v k hvl hvu hlu hcl hcu hsum

/-- The integer-count improvement is never wider than the original chord
envelope. Its entire slack is the chord slack of the one residual value. -/
theorem finite_envelope_le_chord {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (k : ℕ) {l u t c : ℝ} (hlu : l < u) (hcl : l ≤ c) (hcu : c ≤ u)
    (hsum : (∑ i ∈ s, v i) =
      (k : ℝ)*u+((s.card : ℝ)-(k : ℝ)-1)*l+c) :
    (k : ℝ)*|u-t|+((s.card : ℝ)-(k : ℝ)-1)*|l-t|+|c-t| ≤
      chordEnvelope s v l u t := by
  unfold chordEnvelope
  apply (le_div_iff₀ (sub_pos.mpr hlu)).2
  have h := abs_chord_mul hlu hcl hcu (t := t)
  rw [hsum]
  nlinarith

/-- An explicit row list attains the finite envelope: k copies of u, n-k-1
copies of l, and one copy of c. Together with finite_abs_sum_bound this proves
sharpness over real-valued row families described by count, sum and bounds. -/
theorem finite_envelope_attained (n k : ℕ) (hk : k < n)
    (l u t c : ℝ) (hlu : l ≤ u) (hcl : l ≤ c) (hcu : c ≤ u) :
    ∃ values : List ℝ, values.length = n ∧
      (∀ x ∈ values, l ≤ x ∧ x ≤ u) ∧
      values.sum = (k : ℝ)*u+((n : ℝ)-(k : ℝ)-1)*l+c ∧
      (values.map (fun x => |x-t|)).sum =
        (k : ℝ)*|u-t|+((n : ℝ)-(k : ℝ)-1)*|l-t|+|c-t| := by
  have hsub : ((n-k-1 : ℕ) : ℝ) = (n : ℝ)-(k : ℝ)-1 := by
    rw [Nat.cast_sub (by omega : 1 ≤ n-k), Nat.cast_sub (le_of_lt hk), Nat.cast_one]
  refine ⟨List.replicate k u ++ List.replicate (n-k-1) l ++ [c], ?_, ?_, ?_, ?_⟩
  · simp only [List.length_append, List.length_replicate, List.length_singleton]
    omega
  · intro x hx
    simp only [List.mem_append, List.mem_replicate, List.mem_singleton] at hx
    rcases hx with (⟨_, rfl⟩ | ⟨_, rfl⟩) | rfl
    · exact ⟨hlu, le_rfl⟩
    · exact ⟨le_rfl, hlu⟩
    · exact ⟨hcl, hcu⟩
  · simp [List.sum_append, List.sum_replicate, nsmul_eq_mul, hsub, add_assoc]
  · simp [List.map_append, List.map_replicate, List.sum_append,
      List.sum_replicate, nsmul_eq_mul, hsub, add_assoc]

/-- Exact floor arithmetic supplies a valid remainder and sum decomposition.
This is a theorem about mathematical floor; it is not a refinement proof of
Python's integer arithmetic or a floating-point implementation. -/
theorem floor_remainder_contract {S n l u : ℝ} (hlu : l < u) (hS : n*l ≤ S) :
    let r := (S-n*l)/(u-l)
    let k := ⌊r⌋₊
    let c := l+(r-(k : ℝ))*(u-l)
    l ≤ c ∧ c < u ∧ S = (k : ℝ)*u+(n-(k : ℝ)-1)*l+c := by
  dsimp only
  have hd : 0 < u-l := sub_pos.mpr hlu
  have hr : 0 ≤ (S-n*l)/(u-l) := div_nonneg (sub_nonneg.mpr hS) (le_of_lt hd)
  have hk := Nat.floor_le hr
  have hsucc := Nat.lt_floor_add_one ((S-n*l)/(u-l))
  have hlo := mul_nonneg (sub_nonneg.mpr hk) (le_of_lt hd)
  have hhi := mul_lt_mul_of_pos_right hsucc hd
  have hid : ((S-n*l)/(u-l))*(u-l) = S-n*l := div_mul_cancel₀ _ (ne_of_gt hd)
  refine ⟨?_, ?_, ?_⟩
  · linarith
  · nlinarith
  · nlinarith

/-- Finite-count summary checks establish exact rounding with independently
open or closed endpoints, using the same explicit cell contract as Cell.lean. -/
theorem finite_to_cell {ι β : Type*}
    (s : Finset ι) (w v : ι → ℝ) (k : ℕ) (round : ℝ → β) (value : β)
    {a b l u c low high : ℝ} (closedLow closedHigh : Bool)
    (hab : a ≤ b) (hlu : l ≤ u) (hcl : l ≤ c) (hcu : c ≤ u)
    (hwa : ∀ i ∈ s, a ≤ w i) (hwb : ∀ i ∈ s, w i ≤ b)
    (hvl : ∀ i ∈ s, l ≤ v i) (hvu : ∀ i ∈ s, v i ≤ u)
    (hsum : (∑ i ∈ s, v i) =
      (k : ℝ)*u+((s.card : ℝ)-(k : ℝ)-1)*l+c)
    (hd : 0 < ∑ i ∈ s, w i)
    (hgateLow : cellLower closedLow 0
      (residualCenter s v a b low - ((b-a)/2)*finiteEnvelope s.card k l u low c))
    (hgateHigh : cellUpper closedHigh 0
      (residualCenter s v a b high + ((b-a)/2)*finiteEnvelope s.card k l u high c))
    (hcell : ∀ x, cellLower closedLow low x → cellUpper closedHigh high x →
      round x = value) :
    round ((∑ i ∈ s, w i*v i)/(∑ i ∈ s, w i)) = value := by
  have hl := finite_residual_bound s w v k hab hwa hwb hvl hvu hlu hcl hcu hsum (t := low)
  have hh := finite_residual_bound s w v k hab hwa hwb hvl hvu hlu hcl hcu hsum (t := high)
  change |(∑ i ∈ s, w i*(v i-low))-residualCenter s v a b low| ≤
    ((b-a)/2)*finiteEnvelope s.card k l u low c at hl
  change |(∑ i ∈ s, w i*(v i-high))-residualCenter s v a b high| ≤
    ((b-a)/2)*finiteEnvelope s.card k l u high c at hh
  rw [abs_le] at hl hh
  have hlo : residualCenter s v a b low - ((b-a)/2)*finiteEnvelope s.card k l u low c ≤
      (∑ i ∈ s, w i*v i)-low*(∑ i ∈ s, w i) := by
    rw [← residual_sum_identity]
    linarith
  have hhi : (∑ i ∈ s, w i*v i)-high*(∑ i ∈ s, w i) ≤
      residualCenter s v a b high + ((b-a)/2)*finiteEnvelope s.card k l u high c := by
    rw [← residual_sum_identity]
    linarith
  exact residual_to_cell round value closedLow closedHigh hd hlo hhi hgateLow hgateHigh hcell

/-- Choosing a weight-box endpoint according to the deviation sign attains
the upper residual radius for any fixed values. This is an abstract weight
box result, without a claim that a trained attention model realizes it. -/
theorem residual_upper_attained {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (a b t : ℝ) (hab : a ≤ b) :
    let w := fun i => if 0 ≤ v i-t then b else a
    (∀ i ∈ s, a ≤ w i ∧ w i ≤ b) ∧
    (∑ i ∈ s, w i*(v i-t)) =
      residualCenter s v a b t+((b-a)/2)*(∑ i ∈ s, |v i-t|) := by
  dsimp only
  constructor
  · intro i hi
    by_cases h : 0 ≤ v i-t
    · simp only [h, if_true]
      exact ⟨hab, le_rfl⟩
    · simp only [h, if_false]
      exact ⟨le_rfl, hab⟩
  · unfold residualCenter
    rw [← centered_first_moment s v t, Finset.mul_sum, Finset.mul_sum,
      ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    by_cases h : 0 ≤ v i-t
    · rw [if_pos h, abs_of_nonneg h]
      ring
    · rw [if_neg h, abs_of_nonpos (le_of_not_ge h)]
      ring

/-- Reversing the endpoint choices attains the lower residual radius. -/
theorem residual_lower_attained {ι : Type*} (s : Finset ι) (v : ι → ℝ)
    (a b t : ℝ) (hab : a ≤ b) :
    let w := fun i => if 0 ≤ v i-t then a else b
    (∀ i ∈ s, a ≤ w i ∧ w i ≤ b) ∧
    (∑ i ∈ s, w i*(v i-t)) =
      residualCenter s v a b t-((b-a)/2)*(∑ i ∈ s, |v i-t|) := by
  dsimp only
  constructor
  · intro i hi
    by_cases h : 0 ≤ v i-t
    · simp only [h, if_true]
      exact ⟨le_rfl, hab⟩
    · simp only [h, if_false]
      exact ⟨hab, le_rfl⟩
  · unfold residualCenter
    rw [← centered_first_moment s v t, Finset.mul_sum, Finset.mul_sum,
      ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i hi
    by_cases h : 0 ≤ v i-t
    · rw [if_pos h, abs_of_nonneg h]
      ring
    · rw [if_neg h, abs_of_nonpos (le_of_not_ge h)]
      ring

end StateCut
