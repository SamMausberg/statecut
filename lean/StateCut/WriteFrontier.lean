import StateCut.Cuts
import StateCut.State
namespace StateCut

/-- Exact writes follow from containment in singleton intervals. -/
theorem singleton_writes {ι : Type*} (actual lo hi : ι → ℝ)
    (hbox : ∀ i, lo i ≤ actual i ∧ actual i ≤ hi i)
    (hgate : ∀ i, lo i = hi i) : lo = actual := by
  funext i
  have hb := hbox i
  have he := hgate i
  linarith

/-- Check persistent writes and terminal logits, not all transient activations.
All KV coordinates, positions and other mutable state must be represented by
`commit`; no omitted persistent write is licensed by this theorem. -/
theorem checked_write_frontier {n : ℕ} {ι S : Type*}
    (z zl zu : Fin n → ℝ) (c reference : Fin n)
    (href : IsGreedy z reference)
    (hz : ∀ j, zl j ≤ z j ∧ z j ≤ zu j)
    (hbefore : ∀ j, j.val < c.val → zu j < zl c)
    (hafter : ∀ j, c.val < j.val → zu j ≤ zl c)
    (writes wl wu : ι → ℝ)
    (hw : ∀ i, wl i ≤ writes i ∧ writes i ≤ wu i)
    (hsingle : ∀ i, wl i = wu i)
    (commit : (ι → ℝ) → S) :
    (c, commit wl) = (reference, commit writes) := by
  have ht := certified_greedy_equals_reference z zl zu c reference href hz hbefore hafter
  have hs := singleton_writes writes wl wu hw hsingle
  rw [ht, hs]

/-- Arbitrary transient uncertainty is compatible with exact projected writes. -/
theorem unobserved_transient_is_irrelevant {H K S T : Type*}
    (write : H → K) (commit : K → S) (decide : H → T)
    {h actual : H} (hw : write h = write actual) (ht : decide h = decide actual) :
    (decide h, commit (write h)) = (decide actual, commit (write actual)) := by
  rw [hw, ht]

/-- Whole future equality applies to a verified write-frontier proposal too. -/
theorem write_frontier_future {S T : Type*}
    (reference proposal : S → T × S) (accept : S → Bool)
    (hchecked : ∀ s, accept s = true → proposal s = reference s)
    (steps : ℕ) (initial : S) :
    run (filteredStep reference proposal accept) steps initial =
      run reference steps initial := by
  exact all_future_equal reference proposal accept hchecked steps initial

end StateCut
