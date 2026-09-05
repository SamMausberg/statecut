import Mathlib

namespace StateCut

/-- A reference step returns BOTH token and persistent state. -/
def filteredStep {S T : Type*} (reference : S → T × S)
    (proposal : S → T × S) (accept : S → Bool) (s : S) : T × S :=
  if accept s then proposal s else reference s

/-- Acceptance must certify the entire transition, not only its first field. -/
theorem filteredStep_eq {S T : Type*} (reference proposal : S → T × S)
    (accept : S → Bool)
    (hsound : ∀ s, accept s = true → proposal s = reference s) :
    filteredStep reference proposal accept = reference := by
  funext s
  by_cases h : accept s = true
  · simp [filteredStep, h, hsound s h]
  · simp [filteredStep, h]

/-- A chronological deterministic trace with its final persistent state. -/
def run {S T : Type*} (step : S → T × S) : ℕ → S → List T × S
  | 0, s => ([], s)
  | n + 1, s =>
    let first := step s
    let rest := run step n first.2
    (first.1 :: rest.1, rest.2)

/-- Arbitrarily long future decoding is identical, by transition equality. -/
theorem all_future_equal {S T : Type*} (reference proposal : S → T × S)
    (accept : S → Bool)
    (hsound : ∀ s, accept s = true → proposal s = reference s)
    (n : ℕ) (s : S) :
    run (filteredStep reference proposal accept) n s = run reference n s := by
  rw [filteredStep_eq reference proposal accept hsound]

/-- Equal materialization cuts permit every deterministic suffix, including
new KV projections, to be reused without approximation. -/
theorem exact_cut_suffix {C S T : Type*} (suffix : C → T × S)
    {actual recovered : C} (h : recovered = actual) :
    suffix recovered = suffix actual := by
  exact congrArg suffix h

/-- A terminal suffix with no persistent-state writes only needs the token
certificate. The previously formed KV state is exactly the reference state. -/
theorem terminal_cut {S T : Type*} (state : S) {token reference : T}
    (h : token = reference) : (token, state) = (reference, state) := by
  exact congrArg (fun x => (x, state)) h

/-- Model the liveness condition explicitly: persistent state depends ONLY
on the recovered cut C, whereas transient Y may remain enclosed. -/
theorem state_frontier {C Y S T : Type*} (persist : C → S) (decide : Y → T)
    {c cr : C} {y yr : Y} (hc : c = cr) (ht : decide y = decide yr) :
    (decide y, persist c) = (decide yr, persist cr) := by
  cases hc
  exact congrArg (fun t => (t, persist c)) ht

/-- No-summary indistinguishability lemma. This is NOT a lower bound against
all possible preprocessed data structures. -/
theorem indistinguishable_observations {X O T : Type*}
    (observe : X → O) (algorithm : O → T) (desired : X → T)
    {x y : X} (hobs : observe x = observe y) (hdiff : desired x ≠ desired y) :
    ¬ (algorithm (observe x) = desired x ∧ algorithm (observe y) = desired y) := by
  intro h
  apply hdiff
  calc
    desired x = algorithm (observe x) := h.1.symm
    _ = algorithm (observe y) := congrArg algorithm hobs
    _ = desired y := h.2

end StateCut
