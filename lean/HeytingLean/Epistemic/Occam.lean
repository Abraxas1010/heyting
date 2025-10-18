import HeytingLean.LoF.Nucleus

namespace HeytingLean
namespace Epistemic

open HeytingLean.LoF
open scoped Classical

variable {α : Type u} [PrimaryAlgebra α]

/-- A minimal-stage “birthday” for the re-entry closure: fixed points stabilise immediately,
everything else resolves after one additional breath. -/
noncomputable def birth (R : Reentry α) (a : α) : ℕ :=
  if R a = a then 0 else 1

@[simp] lemma birth_eq_zero_of_fixed (R : Reentry α) {a : α}
    (h : R a = a) : birth R a = 0 := by
  simp [birth, h]

@[simp] lemma birth_eq_one_of_not_fixed (R : Reentry α) {a : α}
    (h : R a ≠ a) : birth R a = 1 := by
  simp [birth, h]

lemma birth_le_one (R : Reentry α) (a : α) : birth R a ≤ 1 := by
  classical
  by_cases h : R a = a
  · simp [birth, h]
  · simp [birth, h]

/-- Candidate explanations: the fixed points that stay within the specification `a`. -/
def occamCandidates (R : Reentry α) (a : α) : Set α :=
  {u | u ≤ a ∧ R u = u}

lemma occamCandidate_le (R : Reentry α) {a u : α}
    (hu : u ∈ occamCandidates (R := R) a) : u ≤ a :=
  hu.1

lemma occamCandidate_fixed (R : Reentry α) {a u : α}
    (hu : u ∈ occamCandidates (R := R) a) : R u = u :=
  hu.2

/-- The raw union of invariant explanations lying underneath `a`. -/
def occamCore (R : Reentry α) (a : α) : α :=
  sSup (occamCandidates (R := R) a)

lemma occamCore_le (R : Reentry α) (a : α) :
    occamCore (R := R) a ≤ a := by
  refine sSup_le ?_
  intro u hu
  exact occamCandidate_le (R := R) hu

lemma le_occamCore_of_fixed (R : Reentry α) {a u : α}
    (hu_le : u ≤ a) (hu_fix : R u = u) :
    u ≤ occamCore (R := R) a := by
  change u ≤ sSup _
  refine le_sSup ?_
  exact And.intro hu_le hu_fix

/-- The Occam reduction closes the core back into an invariant explanation. -/
def occam (R : Reentry α) (a : α) : α :=
  R (occamCore (R := R) a)

lemma occam_le_reentry (R : Reentry α) (a : α) :
    occam (R := R) a ≤ R a :=
  R.monotone (occamCore_le (R := R) a)

lemma occam_contains_candidate (R : Reentry α) {a u : α}
    (hu_le : u ≤ a) (hu_fix : R u = u) :
    u ≤ occam (R := R) a := by
  have hCore := le_occamCore_of_fixed (R := R) hu_le hu_fix
  have hMon := R.monotone hCore
  have hGoal := hMon
  simp [hu_fix] at hGoal
  change u ≤ R (occamCore (R := R) a)
  exact hGoal

lemma occam_idempotent (R : Reentry α) (a : α) :
    R (occam (R := R) a) = occam (R := R) a := by
  simp [occam]

lemma occam_monotone (R : Reentry α) :
    Monotone (occam (R := R)) := by
  intro a b h
  have hSet :
      occamCandidates (R := R) a ⊆ occamCandidates (R := R) b := by
    intro u hu
    exact And.intro (le_trans hu.1 h) hu.2
  have hCore :
      occamCore (R := R) a ≤ occamCore (R := R) b :=
    sSup_le_sSup hSet
  exact R.monotone hCore

lemma occam_birth (R : Reentry α) (a : α) :
    birth R (occam (R := R) a) = 0 :=
  birth_eq_zero_of_fixed (R := R) (occam_idempotent (R := R) (a := a))

@[simp] lemma birth_eulerBoundary (R : Reentry α) :
    birth R ((R.eulerBoundary : R.Omega) : α) = 0 :=
  birth_eq_zero_of_fixed (R := R)
    (a := ((R.eulerBoundary : R.Omega) : α))
    (Reentry.Omega.apply_coe (R := R) (a := R.eulerBoundary))

end Epistemic
end HeytingLean
