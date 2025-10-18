import HeytingLean.Logic.Triad

/-!
# Residuated ladder on the Heyting core

We package the deduction/abduction/induction triad as the three faces of the residuation
equivalence, preparing the transition towards MV/effect layers.
-/

namespace HeytingLean
namespace Logic
namespace Residuated

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α] (R : Reentry α)

/-- Deduction is inherited directly from the Heyting core. -/
abbrev deduction := Logic.deduction (R := R)

/-- Abduction captures the `B ≤ A ⇒ C` face of residuation. -/
def abduction (a b c : R.Omega) : Prop :=
  b ≤ a ⇨ c

/-- Induction captures the `A ≤ B ⇒ C` face of residuation. -/
def induction (a b c : R.Omega) : Prop :=
  a ≤ b ⇨ c

lemma deduction_iff_induction (a b c : R.Omega) :
    deduction (R := R) a b c ↔ induction (R := R) a b c :=
  Logic.deduction_iff (R := R) a b c

lemma deduction_iff_abduction (a b c : R.Omega) :
    deduction (R := R) a b c ↔ abduction (R := R) a b c := by
  unfold deduction abduction Logic.deduction
  constructor
  · intro h
    have h' := h
    rw [inf_comm] at h'
    exact (Reentry.heyting_adjunction (R := R) b a c).mp h'
  · intro h
    have h' := (Reentry.heyting_adjunction (R := R) b a c).mpr h
    rw [inf_comm] at h'
    exact h'

/-- The ladder triangle: all three faces are equivalent. -/
lemma abduction_iff_induction (a b c : R.Omega) :
    abduction (R := R) a b c ↔ induction (R := R) a b c :=
  ((deduction_iff_abduction (R := R) a b c).symm).trans
    (deduction_iff_induction (R := R) a b c)

@[simp] lemma deduction_euler_boundary_left (b c : R.Omega) :
    deduction (R := R) R.eulerBoundary b c ↔
      R.eulerBoundary ⊓ b ≤ c := Iff.rfl

@[simp] lemma abduction_euler_boundary_left (b c : R.Omega) :
    abduction (R := R) R.eulerBoundary b c ↔ b ≤ R.eulerBoundary ⇨ c :=
  Iff.rfl

lemma induction_le_from_euler (b c : R.Omega) :
    induction (R := R) R.eulerBoundary b c ↔
      R.eulerBoundary ≤ b ⇨ c := Iff.rfl

lemma eulerBoundary_collapse :
    deduction (R := R) R.eulerBoundary
      (R.eulerBoundary ⇨ (⊥ : R.Omega)) ⊥ :=
  by
    have := Logic.double_neg_collapse (R := R) (a := R.eulerBoundary)
    simpa using this

lemma eulerBoundary_induction_collapse :
    induction (R := R) R.eulerBoundary
      (R.eulerBoundary ⇨ (⊥ : R.Omega)) ⊥ := by
  have := eulerBoundary_collapse (R := R)
  exact (deduction_iff_induction (R := R) _ _ _).mp this

lemma eulerBoundary_abduction_collapse :
    abduction (R := R) R.eulerBoundary
      (R.eulerBoundary ⇨ (⊥ : R.Omega)) ⊥ := by
  have := eulerBoundary_collapse (R := R)
  exact (deduction_iff_abduction (R := R) _ _ _).mp this

lemma himp_closed (a b : R.Omega) :
    R ((a : α) ⇨ (b : α)) = (a : α) ⇨ (b : α) := by
  have hb : R ((b : α)) = (b : α) :=
    Reentry.Omega.apply_coe (R := R) (a := b)
  simpa [hb] using
    Reentry.map_himp_apply (R := R) (a := (a : α)) (b := (b : α))

lemma map_himp_le (a b : α) :
    R (a ⇨ b) ≤ a ⇨ R b :=
  Reentry.map_himp_le (R := R) (a := a) (b := b)

end Residuated
end Logic
end HeytingLean
