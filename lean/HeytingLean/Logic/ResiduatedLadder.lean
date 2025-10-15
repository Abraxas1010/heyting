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

end Residuated
end Logic
end HeytingLean
