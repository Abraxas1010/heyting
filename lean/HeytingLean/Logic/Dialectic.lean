import HeytingLean.LoF.Nucleus

namespace HeytingLean
namespace Logic
namespace Dialectic

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α]

/-- Synthesis of two invariant arguments: close the union and stay in `Ω_R`. -/
def synth (R : Reentry α) (T A : R.Omega) : R.Omega :=
  Reentry.Omega.mk (R := R)
    (R ((T : α) ⊔ (A : α)))
    (by
      have := Reentry.idempotent (R := R) (a := (T : α) ⊔ (A : α))
      simp)

@[simp] lemma synth_coe (R : Reentry α) (T A : R.Omega) :
    ((synth (R := R) T A : R.Omega) : α) =
      R ((T : α) ⊔ (A : α)) := rfl

lemma le_synth_left (R : Reentry α) (T A : R.Omega) :
    T ≤ synth (R := R) T A := by
  change (T : α) ≤ R ((T : α) ⊔ (A : α))
  have h₁ : (T : α) ≤ (T : α) ⊔ (A : α) := le_sup_of_le_left le_rfl
  have hMon := R.monotone h₁
  have hGoal := hMon
  simp [Reentry.Omega.apply_coe (R := R) (a := T)] at hGoal
  exact hGoal

lemma le_synth_right (R : Reentry α) (T A : R.Omega) :
    A ≤ synth (R := R) T A := by
  change (A : α) ≤ R ((T : α) ⊔ (A : α))
  have h₁ : (A : α) ≤ (T : α) ⊔ (A : α) := le_sup_of_le_right le_rfl
  have hMon := R.monotone h₁
  have hGoal := hMon
  simp [Reentry.Omega.apply_coe (R := R) (a := A)] at hGoal
  exact hGoal

lemma synth_le {R : Reentry α} {T A W : R.Omega}
    (hT : T ≤ W) (hA : A ≤ W) :
    synth (R := R) T A ≤ W := by
  change R ((T : α) ⊔ (A : α)) ≤ (W : α)
  have hSup : (T : α) ⊔ (A : α) ≤ (W : α) := sup_le_iff.mpr ⟨hT, hA⟩
  have := R.monotone hSup
  have hGoal := this
  simp [Reentry.Omega.apply_coe (R := R) (a := W)] at hGoal
  exact hGoal

lemma synth_eulerBoundary_self (R : Reentry α) :
    synth (R := R) R.eulerBoundary R.eulerBoundary = R.eulerBoundary := by
  apply le_antisymm
  · exact synth_le (R := R) le_rfl le_rfl
  · exact le_synth_left (R := R) R.eulerBoundary R.eulerBoundary

end Dialectic
end Logic
end HeytingLean
