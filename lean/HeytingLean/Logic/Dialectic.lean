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
      simpa)

@[simp] lemma synth_coe (R : Reentry α) (T A : R.Omega) :
    ((synth (R := R) T A : R.Omega) : α) =
      R ((T : α) ⊔ (A : α)) := rfl

lemma le_synth_left (R : Reentry α) (T A : R.Omega) :
    T ≤ synth (R := R) T A := by
  change (T : α) ≤ R ((T : α) ⊔ (A : α))
  have h₁ : (T : α) ≤ (T : α) ⊔ (A : α) := le_sup_of_le_left le_rfl
  simpa using R.monotone h₁

lemma le_synth_right (R : Reentry α) (T A : R.Omega) :
    A ≤ synth (R := R) T A := by
  change (A : α) ≤ R ((T : α) ⊔ (A : α))
  have h₁ : (A : α) ≤ (T : α) ⊔ (A : α) := le_sup_of_le_right le_rfl
  simpa using R.monotone h₁

lemma synth_le {R : Reentry α} {T A W : R.Omega}
    (hT : T ≤ W) (hA : A ≤ W) :
    synth (R := R) T A ≤ W := by
  change R ((T : α) ⊔ (A : α)) ≤ (W : α)
  have hSup : (T : α) ⊔ (A : α) ≤ (W : α) := sup_le_iff.mpr ⟨hT, hA⟩
  have := R.monotone hSup
  simpa [Reentry.Omega.apply_coe (R := R) (a := W)] using this

end Dialectic
end Logic
end HeytingLean
