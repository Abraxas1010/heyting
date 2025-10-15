import Mathlib.Order.Heyting.Basic
import HeytingLean.LoF.Nucleus

/-!
# Heyting core of a re-entry nucleus

Basic API for the fixed-point sublocale `Ω_R`.
-/

namespace HeytingLean
namespace LoF
namespace Reentry

variable {α : Type u} [PrimaryAlgebra α]

section CoeLemmas

variable (R : Reentry α)

@[simp, norm_cast] lemma coe_inf (a b : R.Omega) :
    ((a ⊓ b : R.Omega) : α) = (a : α) ⊓ (b : α) := rfl

@[simp, norm_cast] lemma coe_himp (a b : R.Omega) :
    ((a ⇨ b : R.Omega) : α) = (a : α) ⇨ (b : α) := rfl

@[simp] lemma apply_coe (a : R.Omega) : R (a : α) = a :=
  Reentry.Omega.apply_coe (R := R) a

end CoeLemmas

section HeytingFacts

variable (R : Reentry α)

instance instHeytingOmega : HeytingAlgebra (R.Omega) := inferInstance

lemma heyting_adjunction (a b c : R.Omega) :
    a ⊓ b ≤ c ↔ a ≤ b ⇨ c :=
  (le_himp_iff (a := a) (b := b) (c := c)).symm

lemma residuation (a b c : R.Omega) :
    c ≤ a ⇨ b ↔ c ⊓ a ≤ b :=
  (heyting_adjunction (R := R) c a b).symm

lemma double_neg (a : R.Omega) :
    a ≤ ((a ⇨ (⊥ : R.Omega)) ⇨ (⊥ : R.Omega)) := by
  have h₁ : (a ⇨ (⊥ : R.Omega)) ≤ a ⇨ (⊥ : R.Omega) := le_rfl
  have h₂ :
      (a ⇨ (⊥ : R.Omega)) ⊓ a ≤ (⊥ : R.Omega) :=
    (heyting_adjunction (R := R)
        (a := a ⇨ (⊥ : R.Omega)) (b := a) (c := ⊥)).mpr h₁
  have h₃ : a ⊓ (a ⇨ (⊥ : R.Omega)) ≤ (⊥ : R.Omega) := by
    convert h₂ using 1
    simp [inf_comm]
  exact
    (heyting_adjunction (R := R)
        (a := a) (b := a ⇨ (⊥ : R.Omega)) (c := ⊥)).mp h₃

section BooleanLimit

open scoped Classical

variable (R : Reentry α)

/-- If the nucleus is the identity, the fixed-point core is equivalent to the ambient type. -/
def booleanEquiv (h : ∀ a : α, R a = a) : R.Omega ≃ α where
  toFun := Subtype.val
  invFun := fun a => Omega.mk (R := R) a (h a)
  left_inv := by
    intro a
    ext
    rfl
  right_inv := by
    intro a
    have hx : R a = a := h a
    simp [Omega.mk]

@[simp] lemma booleanEquiv_apply (h : ∀ a : α, R a = a) (a : R.Omega) :
    booleanEquiv (R := R) h a = (a : α) := rfl

@[simp] lemma booleanEquiv_symm_apply (h : ∀ a : α, R a = a) (a : α) :
    (booleanEquiv (R := R) h).symm a = Omega.mk (R := R) a (h a) := rfl

lemma boolean_limit (h : ∀ a : α, R a = a) (a : α) :
    R ((booleanEquiv (R := R) h).symm a : R.Omega) = a := by
  have hx : R a = a := h a
  dsimp [booleanEquiv]
  change R ((Omega.mk (R := R) a hx : R.Omega) : α) = a
  simp [Omega.mk, hx]

end BooleanLimit

end HeytingFacts

end Reentry
end LoF
end HeytingLean
