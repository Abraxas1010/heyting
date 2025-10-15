import Mathlib.Order.Nucleus
import Mathlib.Order.Sublocale
import HeytingLean.LoF.PrimaryAlgebra

namespace HeytingLean
namespace LoF

/-- `Reentry α` packages the re-entry operation as a nucleus on a primary algebra. -/
structure Reentry (α : Type u) [PrimaryAlgebra α] where
  nucleus : Nucleus α

namespace Reentry

variable {α : Type u} [PrimaryAlgebra α]

instance : CoeFun (Reentry α) (fun _ => α → α) where
  coe R := R.nucleus

@[simp] lemma coe_nucleus (R : Reentry α) : (R.nucleus : α → α) = R := rfl

@[simp] lemma idempotent (R : Reentry α) (a : α) : R (R a) = R a :=
  R.nucleus.idempotent _

@[simp] lemma le_apply (R : Reentry α) (a : α) : a ≤ R a :=
  Nucleus.le_apply (n := R.nucleus) (x := a)

lemma map_inf (R : Reentry α) (a b : α) : R (a ⊓ b) = R a ⊓ R b :=
  Nucleus.map_inf (n := R.nucleus) (x := a) (y := b)

@[simp] lemma monotone (R : Reentry α) : Monotone R :=
  R.nucleus.monotone

/-- Fixed points of the nucleus viewed as the associated sublocale `Ω_R`. -/
abbrev Omega (R : Reentry α) : Type u := R.nucleus.toSublocale

namespace Omega

variable (R : Reentry α)

def mk (a : α) (h : R a = a) :
    R.Omega := ⟨a, ⟨a, h⟩⟩

@[simp] lemma coe_mk (a : α) (h : R a = a) :
    ((Omega.mk (R := R) a h : R.Omega) : α) = a := rfl

@[simp] lemma apply_coe (a : R.Omega) : R (a : α) = a := by
  obtain ⟨x, hx⟩ := a.property
  have hx₁ : R (a : α) = R (R x) := congrArg R hx.symm
  have hx₂ : R (R x) = R x := R.idempotent x
  exact hx₁.trans (hx₂.trans hx)

@[simp] lemma apply_mk (a : α) (h : R a = a) :
    R ((Omega.mk (R := R) a h : R.Omega) : α) = Omega.mk (R := R) a h := by
  simpa using (apply_coe (R := R) (a := Omega.mk (R := R) a h))

end Omega

end Reentry

end LoF
end HeytingLean
