import Mathlib.Logic.Function.Iteration
import HeytingLean.LoF.Nucleus
import HeytingLean.Epistemic.Occam

namespace HeytingLean
namespace Logic
namespace PSR

open HeytingLean.LoF
open Epistemic

variable {α : Type u} [PrimaryAlgebra α]

/-- The Principle of Sufficient Reason: a proposition is sufficient precisely when it is invariant
under the re-entry nucleus. -/
def Sufficient (R : Reentry α) (a : α) : Prop :=
  R a = a

@[simp] lemma sufficient_iff (R : Reentry α) (a : α) :
    Sufficient R a ↔ R a = a := Iff.rfl

lemma sufficient_of_fixed (R : Reentry α) (a : α)
    (ha : R a = a) : Sufficient R a :=
  ha

lemma fixed_of_sufficient (R : Reentry α) {a : α}
    (ha : Sufficient R a) : R a = a :=
  ha

lemma sufficient_stable (R : Reentry α) {a x : α}
    (ha : Sufficient R a) (hx : x ≤ a) :
    R x ≤ a := by
  have hx' : R.nucleus x ≤ R.nucleus a := R.monotone hx
  have ha' : R.nucleus a = a := by
    simpa [Reentry.coe_nucleus] using ha
  have hx'' : R.nucleus x ≤ a := by
    simpa [ha'] using hx'
  simpa [Reentry.coe_nucleus] using hx''

/-- Minimal reasons exist at each dial: the Occam reduction is invariant. -/
lemma occam_sufficient (R : Reentry α) (a : α) :
    Sufficient R (Epistemic.occam (R := R) a) :=
  occam_idempotent (R := R) (a := a)

end PSR
end Logic
end HeytingLean
