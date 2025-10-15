import HeytingLean.LoF.HeytingCore
import HeytingLean.LoF.Nucleus

namespace HeytingLean
namespace Docs

/-- A toy primary algebra given by the power set Boolean algebra on `Unit`. -/
instance (priority := 100) : PrimaryAlgebra (Set Unit) :=
  inferInstance

/-- The identity nucleus on `Set Unit`. -/
def idNucleus : Reentry (Set Unit) :=
{ nucleus := ⟨
    id,
    by intro _; rfl,
    by intro _ _; exact le_rfl,
    by intro _ _; rfl⟩,
  primordial := Set.univ,
  counter := Set.univ,
  primordial_mem := rfl,
  counter_mem := rfl,
  primordial_nonbot := by
    refine lt_of_le_of_ne ?_ ?_
    · exact bot_le
    · exact Set.univ_ne_bot.symm
  counter_nonbot := by
    refine lt_of_le_of_ne ?_ ?_
    · exact bot_le
    · exact Set.univ_ne_bot.symm
  orthogonal := by simp,
  primordial_minimal := by
    intro x hx_fix hx_pos
    have : Set.univ ≤ x := by intro _ _; trivial
    exact this }

/-- Under the identity nucleus the Boolean equivalence experienced in `Tests.Boolean` holds. -/
example : idNucleus.booleanEquiv (by intro _; rfl) ⟨Set.univ, Set.univ, rfl⟩ = Set.univ := rfl

end Docs
end HeytingLean
