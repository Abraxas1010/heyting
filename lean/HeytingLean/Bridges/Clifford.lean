import HeytingLean.Contracts.RoundTrip

/-!
# Clifford bridge

Geometric bridge built from pairs of `α` together with a projector that collapses onto the Heyting
core.
-/

namespace HeytingLean
namespace Bridges
namespace Clifford

open HeytingLean.Contracts
open HeytingLean.LoF

universe u

section
variable (α : Type u) [PrimaryAlgebra α]

/-- Clifford bridge model carrying pairs of `α`. -/
structure Model where
  R : Reentry α

namespace Model

variable {α : Type u} [PrimaryAlgebra α]

noncomputable def encode (M : Model α) (a : M.R.Omega) : α × α :=
  ((a : α), (a : α))

noncomputable def decode (M : Model α) (p : α × α) : M.R.Omega :=
  Reentry.Omega.mk (R := M.R) (M.R p.1) (M.R.idempotent _)

noncomputable def contract (M : Model α) : RoundTrip (R := M.R) (α × α) where
  encode := M.encode
  decode := M.decode
  round := by
    intro a
    apply Subtype.ext
    simp [encode, decode]

noncomputable def project (M : Model α) (p : α × α) : α × α :=
  (M.R p.1, M.R p.1)

lemma project_idem (M : Model α) (p : α × α) :
    M.project (M.project p) = M.project p := by
  ext <;> simp [project]

noncomputable def logicalShadow (M : Model α) : α × α → α :=
  interiorized (R := M.R) M.contract

@[simp] lemma logicalShadow_encode (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.contract.encode a) = M.R a := by
  unfold logicalShadow
  exact interiorized_id (R := M.R) (C := M.contract) a

end Model

end

end Clifford
end Bridges
end HeytingLean
