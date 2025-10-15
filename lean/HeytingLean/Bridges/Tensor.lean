import HeytingLean.Contracts.RoundTrip

/-!
# Tensor bridge

Concrete tensor carriers modelled as finite tuples of `α`, equipped with a round-trip contract that
collapses to the Heyting core via coordinate-wise interiorisation.
-/

namespace HeytingLean
namespace Bridges
namespace Tensor

open HeytingLean.Contracts
open HeytingLean.LoF

universe u

section
variable (α : Type u) [PrimaryAlgebra α]

open scoped Classical

/-- Tensor bridge data: dimension together with the core nucleus. -/
structure Model where
  dim : ℕ
  R : Reentry α

namespace Model

open scoped Classical

variable {α : Type u} [PrimaryAlgebra α]

def Carrier (M : Model α) : Type u :=
  Fin M.dim.succ → α

noncomputable def encode (M : Model α) (a : M.R.Omega) : M.Carrier :=
  fun _ => (a : α)

noncomputable def decode (M : Model α) (v : M.Carrier) : M.R.Omega :=
  let value := ⨅ i, v i
  Reentry.Omega.mk (R := M.R) (M.R value) (M.R.idempotent _)

noncomputable def contract (M : Model α) : RoundTrip (R := M.R) M.Carrier where
  encode := M.encode
  decode := M.decode
  round := by
    intro a
    ext
    classical
    simp [encode, decode]

noncomputable def interpret (M : Model α) (v : M.Carrier) : M.Carrier :=
  fun i => M.R (v i)

lemma interpret_idem (M : Model α) (v : M.Carrier) :
    M.interpret (M.interpret v) = M.interpret v := by
  classical
  funext i
  simp [interpret]

noncomputable def logicalShadow (M : Model α) : M.Carrier → α :=
  interiorized (R := M.R) M.contract

@[simp] lemma logicalShadow_encode (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.contract.encode a) = M.R a := by
  unfold logicalShadow
  exact interiorized_id (R := M.R) (C := M.contract) a

end Model

end

end Tensor
end Bridges
end HeytingLean
