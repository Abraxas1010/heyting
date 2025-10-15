import HeytingLean.Contracts.RoundTrip

/-!
# Graph bridge

The graph bridge uses the ambient type `α` as vertices with adjacency given by the order relation.
-/

namespace HeytingLean
namespace Bridges
namespace Graph

open HeytingLean.Contracts
open HeytingLean.LoF

universe u

section
variable (α : Type u) [PrimaryAlgebra α]

/-- Graph bridge model: vertices and the core nucleus. -/
structure Model where
  R : Reentry α

namespace Model

variable {α : Type u} [PrimaryAlgebra α]

def adjacency (_M : Model α) : α → α → Prop :=
  (· ≤ ·)

noncomputable def encode (M : Model α) (a : M.R.Omega) : α := (a : α)

noncomputable def decode (M : Model α) (x : α) : M.R.Omega :=
  Reentry.Omega.mk (R := M.R) (M.R x) (M.R.idempotent _)

noncomputable def contract (M : Model α) : RoundTrip (R := M.R) α where
  encode := M.encode
  decode := M.decode
  round := by
    intro a
    apply Subtype.ext
    simp [encode, decode]

noncomputable def logicalShadow (M : Model α) : α → α :=
  interiorized (R := M.R) M.contract

@[simp] lemma logicalShadow_encode (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.contract.encode a) = M.R a := by
  unfold logicalShadow
  exact interiorized_id (R := M.R) (C := M.contract) a

lemma adjacency_refl (M : Model α) (a : α) :
    M.adjacency a a := le_rfl

lemma adjacency_trans (M : Model α) {a b c : α}
    (hab : M.adjacency a b) (hbc : M.adjacency b c) :
    M.adjacency a c :=
  le_trans hab hbc

end Model

end

end Graph
end Bridges
end HeytingLean
