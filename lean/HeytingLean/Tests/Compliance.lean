import HeytingLean.Contracts.Examples
import HeytingLean.Logic.ModalDial

open HeytingLean.LoF

namespace HeytingLean
namespace Tests

universe u

variable {α : Type u} [PrimaryAlgebra α]

theorem identity_round_verified (R : Reentry α) (a : R.Omega) :
    (Contracts.Examples.identity (α := α) R).decode
        ((Contracts.Examples.identity (α := α) R).encode a) = a :=
  Contracts.Examples.identity_round (α := α) (R := R) a

theorem tensor_shadow_verified (R : Reentry α) (n : ℕ) (a : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow (Contracts.Examples.tensor (α := α) (R := R) n))
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor (α := α) (R := R) n)).encode a)
        = R a :=
  Contracts.Examples.tensor_shadow (α := α) (R := R) n a

theorem graph_shadow_verified (R : Reentry α) (a : R.Omega) :
    (Bridges.Graph.Model.logicalShadow (Contracts.Examples.graph (α := α) (R := R)))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph (α := α) (R := R))).encode a)
        = R a :=
  Contracts.Examples.graph_shadow (α := α) (R := R) a

theorem clifford_project_idem (R : Reentry α) (p : α × α) :
    Bridges.Clifford.Model.project (Contracts.Examples.clifford (α := α) (R := R))
        (Bridges.Clifford.Model.project (Contracts.Examples.clifford (α := α) (R := R)) p)
        =
      Bridges.Clifford.Model.project (Contracts.Examples.clifford (α := α) (R := R)) p :=
  Bridges.Clifford.Model.project_idem (M := Contracts.Examples.clifford (α := α) (R := R)) p

theorem ladder_dimension_verified (R : Reentry α) :
    (Logic.Modal.DialParam.ladder (α := α) R 3).dimension = 3 :=
  Logic.Modal.DialParam.ladder_dimension (α := α) R 3

end Tests
end HeytingLean
