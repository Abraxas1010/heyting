import Mathlib.Order.Basic
import Mathlib.Data.Set.Lattice
import HeytingLean.Bridges.Graph
import HeytingLean.Contracts.RoundTrip

/-
Minimal Alexandroff scaffolding for the graph bridge.  We keep a reference to the core graph model
and an auxiliary open set while reusing the existing transport.
-/

namespace HeytingLean
namespace Bridges
namespace Graph
namespace Alexandroff

open HeytingLean.Contracts
open HeytingLean.LoF

universe u

variable {α : Type u}

section
variable [PrimaryAlgebra α]

/-- Alexandroff graph model pointing at the core bridge. -/
structure Model where
  core : Graph.Model α
  openSet : Set α := Set.univ

namespace Model

def Carrier (M : Model (α := α)) : Type u := M.core.Carrier

noncomputable def encode (M : Model (α := α)) :
    M.core.R.Omega → M.Carrier :=
  Graph.Model.encode (M := M.core)

noncomputable def decode (M : Model (α := α)) :
    M.Carrier → M.core.R.Omega :=
  Graph.Model.decode (M := M.core)

noncomputable def contract (M : Model (α := α)) :
    Contracts.RoundTrip (M.core.R) M.Carrier :=
  Graph.Model.contract (M := M.core)

noncomputable def logicalShadow (M : Model (α := α)) :
    M.Carrier → α :=
  Graph.Model.logicalShadow (M := M.core)

@[simp] lemma logicalShadow_encode (M : Model (α := α))
    (a : M.core.R.Omega) :
    M.logicalShadow (M.contract.encode a) = M.core.R a :=
  Graph.Model.logicalShadow_encode (M := M.core) (a := a)

@[simp] lemma decode_encode (M : Model (α := α)) (a : M.core.R.Omega) :
    M.decode (M.contract.encode a) = a :=
  Graph.Model.decode_encode (M := M.core) (a := a)

noncomputable def stageMvAdd (M : Model (α := α)) :
    M.Carrier → M.Carrier → M.Carrier :=
  Graph.Model.stageMvAdd (M := M.core)

def stageEffectCompatible (M : Model (α := α))
    (x y : M.Carrier) : Prop :=
  Graph.Model.stageEffectCompatible (M := M.core) x y

noncomputable def stageEffectAdd? (M : Model (α := α))
    (x y : M.Carrier) : Option M.Carrier :=
  Graph.Model.stageEffectAdd? (M := M.core) x y

noncomputable def stageOrthocomplement (M : Model (α := α)) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageOrthocomplement (M := M.core)

noncomputable def stageHimp (M : Model (α := α)) :
    M.Carrier → M.Carrier → M.Carrier :=
  Graph.Model.stageHimp (M := M.core)

noncomputable def stageCollapseAt (M : Model (α := α)) (n : ℕ) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageCollapseAt (M := M.core) n

noncomputable def stageExpandAt (M : Model (α := α)) (n : ℕ) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageExpandAt (M := M.core) n

noncomputable def stageOccam (M : Model (α := α)) :
    M.Carrier → M.Carrier :=
  Graph.Model.stageOccam (M := M.core)

end Model

end

end Alexandroff
end Graph
end Bridges
end HeytingLean
