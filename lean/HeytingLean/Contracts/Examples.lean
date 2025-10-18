import HeytingLean.Contracts.RoundTrip
import HeytingLean.Bridges.RoundTrip
import HeytingLean.Bridges.Tensor
import HeytingLean.Bridges.Graph
import HeytingLean.Bridges.Clifford

/-!
# Contract examples

Simple round-trip examples that exercise the identity bridge.
-/

namespace HeytingLean
namespace Contracts
namespace Examples

open HeytingLean.Bridges
open HeytingLean.Bridges.Tensor
open HeytingLean.Bridges.Graph
open HeytingLean.Bridges.Clifford
open HeytingLean.LoF

universe u

variable (α : Type u) [PrimaryAlgebra α] (R : Reentry α)

def identity : RoundTrip (R := R) α :=
  HeytingLean.Bridges.identityContract (R := R)

@[simp] lemma identity_round (a : R.Omega) :
    (identity α R).decode ((identity α R).encode a) = a := by
  change HeytingLean.Bridges.decode (R := R)
      (HeytingLean.Bridges.encode (R := R) a) = a
  exact HeytingLean.Bridges.decode_encode (R := R) a

@[simp] lemma identity_shadow (a : R.Omega) :
    interiorized (R := R) (identity α R) ((identity α R).encode a) = R a := by
  change interiorized (R := R)
      (HeytingLean.Bridges.identityContract (R := R)) ((HeytingLean.Bridges.identityContract (R := R)).encode a) = R a
  exact interiorized_id (R := R) (C := HeytingLean.Bridges.identityContract (R := R)) a

def tensor (n : ℕ) : HeytingLean.Bridges.Tensor.Model α :=
  ⟨n, R⟩

@[simp] lemma tensor_round (n : ℕ) (a : R.Omega) :
    (HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).decode
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode a) = a := by
  simpa [tensor] using
    (HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).round a

@[simp] lemma tensor_shadow (n : ℕ) (a : R.Omega) :
    (HeytingLean.Bridges.Tensor.Model.logicalShadow (tensor α R n))
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode a) = R a := by
  simp [HeytingLean.Bridges.Tensor.Model.logicalShadow, tensor]

/-- Convenience lemma: the tensor bridge's implication transport reduces via `simp`. -/
@[simp] lemma tensor_shadow_himp (n : ℕ) (a b : R.Omega) :
    (HeytingLean.Bridges.Tensor.Model.logicalShadow (tensor α R n))
      (HeytingLean.Bridges.Tensor.Model.stageHimp
        (tensor α R n)
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode a)
        ((HeytingLean.Bridges.Tensor.Model.contract (tensor α R n)).encode b))
      =
        R (a ⇨ b) := by
  classical
  simp [tensor]
def graph : HeytingLean.Bridges.Graph.Model α :=
  ⟨R⟩

@[simp] lemma graph_round (a : R.Omega) :
    (HeytingLean.Bridges.Graph.Model.contract (graph α R)).decode
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode a) = a := by
  simpa [graph] using
    (HeytingLean.Bridges.Graph.Model.contract (graph α R)).round a

@[simp] lemma graph_shadow (a : R.Omega) :
    (HeytingLean.Bridges.Graph.Model.logicalShadow (graph α R))
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode a) = R a := by
  simp [HeytingLean.Bridges.Graph.Model.logicalShadow, graph]

/-- Convenience lemma: the graph bridge's implication transport reduces via `simp`. -/
@[simp] lemma graph_shadow_himp (a b : R.Omega) :
    (HeytingLean.Bridges.Graph.Model.logicalShadow (graph α R))
      (HeytingLean.Bridges.Graph.Model.stageHimp
        (graph α R)
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode a)
        ((HeytingLean.Bridges.Graph.Model.contract (graph α R)).encode b))
      =
        R (a ⇨ b) := by
  classical
  simp [graph]
def clifford : HeytingLean.Bridges.Clifford.Model α :=
  ⟨R⟩

@[simp] lemma clifford_round (a : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).decode
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode a) = a := by
  simpa [clifford] using
    (HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).round a

@[simp] lemma clifford_shadow (a : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.logicalShadow (clifford α R))
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode a) = R a := by
  simp [HeytingLean.Bridges.Clifford.Model.logicalShadow, clifford]

/-- Convenience lemma: the Clifford bridge's implication transport reduces via `simp`. -/
@[simp] lemma clifford_shadow_himp (a b : R.Omega) :
    (HeytingLean.Bridges.Clifford.Model.logicalShadow (clifford α R))
      (HeytingLean.Bridges.Clifford.Model.stageHimp
        (clifford α R)
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode a)
        ((HeytingLean.Bridges.Clifford.Model.contract (clifford α R)).encode b))
      =
        R (a ⇨ b) := by
  classical
  simp [clifford]
end Examples
end Contracts
end HeytingLean
