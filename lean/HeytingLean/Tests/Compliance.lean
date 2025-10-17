import HeytingLean.Contracts.Examples
import HeytingLean.Logic.ModalDial
import HeytingLean.Logic.Triad
import HeytingLean.Logic.StageSemantics
import HeytingLean.Epistemic.Occam
import HeytingLean.Logic.PSR
import HeytingLean.Logic.Dialectic
import HeytingLean.Ontology.Primordial
import HeytingLean.Bridges.Tensor
import HeytingLean.Bridges.Graph
import HeytingLean.Bridges.Clifford
import HeytingLean.LoF.HeytingCore

open HeytingLean.LoF
open HeytingLean.Ontology
open HeytingLean.Bridges

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

theorem process_counter_disjoint (R : Reentry α) :
    R.process ⊓ R.counterProcess = ⊥ :=
  Reentry.process_inf_counter (R := R)

theorem euler_boundary_equals_process (R : Reentry α) :
    R.eulerBoundary = R.process :=
  Reentry.eulerBoundary_eq_process (R := R)

theorem euler_boundary_le_counter (R : Reentry α) :
    R.eulerBoundary ≤ R.counterProcess :=
  Reentry.eulerBoundary_le_counter (R := R)

theorem theta_cycle_zero_sum (θ : ℝ) :
    (thetaCycle θ).1 + (thetaCycle θ).2 = (0 : ℂ) :=
  thetaCycle_zero_sum θ

theorem tensor_encode_euler (R : Reentry α) (n : ℕ) :
    Bridges.Tensor.Model.encode (Contracts.Examples.tensor (α := α) (R := R) n) R.eulerBoundary
      = fun _ => R.primordial :=
  Bridges.Tensor.Model.eulerBoundary_vector (Contracts.Examples.tensor (α := α) (R := R) n)

theorem graph_encode_euler (R : Reentry α) :
    Bridges.Graph.Model.encode (Contracts.Examples.graph (α := α) (R := R)) R.eulerBoundary
      = R.primordial :=
  Bridges.Graph.Model.encode_eulerBoundary (Contracts.Examples.graph (α := α) (R := R))

theorem clifford_encode_euler (R : Reentry α) :
    Bridges.Clifford.Model.encode (Contracts.Examples.clifford (α := α) (R := R)) R.eulerBoundary
      = (R.primordial, R.primordial) := by
  classical
  let M := Contracts.Examples.clifford (α := α) (R := R)
  have hfst := Bridges.Clifford.Model.encode_eulerBoundary_fst (M := M)
  have hsnd := Bridges.Clifford.Model.encode_eulerBoundary_snd (M := M)
  change Bridges.Clifford.Model.encode M R.eulerBoundary = _
  ext
  · simpa [M] using hfst
  · simpa [M] using hsnd

theorem boolean_limit_verified (R : Reentry α) (h : ∀ a : α, R a = a) (a : α) :
    R (((_root_.HeytingLean.LoF.Reentry.booleanEquiv (R := R) h).symm a) : R.Omega) = a :=
  _root_.HeytingLean.LoF.Reentry.boolean_limit (R := R) h a

theorem mv_add_bottom_verified (P : Logic.Modal.DialParam α)
    (a : P.dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.mvAdd (P := P) ⊥ a = a := by
  simp [HeytingLean.Logic.Stage.DialParam.mvAdd]

theorem effect_add_bottom_verified (P : Logic.Modal.DialParam α)
    (a : P.dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectAdd? (P := P) ⊥ a = some a := by
  classical
  simp [HeytingLean.Logic.Stage.DialParam.effectAdd?,
    HeytingLean.Logic.Stage.DialParam.effectCompatible,
    HeytingLean.Logic.Stage.DialParam.mvAdd]

theorem orthocomplement_disjoint_verified
    (P : Logic.Modal.DialParam α) (a : P.dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectCompatible (P := P) a
        (HeytingLean.Logic.Stage.DialParam.orthocomplement (P := P) a) := by
  unfold HeytingLean.Logic.Stage.DialParam.effectCompatible
    HeytingLean.Logic.Stage.DialParam.orthocomplement
    HeytingLean.Logic.Stage.DialParam.mvNeg
  apply le_antisymm
  · have h :=
      HeytingLean.Logic.double_neg_collapse (R := P.dial.core) (a := a)
    change a ⊓ (a ⇨ (⊥ : P.dial.core.Omega)) ≤ (⊥ : _) at h
    exact h
  · exact bot_le

theorem tensor_shadow_mv_add (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageMvAdd
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b) := by
  classical
  simp [Contracts.Examples.tensor]

theorem tensor_shadow_effect_add (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.stageEffectAdd?
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode b)).map
      (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b).map
          (fun u => (u : α)) := by
  classical
  simpa [Contracts.Examples.tensor]
    using Bridges.Tensor.Model.logicalShadow_stageEffectAdd_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (a := a) (b := b)

theorem graph_shadow_mv_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageMvAdd
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a)
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b) := by
  classical
  simpa [Contracts.Examples.graph]
    using Bridges.Graph.Model.logicalShadow_stageMvAdd_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (a := a) (b := b)

theorem graph_shadow_effect_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Graph.Model.stageEffectAdd?
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a)
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode b)).map
      (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b).map
          (fun u => (u : α)) := by
  classical
  simpa [Contracts.Examples.graph]
    using Bridges.Graph.Model.logicalShadow_stageEffectAdd_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (a := a) (b := b)

theorem clifford_shadow_mv_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageMvAdd
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b) := by
  classical
  simpa [Contracts.Examples.clifford]
    using Bridges.Clifford.Model.logicalShadow_stageMvAdd_encode
      (M := Contracts.Examples.clifford (α := α) (R := R))
      (a := a) (b := b)

theorem clifford_shadow_effect_add (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Clifford.Model.stageEffectAdd?
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode b)).map
      (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base R) a b).map
          (fun u => (u : α)) := by
  classical
  simpa [Contracts.Examples.clifford]
    using Bridges.Clifford.Model.logicalShadow_stageEffectAdd_encode
      (M := Contracts.Examples.clifford (α := α) (R := R))
      (a := a) (b := b)

theorem tensor_shadow_mv_add_stage (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageMvAddAt
        (Contracts.Examples.tensor (α := α) (R := R) n)
        (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2)
        (by
          simpa [Contracts.Examples.tensor]
            using
              HeytingLean.Logic.Modal.DialParam.ladder_core
                (α := α) R 2)
        ((Bridges.Tensor.Model.contract
            (Contracts.Examples.tensor (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract
            (Contracts.Examples.tensor (α := α) (R := R) n)).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAddAt
            (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2)
            (h := by
              simpa [Contracts.Examples.tensor]
                using
                HeytingLean.Logic.Modal.DialParam.ladder_core
                  (α := α) R 2)
            a b) := by
  classical
  set P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2 with hP
  have hCore : P.dial.core = R := by
    simpa [hP] using
      HeytingLean.Logic.Modal.DialParam.ladder_core (α := α) R 2
  have := Bridges.Tensor.Model.logicalShadow_stageMvAddAt_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (P := P) (h := hCore) (a := a) (b := b)
  simpa [P, hP, hCore, Contracts.Examples.tensor]
    using this

theorem tensor_shadow_effect_add_stage (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.stageEffectAddAt?
        (Contracts.Examples.tensor (α := α) (R := R) n)
        (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3)
        (by
          simpa [Contracts.Examples.tensor]
            using
              HeytingLean.Logic.Modal.DialParam.ladder_core
                (α := α) R 3)
        ((Bridges.Tensor.Model.contract
            (Contracts.Examples.tensor (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract
            (Contracts.Examples.tensor (α := α) (R := R) n)).encode b)).map
      (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAddAt?
            (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3)
            (h := by
              simpa [Contracts.Examples.tensor]
                using
                HeytingLean.Logic.Modal.DialParam.ladder_core
                  (α := α) R 3)
            a b).map
          (fun u => (u : α)) := by
  classical
  set P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3 with hP
  have hCore : P.dial.core = R := by
    simpa [hP] using
      HeytingLean.Logic.Modal.DialParam.ladder_core (α := α) R 3
  have := Bridges.Tensor.Model.logicalShadow_stageEffectAddAt_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (P := P) (h := hCore) (a := a) (b := b)
  simpa [P, hP, hCore, Contracts.Examples.tensor]
    using this

theorem tensor_shadow_orthocomplement_stage (R : Reentry α) (n : ℕ)
    (a : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageOrthocomplementAt
        (Contracts.Examples.tensor (α := α) (R := R) n)
        (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4)
        (by
          simpa [Contracts.Examples.tensor]
            using
              HeytingLean.Logic.Modal.DialParam.ladder_core
                (α := α) R 4)
        ((Bridges.Tensor.Model.contract
            (Contracts.Examples.tensor (α := α) (R := R) n)).encode a))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.orthocomplementAt
            (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4)
            (h := by
              simpa [Contracts.Examples.tensor]
                using
                HeytingLean.Logic.Modal.DialParam.ladder_core
                  (α := α) R 4)
            a) := by
  classical
  set P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4 with hP
  have hCore : P.dial.core = R := by
    simpa [hP] using
      HeytingLean.Logic.Modal.DialParam.ladder_core (α := α) R 4
  have := Bridges.Tensor.Model.logicalShadow_stageOrthocomplementAt_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (P := P) (h := hCore) (a := a)
  simpa [P, hP, hCore, Contracts.Examples.tensor]
    using this

theorem tensor_shadow_mv_add_mv (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageMvAddMv
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.mvParam R) a b) := by
  classical
  simpa [Contracts.Examples.tensor]
    using Bridges.Tensor.Model.stageMvAddMv_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (a := a) (b := b)

theorem tensor_shadow_effect_add_effect (R : Reentry α) (n : ℕ)
    (a b : R.Omega) :
    (Bridges.Tensor.Model.stageEffectAddEffect?
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode b)).map
      (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.effectParam R) a b).map
          (fun u => (u : α)) := by
  classical
  simpa [Contracts.Examples.tensor]
    using Bridges.Tensor.Model.stageEffectAddEffect_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (a := a) (b := b)

theorem tensor_shadow_orthocomplement_orthomodular (R : Reentry α)
    (n : ℕ) (a : R.Omega) :
    (Bridges.Tensor.Model.logicalShadow
        (Contracts.Examples.tensor (α := α) (R := R) n))
      (Bridges.Tensor.Model.stageOrthocomplementOrthomodular
        (Contracts.Examples.tensor (α := α) (R := R) n)
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor
            (α := α) (R := R) n)).encode a))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.orthocomplement
            (P := HeytingLean.Logic.Modal.DialParam.orthomodularParam R) a) := by
  classical
  simpa [Contracts.Examples.tensor]
    using Bridges.Tensor.Model.stageOrthocomplementOrthomodular_encode
      (M := Contracts.Examples.tensor (α := α) (R := R) n)
      (a := a)

theorem graph_shadow_mv_add_mv (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageMvAddMv
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a)
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.mvParam R) a b) := by
  classical
  simpa [Contracts.Examples.graph]
    using Bridges.Graph.Model.stageMvAddMv_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (a := a) (b := b)

theorem graph_shadow_effect_add_effect (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Graph.Model.stageEffectAddEffect?
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a)
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode b)).map
      (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.effectParam R) a b).map
          (fun u => (u : α)) := by
  classical
  simpa [Contracts.Examples.graph]
    using Bridges.Graph.Model.stageEffectAddEffect_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (a := a) (b := b)

theorem graph_shadow_orthocomplement_orthomodular (R : Reentry α)
    (a : R.Omega) :
    (Bridges.Graph.Model.logicalShadow
        (Contracts.Examples.graph (α := α) (R := R)))
      (Bridges.Graph.Model.stageOrthocomplementOrthomodular
        (Contracts.Examples.graph (α := α) (R := R))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph
            (α := α) (R := R))).encode a))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.orthocomplement
            (P := HeytingLean.Logic.Modal.DialParam.orthomodularParam R) a) := by
  classical
  simpa [Contracts.Examples.graph]
    using Bridges.Graph.Model.stageOrthocomplementOrthomodular_encode
      (M := Contracts.Examples.graph (α := α) (R := R))
      (a := a)

theorem clifford_shadow_mv_add_mv (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageMvAddMv
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode b))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.mvParam R) a b) := by
  classical
  simpa [Contracts.Examples.clifford]
    using Bridges.Clifford.Model.stageMvAddMv_encode
      (M := Contracts.Examples.clifford (α := α) (R := R))
      (a := a) (b := b)

theorem clifford_shadow_effect_add_effect (R : Reentry α)
    (a b : R.Omega) :
    (Bridges.Clifford.Model.stageEffectAddEffect?
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode b)).map
      (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.effectParam R) a b).map
          (fun u => (u : α)) := by
  classical
  simpa [Contracts.Examples.clifford]
    using Bridges.Clifford.Model.stageEffectAddEffect_encode
      (M := Contracts.Examples.clifford (α := α) (R := R))
      (a := a) (b := b)

theorem clifford_shadow_orthocomplement_orthomodular (R : Reentry α)
    (a : R.Omega) :
    (Bridges.Clifford.Model.logicalShadow
        (Contracts.Examples.clifford (α := α) (R := R)))
      (Bridges.Clifford.Model.stageOrthocomplementOrthomodular
        (Contracts.Examples.clifford (α := α) (R := R))
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a))
      =
        R
          (HeytingLean.Logic.Stage.DialParam.orthocomplement
            (P := HeytingLean.Logic.Modal.DialParam.orthomodularParam R) a) := by
  classical
  simpa [Contracts.Examples.clifford]
    using Bridges.Clifford.Model.stageOrthocomplementOrthomodular_encode
      (M := Contracts.Examples.clifford (α := α) (R := R))
      (a := a)

theorem ladder_stage_boolean (R : Reentry α) :
    (Logic.Modal.DialParam.booleanParam (α := α) R).stage
      = Logic.Modal.DialParam.Stage.boolean :=
  Logic.Modal.DialParam.booleanParam_stage (α := α) R

theorem ladder_stage_effect (R : Reentry α) :
    (Logic.Modal.DialParam.effectParam (α := α) R).stage
      = Logic.Modal.DialParam.Stage.effect :=
  Logic.Modal.DialParam.effectParam_stage (α := α) R

theorem mv_param_add_verified (R : Reentry α)
    (a b : R.Omega) :
    Logic.Stage.DialParam.mvAdd
        (P := Logic.Modal.DialParam.mvParam (α := α) R) a b
      = a ⊔ b :=
  Logic.Stage.DialParam.mv_mvAdd (α := α) (R := R) a b

theorem orthomodular_param_complement_verified (R : Reentry α)
    (a : R.Omega) :
    Logic.Stage.DialParam.orthocomplement
        (P := Logic.Modal.DialParam.orthomodularParam (α := α) R) a
      = a ⇨ ⊥ :=
  Logic.Stage.DialParam.orthomodular_orthocomplement
    (α := α) (R := R) a

theorem occam_le_reentry_verified (R : Reentry α) (a : α) :
    Epistemic.occam (R := R) a ≤ R a :=
  Epistemic.occam_le_reentry (R := R) (a := a)

theorem occam_birth_verified (R : Reentry α) (a : α) :
    Epistemic.birth R (Epistemic.occam (R := R) a) = 0 :=
  Epistemic.occam_birth (R := R) (a := a)

theorem psr_stability_verified (R : Reentry α) {a x : α}
    (h : Logic.PSR.Sufficient R a) (hx : x ≤ a) :
    R x ≤ a :=
  Logic.PSR.sufficient_stable (R := R) h hx

theorem dialectic_synth_le_verified (R : Reentry α)
    (T A W : R.Omega) (hT : T ≤ W) (hA : A ≤ W) :
    Logic.Dialectic.synth (R := R) T A ≤ W :=
  Logic.Dialectic.synth_le (R := R) hT hA

end Tests
end HeytingLean
