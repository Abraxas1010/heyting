import HeytingLean.Contracts.Examples
import HeytingLean.Logic.ModalDial
import HeytingLean.Logic.Triad
import HeytingLean.Logic.ResiduatedLadder
import HeytingLean.Logic.StageSemantics
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

theorem tensor_rt2_verified (R : Reentry α) (n : ℕ) (a : R.Omega) :
    (Contracts.Examples.tensor (α := α) (R := R) n).logicalShadow
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode a)
      = R a :=
  tensor_shadow_verified (R := R) (n := n) (a := a)

theorem tensor_round_verified (R : Reentry α) (n : ℕ) (a : R.Omega) :
    (Bridges.Tensor.Model.contract (Contracts.Examples.tensor (α := α) (R := R) n)).decode
        ((Bridges.Tensor.Model.contract (Contracts.Examples.tensor (α := α) (R := R) n)).encode a)
      = a := by
  classical
  simpa [Contracts.Examples.tensor]
    using Contracts.Examples.tensor_round (α := α) (R := R) (n := n) (a := a)

theorem graph_shadow_verified (R : Reentry α) (a : R.Omega) :
    (Bridges.Graph.Model.logicalShadow (Contracts.Examples.graph (α := α) (R := R)))
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph (α := α) (R := R))).encode a)
        = R a :=
  Contracts.Examples.graph_shadow (α := α) (R := R) a

theorem graph_rt2_verified (R : Reentry α) (a : R.Omega) :
    (Contracts.Examples.graph (α := α) (R := R)).logicalShadow
        ((Contracts.Examples.graph (α := α) (R := R)).contract.encode a)
      = R a :=
  graph_shadow_verified (R := R) (a := a)

/-- Triangle (TRI-1): deduction, abduction, and induction coincide on the Heyting core. -/
theorem residuated_triangle_verified (R : Reentry α)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R) a b c ↔
      HeytingLean.Logic.Residuated.induction (R := R) a b c :=
  HeytingLean.Logic.Residuated.abduction_iff_induction (R := R) a b c

/-- Triangle (TRI-2) for the tensor bridge reduces to the core triangle via RT. -/
theorem tensor_triangle_lens_verified (R : Reentry α) (n : ℕ)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R)
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode a))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode b))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode c)) ↔
      HeytingLean.Logic.Residuated.induction (R := R)
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode a))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode b))
        ((Contracts.Examples.tensor (α := α) (R := R) n).contract.decode
          ((Contracts.Examples.tensor (α := α) (R := R) n).contract.encode c)) :=
by
  classical
  set M := Contracts.Examples.tensor (α := α) (R := R) n
  have ha := M.contract.round a
  have hb := M.contract.round b
  have hc := M.contract.round c
  simpa [M, ha, hb, hc] using residuated_triangle_verified (R := R) a b c

/-- Triangle (TRI-2) for the graph bridge reduces to the core triangle via the bridge contract. -/
theorem graph_triangle_lens_verified (R : Reentry α)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R)
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode c)) ↔
      HeytingLean.Logic.Residuated.induction (R := R)
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.graph (α := α) (R := R)).contract.decode
          ((Contracts.Examples.graph (α := α) (R := R)).contract.encode c)) :=
by
  classical
  set M := Contracts.Examples.graph (α := α) (R := R)
  have ha := M.contract.round a
  have hb := M.contract.round b
  have hc := M.contract.round c
  simpa [M, ha, hb, hc] using residuated_triangle_verified (R := R) a b c

/-- Triangle (TRI-2) for the Clifford bridge reduces to the core triangle via the bridge contract. -/
theorem clifford_triangle_lens_verified (R : Reentry α)
    (a b c : R.Omega) :
    HeytingLean.Logic.Residuated.abduction (R := R)
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode c)) ↔
      HeytingLean.Logic.Residuated.induction (R := R)
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode a))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode b))
        ((Contracts.Examples.clifford (α := α) (R := R)).contract.decode
          ((Contracts.Examples.clifford (α := α) (R := R)).contract.encode c)) :=
by
  classical
  set M := Contracts.Examples.clifford (α := α) (R := R)
  have ha := M.contract.round a
  have hb := M.contract.round b
  have hc := M.contract.round c
  simpa [M, ha, hb, hc] using residuated_triangle_verified (R := R) a b c

theorem graph_round_verified (R : Reentry α) (a : R.Omega) :
    (Bridges.Graph.Model.contract (Contracts.Examples.graph (α := α) (R := R))).decode
        ((Bridges.Graph.Model.contract (Contracts.Examples.graph (α := α) (R := R))).encode a)
      = a := by
  classical
  simpa [Contracts.Examples.graph]
    using Contracts.Examples.graph_round (α := α) (R := R) (a := a)

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

/-- Boolean-stage exemplar: MV addition with bottom is neutral. -/
theorem ladder_boolean_mv_zero (R : Reentry α)
    (a : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 0).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 0)
        (⊥) a = a := by
  simpa using
    (mv_add_bottom_verified
      (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 0)
      (a := a))

/-- MV-stage exemplar: addition commutes at the second ladder level. -/
theorem ladder_mv_comm (R : Reentry α)
    (a b : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2) a b =
      HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 2) b a := by
  classical
  simp [HeytingLean.Logic.Stage.DialParam.mvAdd, sup_comm]

/-- Effect-stage exemplar: adding an element to its orthocomplement is defined. -/
theorem ladder_effect_add_orthocomplement (R : Reentry α)
    (a : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectAdd?
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3)
        a
        (HeytingLean.Logic.Stage.DialParam.orthocomplement
          (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a)
      =
        some
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a
            (HeytingLean.Logic.Stage.DialParam.orthocomplement
              (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a)) := by
  classical
  have hCompat :=
    orthocomplement_disjoint_verified
      (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) (a := a)
  unfold HeytingLean.Logic.Stage.DialParam.effectAdd?
  by_cases h : HeytingLean.Logic.Stage.DialParam.effectCompatible
      (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a
      (HeytingLean.Logic.Stage.DialParam.orthocomplement
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 3) a)
  · simp [h, HeytingLean.Logic.Stage.DialParam.effectCompatible,
      HeytingLean.Logic.Stage.DialParam.orthocomplement,
      HeytingLean.Logic.Stage.DialParam.mvNeg]
  · exact (h hCompat).elim

/-- Orthomodular-stage exemplar: elements are disjoint from their orthocomplements. -/
theorem ladder_orthomodular_disjoint (R : Reentry α)
    (a : (HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4).dial.core.Omega) :
    HeytingLean.Logic.Stage.DialParam.effectCompatible
        (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4) a
        (HeytingLean.Logic.Stage.DialParam.orthocomplement
          (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4) a) :=
  orthocomplement_disjoint_verified
    (P := HeytingLean.Logic.Modal.DialParam.ladder (α := α) R 4) (a := a)

lemma tensor_shadow_mv_add (R : Reentry α) (n : ℕ)
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

lemma tensor_shadow_effect_add (R : Reentry α) (n : ℕ)
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

theorem clifford_round_verified (R : Reentry α) (a : R.Omega) :
    (Bridges.Clifford.Model.contract (Contracts.Examples.clifford (α := α) (R := R))).decode
        ((Bridges.Clifford.Model.contract (Contracts.Examples.clifford
            (α := α) (R := R))).encode a)
      = a := by
  classical
  simpa [Contracts.Examples.clifford]
    using Contracts.Examples.clifford_round (α := α) (R := R) (a := a)

end Tests
end HeytingLean
