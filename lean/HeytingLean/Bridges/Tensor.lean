import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics

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

@[simp] lemma logicalShadow_encode' (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.encode a) = M.R a := by
  change M.logicalShadow (M.contract.encode a) = M.R a
  exact logicalShadow_encode (M := M) (a := a)

@[simp] lemma decode_encode (M : Model α) (a : M.R.Omega) :
    M.decode (M.contract.encode a) = a := by
  change (M.contract.decode (M.contract.encode a)) = a
  exact M.contract.round a
lemma eulerBoundary_vector (M : Model α) :
    M.encode M.R.eulerBoundary = fun _ => M.R.primordial := by
  classical
  funext i
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

def pointwiseMin (M : Model α) (v w : M.Carrier) : M.Carrier :=
  fun i => v i ⊓ w i

def pointwiseMax (M : Model α) (v w : M.Carrier) : M.Carrier :=
  fun i => v i ⊔ w i

@[simp] lemma encode_inf (M : Model α) (a b : M.R.Omega) :
    M.encode (a ⊓ b) = M.pointwiseMin (M.encode a) (M.encode b) := by
  classical
  funext i
  simp [encode, pointwiseMin]

/-- Stage-style MV addition lifted to the tensor carrier. -/
noncomputable def stageMvAdd (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun v w =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode v) (M.decode w))

/-- Stage-style effect compatibility viewed on the tensor carrier. -/
def stageEffectCompatible (M : Model α) (v w : M.Carrier) : Prop :=
  HeytingLean.Logic.Stage.DialParam.effectCompatible
    (P := HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.decode v) (M.decode w)

/-- Stage-style partial effect addition on the tensor carrier. -/
noncomputable def stageEffectAdd?
    (M : Model α) (v w : M.Carrier) : Option M.Carrier :=
  (HeytingLean.Logic.Stage.DialParam.effectAdd?
      (P := HeytingLean.Logic.Modal.DialParam.base M.R)
      (M.decode v) (M.decode w)).map M.encode

/-- Stage-style orthocomplement lifted to the tensor carrier. -/
noncomputable def stageOrthocomplement (M : Model α) :
    M.Carrier → M.Carrier :=
  fun v =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.orthocomplement
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode v))

/-- Stage-style Heyting implication lifted to the tensor carrier. -/
noncomputable def stageHimp (M : Model α) :
    M.Carrier → M.Carrier → M.Carrier :=
  fun v w =>
    M.encode ((M.decode v) ⇨ (M.decode w))

variable {α : Type u} [PrimaryAlgebra α]

@[simp] theorem stageMvAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.stageMvAdd
        (M.contract.encode a) (M.contract.encode b)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b) := by
  classical
  simp [Model.stageMvAdd, Model.decode_encode]

@[simp] theorem stageEffectCompatible_encode (M : Model α) (a b : M.R.Omega) :
    M.stageEffectCompatible
        (M.contract.encode a) (M.contract.encode b) ↔
      HeytingLean.Logic.Stage.DialParam.effectCompatible
        (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b := by
  simp [Model.stageEffectCompatible, Model.decode_encode]

@[simp] theorem stageEffectAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.stageEffectAdd?
        (M.contract.encode a) (M.contract.encode b)
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b).map
          M.encode := by
  classical
  simp [Model.stageEffectAdd?, Model.decode_encode]

@[simp] theorem stageOrthocomplement_encode (M : Model α) (a : M.R.Omega) :
    M.stageOrthocomplement (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.orthocomplement
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a) := by
  classical
  simp [Model.stageOrthocomplement, Model.decode_encode]

@[simp] lemma stageHimp_encode (M : Model α) (a b : M.R.Omega) :
    M.stageHimp
        (M.contract.encode a) (M.contract.encode b)
      =
        M.encode (a ⇨ b) := by
  classical
  simp [Model.stageHimp, Model.decode_encode]

@[simp] lemma logicalShadow_stageMvAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.logicalShadow
        (M.stageMvAdd (M.contract.encode a) (M.contract.encode b))
      =
        M.R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b) := by
  classical
  simp [stageMvAdd_encode, Model.logicalShadow_encode']

@[simp] theorem logicalShadow_stageEffectAdd_encode (M : Model α) (a b : M.R.Omega) :
    (M.stageEffectAdd?
        (M.contract.encode a) (M.contract.encode b)).map M.logicalShadow
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b).map
          (fun u => (u : α)) := by
  classical
  unfold Model.stageEffectAdd?
  cases h :
      HeytingLean.Logic.Stage.DialParam.effectAdd?
        (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b with
  | none =>
      simp [h]
  | some u =>
      simp [h, Model.logicalShadow_encode', Reentry.Omega.apply_coe]

@[simp] lemma logicalShadow_stageOrthocomplement_encode
    (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.stageOrthocomplement (M.contract.encode a)) =
      M.R
        (HeytingLean.Logic.Stage.DialParam.orthocomplement
          (P := HeytingLean.Logic.Modal.DialParam.base M.R) a) := by
  classical
  simp [stageOrthocomplement_encode, Model.logicalShadow_encode']

@[simp] lemma logicalShadow_stageHimp_encode
    (M : Model α) (a b : M.R.Omega) :
    M.logicalShadow
        (M.stageHimp (M.contract.encode a) (M.contract.encode b)) =
      M.R (a ⇨ b) := by
  classical
  simp [stageHimp_encode, Model.logicalShadow_encode']

end Model

end

end Tensor
end Bridges
end HeytingLean
