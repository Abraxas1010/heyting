import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics

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

@[simp] lemma logicalShadow_encode' (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.encode a) = M.R a := by
  change M.logicalShadow (M.contract.encode a) = M.R a
  simpa using (logicalShadow_encode (M := M) (a := a))

@[simp] lemma decode_encode (M : Model α) (a : M.R.Omega) :
    M.decode (M.contract.encode a) = a := by
  change (M.contract.decode (M.contract.encode a)) = a
  simpa using M.contract.round a
lemma encode_eulerBoundary_fst (M : Model α) :
    (M.encode M.R.eulerBoundary).1 = M.R.primordial := by
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

lemma encode_eulerBoundary_snd (M : Model α) :
    (M.encode M.R.eulerBoundary).2 = M.R.primordial := by
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

/-- Stage-style MV addition lifted to the Clifford carrier at dial parameter `P`. -/
noncomputable def stageMvAddAt (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) :
    (α × α) → (α × α) → α × α :=
  fun p q =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.mvAddAt
        (P := P) (h := h) (M.decode p) (M.decode q))

/-- Stage-style effect compatibility on the Clifford carrier at dial parameter `P`. -/
def stageEffectCompatibleAt (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (p q : α × α) : Prop :=
  HeytingLean.Logic.Stage.DialParam.effectCompatibleAt
    (P := P) (h := h) (M.decode p) (M.decode q)

/-- Stage-style partial effect addition on the Clifford carrier at dial parameter `P`. -/
noncomputable def stageEffectAddAt?
    (M : Model α) (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (p q : α × α) : Option (α × α) :=
  (HeytingLean.Logic.Stage.DialParam.effectAddAt?
      (P := P) (h := h) (M.decode p) (M.decode q)).map M.encode

/-- Stage-style orthocomplement on the Clifford carrier at dial parameter `P`. -/
noncomputable def stageOrthocomplementAt (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) :
    (α × α) → (α × α) :=
  fun p =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.orthocomplementAt
        (P := P) (h := h) (M.decode p))

/-- MV-stage addition obtained from the canonical ladder parameter. -/
noncomputable def stageMvAddMv (M : Model α) :
    (α × α) → (α × α) → α × α :=
  M.stageMvAddAt
    (HeytingLean.Logic.Modal.DialParam.mvParam M.R)
    (by
      simpa using
        HeytingLean.Logic.Modal.DialParam.mvParam_core
          (α := α) (R := M.R))

/-- Effect-stage partial addition obtained from the canonical ladder parameter. -/
noncomputable def stageEffectAddEffect?
    (M : Model α) :
    (α × α) → (α × α) → Option (α × α) :=
  M.stageEffectAddAt?
    (HeytingLean.Logic.Modal.DialParam.effectParam M.R)
    (by
      simpa using
        HeytingLean.Logic.Modal.DialParam.effectParam_core
          (α := α) (R := M.R))

/-- Orthomodular-stage orthocomplement obtained from the canonical ladder parameter. -/
noncomputable def stageOrthocomplementOrthomodular
    (M : Model α) : (α × α) → (α × α) :=
  M.stageOrthocomplementAt
    (HeytingLean.Logic.Modal.DialParam.orthomodularParam M.R)
    (by
      simpa using
        HeytingLean.Logic.Modal.DialParam.orthomodularParam_core
          (α := α) (R := M.R))

@[simp] lemma baseCore_eq (M : Model α) :
    (HeytingLean.Logic.Modal.DialParam.base M.R).dial.core = M.R := by
  simpa using HeytingLean.Logic.Modal.DialParam.base_core (R := M.R)

/-- Base-stage MV addition (Boolean dial) for backward compatibility. -/
noncomputable def stageMvAdd (M : Model α) :
    (α × α) → (α × α) → α × α :=
  M.stageMvAddAt
    (HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.baseCore_eq)

/-- Base-stage effect compatibility for backward compatibility. -/
def stageEffectCompatible (M : Model α) (p q : α × α) : Prop :=
  M.stageEffectCompatibleAt
    (HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.baseCore_eq) p q

/-- Base-stage partial effect addition for backward compatibility. -/
noncomputable def stageEffectAdd?
    (M : Model α) (p q : α × α) : Option (α × α) :=
  M.stageEffectAddAt?
    (HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.baseCore_eq) p q

/-- Base-stage orthocomplement for backward compatibility. -/
noncomputable def stageOrthocomplement (M : Model α) :
    (α × α) → (α × α) :=
  M.stageOrthocomplementAt
    (HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.baseCore_eq)

variable {α : Type u} [PrimaryAlgebra α]

@[simp] lemma stageMvAddAt_encode (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R)
    (a b : M.R.Omega) :
    M.stageMvAddAt P h
        (M.contract.encode a) (M.contract.encode b)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.mvAddAt
            (P := P) (h := h) a b) := by
  classical
  simp [Model.stageMvAddAt, Model.decode_encode]

@[simp] lemma stageEffectCompatibleAt_encode (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (a b : M.R.Omega) :
    M.stageEffectCompatibleAt P h
        (M.contract.encode a) (M.contract.encode b) ↔
      HeytingLean.Logic.Stage.DialParam.effectCompatibleAt
        (P := P) (h := h) a b := by
  simp [Model.stageEffectCompatibleAt, Model.decode_encode]

@[simp] lemma stageEffectAddAt_encode (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (a b : M.R.Omega) :
    M.stageEffectAddAt? P h
        (M.contract.encode a) (M.contract.encode b)
      =
        (HeytingLean.Logic.Stage.DialParam.effectAddAt?
            (P := P) (h := h) a b).map M.encode := by
  classical
  simp [Model.stageEffectAddAt?, Model.decode_encode]

@[simp] lemma stageOrthocomplementAt_encode (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (a : M.R.Omega) :
    M.stageOrthocomplementAt P h (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.orthocomplementAt
            (P := P) (h := h) a) := by
  classical
  simp [Model.stageOrthocomplementAt, Model.decode_encode]

@[simp] lemma stageMvAddMv_encode (M : Model α) (a b : M.R.Omega) :
    M.stageMvAddMv
        (M.contract.encode a) (M.contract.encode b)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.mvParam M.R) a b) := by
  classical
  simp [Model.stageMvAddMv, Model.stageMvAddAt_encode]

@[simp] lemma stageEffectAddEffect_encode
    (M : Model α) (a b : M.R.Omega) :
    M.stageEffectAddEffect?
        (M.contract.encode a) (M.contract.encode b)
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.effectParam M.R) a b).map
          M.encode := by
  classical
  have h :=
    M.stageEffectAddAt_encode
      (P := HeytingLean.Logic.Modal.DialParam.effectParam M.R)
      (h := by
        simpa using
          HeytingLean.Logic.Modal.DialParam.effectParam_core
            (α := α) (R := M.R))
      (a := a) (b := b)
  simpa [Model.stageEffectAddEffect?, Model.stageEffectAddAt?,
        HeytingLean.Logic.Modal.DialParam.effectParam_core,
        HeytingLean.Logic.Stage.DialParam.effectAddAt_eq_effectAdd,
        HeytingLean.Logic.Stage.DialParam.toCore_rfl]
    using h

@[simp] lemma stageOrthocomplementOrthomodular_encode
    (M : Model α) (a : M.R.Omega) :
    M.stageOrthocomplementOrthomodular (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.orthocomplement
            (P := HeytingLean.Logic.Modal.DialParam.orthomodularParam M.R) a) := by
  classical
  simp [Model.stageOrthocomplementOrthomodular,
    Model.stageOrthocomplementAt_encode]

@[simp] lemma logicalShadow_stageMvAddAt_encode (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (a b : M.R.Omega) :
    M.logicalShadow
        (M.stageMvAddAt P h
          (M.contract.encode a) (M.contract.encode b))
      =
        M.R
          (HeytingLean.Logic.Stage.DialParam.mvAddAt
            (P := P) (h := h) a b) := by
  classical
  simp [Model.stageMvAddAt, Model.logicalShadow_encode',
    Model.decode_encode]

lemma logicalShadow_stageEffectAddAt_encode (M : Model α)
    (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (a b : M.R.Omega) :
    (M.stageEffectAddAt? P h
        (M.contract.encode a) (M.contract.encode b)).map
      M.logicalShadow
      =
        (HeytingLean.Logic.Stage.DialParam.effectAddAt?
            (P := P) (h := h) a b).map
          (fun u => (u : α)) := by
  classical
  unfold Model.stageEffectAddAt?
  cases hAdd :
      HeytingLean.Logic.Stage.DialParam.effectAddAt?
        (P := P) (h := h) a b with
  | none =>
      simp [hAdd, Model.logicalShadow_encode']
  | some u =>
      simp [hAdd, Model.logicalShadow_encode',
        Reentry.Omega.apply_coe]

@[simp] lemma logicalShadow_stageOrthocomplementAt_encode
    (M : Model α) (P : HeytingLean.Logic.Modal.DialParam α)
    (h : P.dial.core = M.R) (a : M.R.Omega) :
    M.logicalShadow
        (M.stageOrthocomplementAt P h
          (M.contract.encode a))
      =
        M.R
          (HeytingLean.Logic.Stage.DialParam.orthocomplementAt
            (P := P) (h := h) a) := by
  classical
  simp [Model.stageOrthocomplementAt, Model.logicalShadow_encode']

@[simp] lemma stageMvAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.stageMvAdd
        (M.contract.encode a) (M.contract.encode b)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b) := by
  classical
  simp [Model.stageMvAdd, Model.stageMvAddAt, Model.baseCore_eq,
    Model.decode_encode, HeytingLean.Logic.Stage.DialParam.mvAddAt,
    HeytingLean.Logic.Stage.DialParam.mvAdd]

@[simp] lemma stageEffectCompatible_encode (M : Model α) (a b : M.R.Omega) :
    M.stageEffectCompatible
        (M.contract.encode a) (M.contract.encode b) ↔
      HeytingLean.Logic.Stage.DialParam.effectCompatible
        (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b := by
  classical
  simp [Model.stageEffectCompatible, Model.stageEffectCompatibleAt,
    Model.baseCore_eq, Model.decode_encode,
    HeytingLean.Logic.Stage.DialParam.effectCompatibleAt,
    HeytingLean.Logic.Stage.DialParam.effectCompatible]

@[simp] lemma stageEffectAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.stageEffectAdd?
        (M.contract.encode a) (M.contract.encode b)
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b).map
          M.encode := by
  classical
  simp [Model.stageEffectAdd?, Model.stageEffectAddAt?, Model.baseCore_eq,
    Model.decode_encode, HeytingLean.Logic.Stage.DialParam.effectAddAt?,
    HeytingLean.Logic.Stage.DialParam.effectAdd?]

@[simp] lemma stageOrthocomplement_encode (M : Model α) (a : M.R.Omega) :
    M.stageOrthocomplement (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.orthocomplement
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a) := by
  classical
  simp [Model.stageOrthocomplement, Model.stageOrthocomplementAt,
    Model.baseCore_eq, Model.decode_encode,
    HeytingLean.Logic.Stage.DialParam.orthocomplementAt,
    HeytingLean.Logic.Stage.DialParam.orthocomplement]

@[simp] lemma logicalShadow_stageMvAdd_encode (M : Model α) (a b : M.R.Omega) :
    M.logicalShadow
        (M.stageMvAdd (M.contract.encode a) (M.contract.encode b))
      =
        M.R
          (HeytingLean.Logic.Stage.DialParam.mvAdd
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b) := by
  classical
  simp [Model.stageMvAdd, Model.stageMvAddAt, Model.baseCore_eq,
    Model.decode_encode, Model.logicalShadow_encode',
    HeytingLean.Logic.Stage.DialParam.mvAddAt,
    HeytingLean.Logic.Stage.DialParam.mvAdd]

lemma logicalShadow_stageEffectAdd_encode (M : Model α) (a b : M.R.Omega) :
    (M.stageEffectAdd?
        (M.contract.encode a) (M.contract.encode b)).map M.logicalShadow
      =
        (HeytingLean.Logic.Stage.DialParam.effectAdd?
            (P := HeytingLean.Logic.Modal.DialParam.base M.R) a b).map
          (fun u => (u : α)) := by
  classical
  unfold Model.stageEffectAdd?
  have h :=
    M.logicalShadow_stageEffectAddAt_encode
      (HeytingLean.Logic.Modal.DialParam.base M.R)
      (M.baseCore_eq) (a := a) (b := b)
  simpa [Model.stageEffectAddAt?, Model.baseCore_eq,
    HeytingLean.Logic.Stage.DialParam.effectAddAt?,
    HeytingLean.Logic.Stage.DialParam.effectAdd?,
    Model.decode_encode, Model.logicalShadow_encode']
    using h

@[simp] lemma logicalShadow_stageOrthocomplement_encode
    (M : Model α) (a : M.R.Omega) :
    M.logicalShadow (M.stageOrthocomplement (M.contract.encode a)) =
      M.R
        (HeytingLean.Logic.Stage.DialParam.orthocomplement
          (P := HeytingLean.Logic.Modal.DialParam.base M.R) a) := by
  classical
  simp [Model.stageOrthocomplement, Model.stageOrthocomplementAt,
    Model.baseCore_eq, Model.decode_encode, Model.logicalShadow_encode',
    HeytingLean.Logic.Stage.DialParam.orthocomplementAt,
    HeytingLean.Logic.Stage.DialParam.orthocomplement]

end Model

end

end Clifford
end Bridges
end HeytingLean
