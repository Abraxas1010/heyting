import HeytingLean.Contracts.RoundTrip
import HeytingLean.Logic.StageSemantics
import HeytingLean.Epistemic.Occam

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
open HeytingLean.Epistemic

universe u

section
variable (α : Type u) [PrimaryAlgebra α]

open scoped Classical

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
  exact logicalShadow_encode (M := M) (a := a)

@[simp] lemma decode_encode (M : Model α) (a : M.R.Omega) :
    M.decode (M.contract.encode a) = a := by
  change (M.contract.decode (M.contract.encode a)) = a
  exact M.contract.round a
lemma encode_eulerBoundary_fst (M : Model α) :
    (M.encode M.R.eulerBoundary).1 = M.R.primordial := by
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

lemma encode_eulerBoundary_snd (M : Model α) :
    (M.encode M.R.eulerBoundary).2 = M.R.primordial := by
  simp [Model.encode, Reentry.eulerBoundary_eq_process, Reentry.process_coe]

/-- Stage-style MV addition lifted to the Clifford carrier. -/
noncomputable def stageMvAdd (M : Model α) :
    (α × α) → (α × α) → α × α :=
  fun p q =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.mvAdd
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode p) (M.decode q))

/-- Stage-style effect compatibility on the Clifford carrier. -/
def stageEffectCompatible (M : Model α) (p q : α × α) : Prop :=
  HeytingLean.Logic.Stage.DialParam.effectCompatible
    (P := HeytingLean.Logic.Modal.DialParam.base M.R)
    (M.decode p) (M.decode q)

/-- Stage-style partial effect addition on the Clifford carrier. -/
noncomputable def stageEffectAdd?
    (M : Model α) (p q : α × α) : Option (α × α) :=
  (HeytingLean.Logic.Stage.DialParam.effectAdd?
      (P := HeytingLean.Logic.Modal.DialParam.base M.R)
      (M.decode p) (M.decode q)).map M.encode

/-- Stage-style orthocomplement lifted to the Clifford carrier. -/
noncomputable def stageOrthocomplement (M : Model α) :
    (α × α) → (α × α) :=
  fun p =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.orthocomplement
        (P := HeytingLean.Logic.Modal.DialParam.base M.R)
        (M.decode p))

/-- Stage-style Heyting implication lifted to the Clifford carrier. -/
noncomputable def stageHimp (M : Model α) :
    (α × α) → (α × α) → α × α :=
  fun p q =>
    M.encode ((M.decode p) ⇨ (M.decode q))

/-- Stage-style collapse (at ladder index `n`) on the Clifford carrier. -/
noncomputable def stageCollapseAt (M : Model α) (n : ℕ) :
    (α × α) → α × α :=
  fun p =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.collapseAtOmega
        (α := α) (R := M.R) n (M.decode p))

/-- Stage-style expansion (at ladder index `n`) on the Clifford carrier. -/
noncomputable def stageExpandAt (M : Model α) (n : ℕ) :
    (α × α) → α × α :=
  fun p =>
    M.encode
      (HeytingLean.Logic.Stage.DialParam.expandAtOmega
        (α := α) (R := M.R) n (M.decode p))

/-- Stage-style Occam reduction lifted to the Clifford carrier. -/
noncomputable def stageOccam (M : Model α) :
    (α × α) → α × α :=
  fun p =>
    let core : α := ((M.decode p : M.R.Omega) : α)
    M.encode
      (Reentry.Omega.mk (R := M.R)
        (Epistemic.occam (R := M.R) core)
        (Epistemic.occam_idempotent (R := M.R) (a := core)))

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

@[simp] lemma stageCollapseAt_encode (M : Model α) (n : ℕ)
    (a : M.R.Omega) :
    M.stageCollapseAt n (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.collapseAtOmega
            (α := α) (R := M.R) n a) := by
  classical
  simp [Model.stageCollapseAt, Model.decode_encode]

@[simp] lemma stageExpandAt_encode (M : Model α) (n : ℕ)
    (a : M.R.Omega) :
    M.stageExpandAt n (M.contract.encode a)
      =
        M.encode
          (HeytingLean.Logic.Stage.DialParam.expandAtOmega
            (α := α) (R := M.R) n a) := by
  classical
  simp [Model.stageExpandAt, Model.decode_encode]

@[simp] lemma stageOccam_encode (M : Model α) (a : M.R.Omega) :
    M.stageOccam (M.contract.encode a) =
      M.encode
        (Reentry.Omega.mk (R := M.R)
          (Epistemic.occam (R := M.R) (a : α))
          (Epistemic.occam_idempotent
            (R := M.R) (a := (a : α)))) := by
  classical
  simp [Model.stageOccam, Model.decode_encode]

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

@[simp] lemma logicalShadow_stageCollapseAt_encode
    (M : Model α) (n : ℕ) (a : M.R.Omega) :
    M.logicalShadow
        (M.stageCollapseAt n (M.contract.encode a)) =
      M.R
        (HeytingLean.Logic.Modal.DialParam.collapseAt
          (α := α) (R := M.R) n (a : α)) := by
  classical
  simp [stageCollapseAt_encode, Model.logicalShadow_encode']

@[simp] lemma logicalShadow_stageExpandAt_encode
    (M : Model α) (n : ℕ) (a : M.R.Omega) :
    M.logicalShadow
        (M.stageExpandAt n (M.contract.encode a)) =
      M.R
        (HeytingLean.Logic.Modal.DialParam.expandAt
          (α := α) (R := M.R) n (a : α)) := by
  classical
  simp [stageExpandAt_encode, Model.logicalShadow_encode']

@[simp] lemma logicalShadow_stageOccam_encode
    (M : Model α) (a : M.R.Omega) :
    M.logicalShadow
        (M.stageOccam (M.contract.encode a)) =
      Epistemic.occam (R := M.R) (a : α) := by
  classical
  simp [stageOccam_encode, Model.logicalShadow_encode',
    Epistemic.occam_idempotent]

end Model

end

end Clifford
end Bridges
end HeytingLean
