import HeytingLean.Logic.ModalDial
import HeytingLean.Logic.Triad
import HeytingLean.LoF.HeytingCore

/-
# Stage semantics transport

This module packages staged (MV / effect / orthomodular) operations on the Heyting core together
with a reusable transport interface for bridges.  The intent is to keep a single source of truth
for the operations on the core `Ω_R` and to expose helpers that automatically commute with the
round-trip data supplied by bridges.
-/

namespace HeytingLean
namespace Logic
namespace Stage

open HeytingLean.LoF
open scoped Classical

universe u

/-- Łukasiewicz / MV-style structure available on the core carrier. -/
class MvCore (Ω : Type u) where
  mvAdd : Ω → Ω → Ω
  mvNeg : Ω → Ω
  zero  : Ω
  one   : Ω

/-- Effect-algebra style structure with partial addition. -/
class EffectCore (Ω : Type u) where
  effectAdd? : Ω → Ω → Option Ω
  compat     : Ω → Ω → Prop
  orthosupp  : Ω → Ω
  zero       : Ω
  one        : Ω
  compat_iff_defined :
    ∀ u v, compat u v ↔ (effectAdd? u v).isSome

/-- Orthomodular lattice façade (no laws recorded yet, only operations). -/
class OmlCore (Ω : Type u) where
  meet : Ω → Ω → Ω
  join : Ω → Ω → Ω
  compl : Ω → Ω
  bot : Ω
  top : Ω

variable {α : Type u} [PrimaryAlgebra α]

namespace DialParam

variable (P : Modal.DialParam α)

/-- MV-style addition realised via the Heyting join. -/
@[simp] def mvAdd (a b : P.dial.core.Omega) : P.dial.core.Omega :=
  a ⊔ b

/-- MV-style negation realised via implication into bottom. -/
@[simp] def mvNeg (a : P.dial.core.Omega) : P.dial.core.Omega :=
  a ⇨ (⊥ : P.dial.core.Omega)

/-- MV-stage zero. -/
@[simp] def mvZero : P.dial.core.Omega := ⊥
/-- MV-stage one. -/
@[simp] def mvOne : P.dial.core.Omega := ⊤

/-- Effect-compatibility predicate (disjointness). -/
def effectCompatible (a b : P.dial.core.Omega) : Prop :=
  a ⊓ b = ⊥

/-- Partial effect-style addition returning a value only on compatible arguments. -/
noncomputable def effectAdd? (a b : P.dial.core.Omega) :
    Option P.dial.core.Omega :=
  if _ : DialParam.effectCompatible (P := P) a b then
    some (DialParam.mvAdd (P := P) a b)
  else
    none

/-- Orthocomplement induced by Heyting negation. -/
@[simp] def orthocomplement (a : P.dial.core.Omega) :
    P.dial.core.Omega :=
  DialParam.mvNeg (P := P) a

/-- Orthomodular meet (plain Heyting meet). -/
@[simp] def omlMeet (a b : P.dial.core.Omega) : P.dial.core.Omega := a ⊓ b

/-- Orthomodular join (plain Heyting join). -/
@[simp] def omlJoin (a b : P.dial.core.Omega) : P.dial.core.Omega := a ⊔ b

/-- Orthomodular bottom element. -/
@[simp] def omlBot : P.dial.core.Omega := ⊥
/-- Orthomodular top element. -/
@[simp] def omlTop : P.dial.core.Omega := ⊤

instance instMvCore : MvCore P.dial.core.Omega where
  mvAdd := DialParam.mvAdd (P := P)
  mvNeg := DialParam.mvNeg (P := P)
  zero := DialParam.mvZero (P := P)
  one := DialParam.mvOne (P := P)

noncomputable instance instEffectCore : EffectCore P.dial.core.Omega where
  effectAdd? := DialParam.effectAdd? (P := P)
  compat := DialParam.effectCompatible (P := P)
  orthosupp := DialParam.orthocomplement (P := P)
  zero := DialParam.mvZero (P := P)
  one := DialParam.mvOne (P := P)
  compat_iff_defined := by
    intro u v
    classical
    unfold DialParam.effectAdd? DialParam.effectCompatible
    by_cases h : u ⊓ v = (⊥ : P.dial.core.Omega)
    · simp [h]
    · simp [h]

instance instOmlCore : OmlCore P.dial.core.Omega where
  meet := DialParam.omlMeet (P := P)
  join := DialParam.omlJoin (P := P)
  compl := DialParam.orthocomplement (P := P)
  bot := DialParam.omlBot (P := P)
  top := DialParam.omlTop (P := P)

section Laws

/-- Collapse restricted to the Heyting core, producing a fixed point element. -/
noncomputable def collapseOmega :
    P.dial.core.Omega → P.dial.core.Omega :=
  fun a =>
    Reentry.Omega.mk (R := P.dial.core)
      (P.dial.collapse (a : α))
      (by
        unfold Modal.Dial.collapse
        simpa [Modal.Dial.collapse] using
          P.dial.core.idempotent (P.dial.diamond (a : α)))

@[simp] lemma collapseOmega_coe (a : P.dial.core.Omega) :
    ((DialParam.collapseOmega (P := P) a :
        P.dial.core.Omega) : α)
      = P.dial.collapse (a : α) := rfl

def toCore {R : LoF.Reentry α} (P : Modal.DialParam α)
    (h : P.dial.core = R) :
    R.Omega → P.dial.core.Omega := by
  cases h
  intro x
  exact x

def fromCore {R : LoF.Reentry α} (P : Modal.DialParam α)
    (h : P.dial.core = R) :
    P.dial.core.Omega → R.Omega := by
  cases h
  intro x
  exact x

@[simp] lemma toCore_rfl (P : Modal.DialParam α)
    (x : P.dial.core.Omega) :
    DialParam.toCore (P := P) rfl x = x := rfl

@[simp] lemma fromCore_rfl (P : Modal.DialParam α)
    (x : P.dial.core.Omega) :
    DialParam.fromCore (P := P) rfl x = x := rfl

@[simp] lemma fromCore_toCore (P : Modal.DialParam α)
    {R : LoF.Reentry α} (h : P.dial.core = R)
    (x : R.Omega) :
    DialParam.fromCore (P := P) h (DialParam.toCore (P := P) h x) = x := by
  cases h
  rfl

@[simp] lemma toCore_fromCore (P : Modal.DialParam α)
    {R : LoF.Reentry α} (h : P.dial.core = R)
    (x : P.dial.core.Omega) :
    DialParam.toCore (P := P) h (DialParam.fromCore (P := P) h x) = x := by
  cases h
  rfl

@[simp] lemma mvAdd_comm (a b : P.dial.core.Omega) :
    DialParam.mvAdd (P := P) a b =
      DialParam.mvAdd (P := P) b a := by
  simp [DialParam.mvAdd, sup_comm]

@[simp] lemma mvAdd_assoc (a b c : P.dial.core.Omega) :
    DialParam.mvAdd (P := P) (DialParam.mvAdd (P := P) a b) c =
      DialParam.mvAdd (P := P) a (DialParam.mvAdd (P := P) b c) := by
  simp [DialParam.mvAdd, sup_assoc]

@[simp] lemma mvAdd_zero_left (a : P.dial.core.Omega) :
    DialParam.mvAdd (P := P) (DialParam.mvZero (P := P)) a = a := by
  simp [DialParam.mvAdd, DialParam.mvZero]

@[simp] lemma mvAdd_zero_right (a : P.dial.core.Omega) :
    DialParam.mvAdd (P := P) a (DialParam.mvZero (P := P)) = a := by
  simp [DialParam.mvAdd, DialParam.mvZero]

lemma mvNeg_le_double (a : P.dial.core.Omega) :
    a ≤ DialParam.mvNeg (P := P)
      (DialParam.mvNeg (P := P) a) := by
  change a ≤ (a ⇨ (⊥ : _)) ⇨ (⊥ : _)
  simpa [DialParam.mvNeg] using
    Reentry.double_neg (R := P.dial.core) (a := a)

@[simp] lemma effectCompatible_comm (a b : P.dial.core.Omega) :
    DialParam.effectCompatible (P := P) a b ↔
      DialParam.effectCompatible (P := P) b a := by
  unfold DialParam.effectCompatible
  constructor <;> intro h <;> simpa [inf_comm] using h

lemma effectAdd?_of_compatible (a b : P.dial.core.Omega)
    (h : DialParam.effectCompatible (P := P) a b) :
    DialParam.effectAdd? (P := P) a b
      = some (DialParam.mvAdd (P := P) a b) := by
  classical
  unfold DialParam.effectAdd?
  have h' : DialParam.effectCompatible (P := P) a b := h
  simp [h']

lemma effectAdd?_of_not_compatible (a b : P.dial.core.Omega)
    (h : ¬ DialParam.effectCompatible (P := P) a b) :
    DialParam.effectAdd? (P := P) a b = none := by
  classical
  unfold DialParam.effectAdd?
  have h' : ¬ DialParam.effectCompatible (P := P) a b := h
  simp [h']

lemma effectAdd?_isSome_iff (a b : P.dial.core.Omega) :
    (DialParam.effectAdd? (P := P) a b).isSome ↔
      DialParam.effectCompatible (P := P) a b := by
  classical
  unfold DialParam.effectAdd?
  split_ifs with h
  · simp [h]
  · simp [h]

lemma orthocomplement_disjoint
    (a : P.dial.core.Omega) :
    DialParam.omlMeet (P := P) a
        (DialParam.orthocomplement (P := P) a)
      = ⊥ := by
  unfold DialParam.omlMeet DialParam.orthocomplement DialParam.mvNeg
  apply le_antisymm
  · have h :=
      HeytingLean.Logic.double_neg_collapse (R := P.dial.core) (a := a)
    change a ⊓ (a ⇨ (⊥ : _)) ≤ (⊥ : _) at h
    simpa using h
  · exact bot_le

lemma effectCompatible_orthocomplement
    (a : P.dial.core.Omega) :
    DialParam.effectCompatible (P := P) a
      (DialParam.orthocomplement (P := P) a) := by
  unfold DialParam.effectCompatible
  simpa using
    (orthocomplement_disjoint (P := P) (a := a))

@[simp] lemma orthocomplement_effectCompatible
    (a : P.dial.core.Omega) :
    DialParam.effectCompatible (P := P)
        (DialParam.orthocomplement (P := P) a) a := by
  unfold DialParam.effectCompatible
  simpa [inf_comm] using
    (orthocomplement_disjoint (P := P) (a := a))

end Laws

lemma collapseOmega_effectCompatible
    (P : Modal.DialParam α)
    (a : P.dial.core.Omega) :
    DialParam.effectCompatible (P := P)
      (DialParam.collapseOmega (P := P) a)
      (DialParam.orthocomplement (P := P)
        (DialParam.collapseOmega (P := P) a)) := by
  simpa using
    (effectCompatible_orthocomplement (P := P)
      (a := DialParam.collapseOmega (P := P) a))

section Transport

variable {R : LoF.Reentry α}
variable (P : Modal.DialParam α)
variable (h : P.dial.core = R)

/-- Transport MV addition along a proof that the dial core coincides with `R`. -/
@[simp] def mvAddAt (a b : R.Omega) : R.Omega :=
  DialParam.fromCore (P := P) h
    (DialParam.mvAdd (P := P)
      (DialParam.toCore (P := P) h a)
      (DialParam.toCore (P := P) h b))

/-- Transport MV negation along a proof that the dial core coincides with `R`. -/
@[simp] def mvNegAt (a : R.Omega) : R.Omega :=
  DialParam.fromCore (P := P) h
    (DialParam.mvNeg (P := P)
      (DialParam.toCore (P := P) h a))

/-- Transport MV zero along a proof that the dial core coincides with `R`. -/
@[simp] def mvZeroAt : R.Omega :=
  DialParam.fromCore (P := P) h
    (DialParam.mvZero (P := P))

/-- Transport MV one along a proof that the dial core coincides with `R`. -/
@[simp] def mvOneAt : R.Omega :=
  DialParam.fromCore (P := P) h
    (DialParam.mvOne (P := P))

/-- Transport effect compatibility along a proof that the dial core coincides with `R`. -/
def effectCompatibleAt (a b : R.Omega) : Prop :=
  DialParam.effectCompatible (P := P)
    (DialParam.toCore (P := P) h a)
    (DialParam.toCore (P := P) h b)

/-- Transport partial effect addition along a proof that the dial core coincides with `R`. -/
noncomputable def effectAddAt?
    (a b : R.Omega) : Option R.Omega :=
  (DialParam.effectAdd? (P := P)
      (DialParam.toCore (P := P) h a)
      (DialParam.toCore (P := P) h b)).map
    (DialParam.fromCore (P := P) h)

/-- Transport the orthocomplement along a proof that the dial core coincides with `R`. -/
@[simp] def orthocomplementAt (a : R.Omega) : R.Omega :=
  DialParam.fromCore (P := P) h
    (DialParam.orthocomplement (P := P)
      (DialParam.toCore (P := P) h a))

variable {P} {h}

lemma mvAddAt_eq (a b : R.Omega) :
    DialParam.mvAddAt (P := P) (h := h) a b =
      DialParam.fromCore (P := P) h
        (DialParam.mvAdd (P := P)
          (DialParam.toCore (P := P) h a)
          (DialParam.toCore (P := P) h b)) := rfl

lemma mvAddAt_comm (a b : R.Omega) :
    DialParam.mvAddAt (P := P) (h := h) a b =
      DialParam.mvAddAt (P := P) (h := h) b a := by
  cases h
  simp [DialParam.mvAddAt, DialParam.mvAdd, sup_comm]

lemma mvAddAt_assoc (a b c : R.Omega) :
    DialParam.mvAddAt (P := P) (h := h)
        (DialParam.mvAddAt (P := P) (h := h) a b) c
      =
        DialParam.mvAddAt (P := P) (h := h) a
          (DialParam.mvAddAt (P := P) (h := h) b c) := by
  cases h
  simp [DialParam.mvAddAt, DialParam.mvAdd, sup_assoc]

lemma mvAddAt_zero_left (a : R.Omega) :
    DialParam.mvAddAt (P := P) (h := h)
        (DialParam.mvZeroAt (P := P) (h := h)) a = a := by
  cases h
  simp [DialParam.mvAddAt, DialParam.mvZeroAt,
    DialParam.mvAdd, DialParam.mvZero]

lemma mvAddAt_zero_right (a : R.Omega) :
    DialParam.mvAddAt (P := P) (h := h) a
        (DialParam.mvZeroAt (P := P) (h := h)) = a := by
  cases h
  simp [DialParam.mvAddAt, DialParam.mvZeroAt,
    DialParam.mvAdd, DialParam.mvZero]

lemma effectCompatibleAt_comm (a b : R.Omega) :
    DialParam.effectCompatibleAt (P := P) (h := h) a b ↔
      DialParam.effectCompatibleAt (P := P) (h := h) b a := by
  cases h
  simp [DialParam.effectCompatibleAt,
    DialParam.effectCompatible_comm]

lemma effectAddAt?_map (a b : R.Omega) :
    (DialParam.effectAddAt? (P := P) (h := h) a b).map
        (DialParam.toCore (P := P) h) =
      DialParam.effectAdd? (P := P)
        (DialParam.toCore (P := P) h a)
        (DialParam.toCore (P := P) h b) := by
  cases h
  simp [DialParam.effectAddAt?, DialParam.effectAdd?,
    DialParam.effectCompatible]

lemma effectAddAt?_isSome (a b : R.Omega) :
    (DialParam.effectAddAt? (P := P) (h := h) a b).isSome ↔
      DialParam.effectCompatibleAt (P := P) (h := h) a b := by
  classical
  cases h
  simp [DialParam.effectAddAt?, DialParam.effectCompatibleAt,
    DialParam.effectAdd?_isSome_iff]

@[simp] lemma effectAddAt_eq_effectAdd (P : Modal.DialParam α)
    {R : LoF.Reentry α} (h : P.dial.core = R)
    (a b : R.Omega) :
    DialParam.effectAddAt? (P := P) (h := h) a b =
      DialParam.effectAdd? (P := P)
        (DialParam.toCore (P := P) h a)
        (DialParam.toCore (P := P) h b) := by
  cases h
  simp [DialParam.effectAddAt?, DialParam.effectAdd?]

lemma effectCompatibleAt_orthocomplement
    (a : R.Omega) :
    DialParam.effectCompatibleAt (P := P) (h := h) a
      (DialParam.orthocomplementAt (P := P) (h := h) a) := by
  cases h
  simpa [DialParam.effectCompatibleAt, DialParam.orthocomplementAt]
    using (DialParam.effectCompatible_orthocomplement
      (P := P) (a := a))

lemma orthocomplementAt_effectCompatible
    (a : R.Omega) :
    DialParam.effectCompatibleAt (P := P) (h := h)
        (DialParam.orthocomplementAt (P := P) (h := h) a) a := by
  cases h
  simpa [DialParam.effectCompatibleAt, DialParam.orthocomplementAt]
    using (DialParam.orthocomplement_effectCompatible
      (P := P) (a := a))

end Transport

section NamedStages

variable (R : LoF.Reentry α)

@[simp] lemma boolean_mvAdd (a b : R.Omega) :
    DialParam.mvAdd
        (P := Modal.DialParam.booleanParam (α := α) R) a b = a ⊔ b := by
  simp [Modal.DialParam.booleanParam, DialParam.mvAdd]

@[simp] lemma mv_mvAdd (a b : R.Omega) :
    DialParam.mvAdd
        (P := Modal.DialParam.mvParam (α := α) R) a b = a ⊔ b := by
  simp [Modal.DialParam.mvParam, DialParam.mvAdd]

@[simp] lemma mv_mvNeg (a : R.Omega) :
    DialParam.mvNeg
        (P := Modal.DialParam.mvParam (α := α) R) a = a ⇨ ⊥ := by
  simp [Modal.DialParam.mvParam, DialParam.mvNeg]

@[simp] lemma effect_effectAdd?_of_compatible
    (a b : R.Omega)
    (h : DialParam.effectCompatible
            (P := Modal.DialParam.effectParam (α := α) R) a b) :
    DialParam.effectAdd?
        (P := Modal.DialParam.effectParam (α := α) R) a b
      = some (a ⊔ b) := by
  simpa [Modal.DialParam.effectParam,
    DialParam.effectAdd?] using
      effectAdd?_of_compatible
        (P := Modal.DialParam.effectParam (α := α) R)
        (a := a) (b := b) h

@[simp] lemma orthomodular_orthocomplement (a : R.Omega) :
    DialParam.orthocomplement
        (P := Modal.DialParam.orthomodularParam (α := α) R) a
      = a ⇨ ⊥ := by
  simp [Modal.DialParam.orthomodularParam, DialParam.orthocomplement, DialParam.mvNeg]

end NamedStages

end DialParam

/-- Bridges expose shadow/lift data satisfying a round-trip contract. -/
structure Bridge (α Ω : Type u) [LE α] [LE Ω] where
  shadow : α → Ω
  lift : Ω → α
  rt₁ : ∀ u, shadow (lift u) = u
  rt₂ : ∀ x, lift (shadow x) ≤ x

namespace Bridge

variable {α Ω : Type u} [LE α] [LE Ω] (B : Bridge α Ω)

/-- Transport MV addition across a bridge. -/
def stageMvAdd [MvCore Ω] (x y : α) : α :=
  B.lift (MvCore.mvAdd (B.shadow x) (B.shadow y))

/-- Transport MV negation across a bridge. -/
def stageMvNeg [MvCore Ω] (x : α) : α :=
  B.lift (MvCore.mvNeg (B.shadow x))

/-- Transport MV zero. -/
def stageMvZero [MvCore Ω] : α :=
  B.lift (MvCore.zero (Ω := Ω))

/-- Transport MV one. -/
def stageMvOne [MvCore Ω] : α :=
  B.lift (MvCore.one (Ω := Ω))

/-- Transport effect compatibility across a bridge. -/
def stageEffectCompatible [EffectCore Ω] (x y : α) : Prop :=
  EffectCore.compat (B.shadow x) (B.shadow y)

/-- Transport partial effect addition across a bridge. -/
def stageEffectAdd? [EffectCore Ω] (x y : α) : Option α :=
  (EffectCore.effectAdd? (B.shadow x) (B.shadow y)).map B.lift

/-- Transport the effect orthosupplement across a bridge. -/
def stageOrthosupp [EffectCore Ω] (x : α) : α :=
  B.lift (EffectCore.orthosupp (B.shadow x))

/-- Transport orthocomplement across a bridge. -/
def stageOrthocomplement [OmlCore Ω] (x : α) : α :=
  B.lift (OmlCore.compl (B.shadow x))

@[simp] theorem shadow_stageMvAdd [MvCore Ω] (x y : α) :
    B.shadow (B.stageMvAdd x y) =
      MvCore.mvAdd (B.shadow x) (B.shadow y) := by
  unfold stageMvAdd
  simpa using B.rt₁ (MvCore.mvAdd (B.shadow x) (B.shadow y))

@[simp] theorem shadow_stageMvNeg [MvCore Ω] (x : α) :
    B.shadow (B.stageMvNeg x) = MvCore.mvNeg (B.shadow x) := by
  unfold stageMvNeg
  simpa using B.rt₁ (MvCore.mvNeg (B.shadow x))

@[simp] theorem stageEffectAdd?_isSome [EffectCore Ω] (x y : α) :
    (B.stageEffectAdd? x y).isSome ↔
      B.stageEffectCompatible x y := by
  unfold stageEffectAdd? stageEffectCompatible
  have h := EffectCore.compat_iff_defined (Ω := Ω)
  specialize h (B.shadow x) (B.shadow y)
  cases h' : EffectCore.effectAdd? (B.shadow x) (B.shadow y) with
  | none =>
      simp [Option.isSome, h', Option.map, h] at *
  | some w =>
      simp [Option.isSome, h', Option.map, h] at *

@[simp] theorem shadow_stageEffectAdd?_map [EffectCore Ω] (x y : α) :
    (B.stageEffectAdd? x y).map B.shadow =
      EffectCore.effectAdd? (B.shadow x) (B.shadow y) := by
  unfold stageEffectAdd?
  cases h : EffectCore.effectAdd? (B.shadow x) (B.shadow y) with
  | none =>
      simp
  | some w =>
      simp [B.rt₁]

@[simp] theorem shadow_stageOrthosupp [EffectCore Ω] (x : α) :
    B.shadow (B.stageOrthosupp x) =
      EffectCore.orthosupp (B.shadow x) := by
  unfold stageOrthosupp
  simpa using B.rt₁ (EffectCore.orthosupp (B.shadow x))

@[simp] theorem shadow_stageOrthocomplement [OmlCore Ω] (x : α) :
    B.shadow (B.stageOrthocomplement x) =
      OmlCore.compl (B.shadow x) := by
  unfold stageOrthocomplement
  simpa using B.rt₁ (OmlCore.compl (B.shadow x))

end Bridge

end Stage
end Logic
end HeytingLean
