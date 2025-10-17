import HeytingLean.LoF.Nucleus

/-!
# Modal dial on the Heyting core

A modal dial packages a core re-entry nucleus together with breathing operators `□` and `◇`
respecting the ordering hierarchy required in the roadmap.
-/

namespace HeytingLean
namespace Logic
namespace Modal
open HeytingLean.LoF

universe u

/-- A modal dial enriches a core nucleus with interior/exterior companions. -/
structure Dial (α : Type u) [PrimaryAlgebra α] where
  core : Reentry α
  box : Reentry α
  diamond : Reentry α
  box_le_core : ∀ a : α, box a ≤ core a
  core_le_diamond : ∀ a : α, core a ≤ diamond a
  box_le_diamond : ∀ a : α, box a ≤ diamond a

namespace Dial

variable {α : Type u} [PrimaryAlgebra α] (D : Dial α)

@[simp] lemma box_le_core_apply (a : α) : D.box a ≤ D.core a :=
  D.box_le_core a

@[simp] lemma core_le_diamond_apply (a : α) : D.core a ≤ D.diamond a :=
  D.core_le_diamond a

/-- The breathing cycle collapses a proposition by interiorising after an exterior move. -/
def collapse (a : α) : α :=
  D.core (D.diamond a)

/-- Collapse is monotone because it is a composition of monotone nuclei. -/
lemma collapse_monotone : Monotone D.collapse := by
  intro a b h
  unfold collapse
  have := D.diamond.monotone h
  exact D.core.monotone this

/-- The breathing cycle remains below the next exterior pass. -/
lemma collapse_le_next (a : α) : D.collapse a ≤ D.diamond (D.diamond a) := by
  unfold collapse
  exact D.core_le_diamond_apply _

@[simp] lemma box_le_diamond_apply (a : α) : D.box a ≤ D.diamond a :=
  D.box_le_diamond a

/-- A breathing cycle absorbs the boxed contribution. -/
lemma box_le_collapse (a : α) : D.box a ≤ D.collapse a := by
  unfold collapse
  have h₁ : D.box a ≤ D.diamond a := D.box_le_diamond_apply a
  have h₂ := D.box.monotone h₁
  have h₃ : D.box (D.diamond a) ≤ D.core (D.diamond a) :=
    D.box_le_core_apply _
  exact le_trans (by simpa using h₂) h₃

@[simp] lemma collapse_eq (a : α) : D.collapse a = D.core (D.diamond a) := rfl

@[simp] def trivial (R : Reentry α) : Dial α where
  core := R
  box := R
  diamond := R
  box_le_core := by intro; rfl
  core_le_diamond := by intro; rfl
  box_le_diamond := by intro; rfl

end Dial

/-- Dial parameters annotate the dimensional ladder for modal breathing. -/
structure DialParam (α : Type u) [PrimaryAlgebra α] where
  dimension : ℕ
  dial : Dial α

namespace DialParam

variable {α : Type u} [PrimaryAlgebra α] (P : DialParam α)

/-- Collapse interpreted at the parameter level. -/
def collapse : α → α :=
  P.dial.collapse

lemma collapse_monotone : Monotone P.collapse :=
  P.dial.collapse_monotone

/-- The breathing endpoints packaged as a pair. -/
def breathe (a : α) : α × α :=
  (P.dial.box a, P.dial.diamond a)

/-- Promote the parameter to the next dimensional layer. -/
def elevate : DialParam α :=
  { dimension := P.dimension + 1
    dial := P.dial }

@[simp] lemma elevate_dimension :
    P.elevate.dimension = P.dimension + 1 := rfl

@[simp] lemma elevate_collapse :
    P.elevate.collapse = P.collapse := rfl



def le (P Q : DialParam α) : Prop :=
  P.dimension ≤ Q.dimension ∧
    ∀ a, P.dial.box a ≤ Q.dial.box a ∧ P.dial.diamond a ≤ Q.dial.diamond a

@[simp] lemma le_refl : P.le P := by
  refine ⟨le_rfl, ?_⟩
  intro a
  exact ⟨le_rfl, le_rfl⟩

lemma le_trans {Q R' : DialParam α} (hPQ : P.le Q) (hQR : Q.le R') :
    P.le R' := by
  refine ⟨Nat.le_trans hPQ.1 hQR.1, ?_⟩
  intro a
  obtain ⟨hBoxPQ, hDiamondPQ⟩ := hPQ.2 a
  obtain ⟨hBoxQR, hDiamondQR⟩ := hQR.2 a
  exact ⟨_root_.le_trans hBoxPQ hBoxQR, _root_.le_trans hDiamondPQ hDiamondQR⟩

lemma le_elevate : P.le P.elevate := by
  refine ⟨Nat.le_succ _, ?_⟩
  intro a
  exact ⟨le_rfl, le_rfl⟩

def base (R : Reentry α) : DialParam α :=
  { dimension := 0
    dial := Dial.trivial R }

@[simp] lemma base_dimension (R : Reentry α) :
    (base R).dimension = 0 := rfl

@[simp] lemma base_core (R : Reentry α) :
    (base R).dial.core = R := rfl

@[simp] lemma elevate_core (P : DialParam α) :
    (P.elevate).dial.core = P.dial.core := rfl

def ladder (R : Reentry α) : ℕ → DialParam α
  | 0 => base R
  | n + 1 => (ladder R n).elevate

@[simp] lemma ladder_zero (R : Reentry α) :
    ladder R 0 = base R := rfl

@[simp] lemma ladder_succ (R : Reentry α) (n : ℕ) :
    ladder R (n + 1) = (ladder R n).elevate := rfl

@[simp] lemma ladder_dimension (R : Reentry α) :
    ∀ n, (ladder R n).dimension = n
  | 0 => rfl
  | Nat.succ n =>
      have ih := ladder_dimension R n
      calc
        (ladder R (Nat.succ n)).dimension
            = ((ladder R n).elevate).dimension := by rfl
        _ = (ladder R n).dimension + 1 := (ladder R n).elevate_dimension
        _ = n + 1 := by rw [ih]
        _ = Nat.succ n := (Nat.succ_eq_add_one n).symm

@[simp] lemma ladder_core (R : Reentry α) :
    ∀ n, (ladder R n).dial.core = R
  | 0 => rfl
  | Nat.succ n =>
      calc
        (ladder R (Nat.succ n)).dial.core
            = ((ladder R n).elevate).dial.core := rfl
        _ = (ladder R n).dial.core := elevate_core _
        _ = R := ladder_core R n

def one (R : Reentry α) : DialParam α := ladder R 1
def two (R : Reentry α) : DialParam α := ladder R 2
def three (R : Reentry α) : DialParam α := ladder R 3

lemma base_le_one (R : Reentry α) :
    (base R).le (one R) := by
  simpa [one, ladder] using (base R).le_elevate

inductive Stage
  | boolean
  | heyting
  | mv
  | effect
  | orthomodular
  | beyond
  deriving DecidableEq, Repr

def Stage.next : Stage → Stage
  | Stage.boolean => Stage.heyting
  | Stage.heyting => Stage.mv
  | Stage.mv => Stage.effect
  | Stage.effect => Stage.orthomodular
  | Stage.orthomodular => Stage.beyond
  | Stage.beyond => Stage.beyond

def stageOfNat : ℕ → Stage
  | 0 => Stage.boolean
  | 1 => Stage.heyting
  | 2 => Stage.mv
  | 3 => Stage.effect
  | 4 => Stage.orthomodular
  | _ => Stage.beyond

lemma stageOfNat_succ (n : ℕ) :
    stageOfNat (Nat.succ n) = (stageOfNat n).next := by
  classical
  cases n with
  | zero =>
      simp [stageOfNat, Stage.next]
  | succ n₁ =>
      cases n₁ with
      | zero =>
          simp [stageOfNat, Stage.next]
      | succ n₂ =>
          cases n₂ with
          | zero =>
              simp [stageOfNat, Stage.next]
          | succ n₃ =>
              cases n₃ with
              | zero =>
                  simp [stageOfNat, Stage.next]
              | succ n₄ =>
                  cases n₄ with
                  | zero =>
                      simp [stageOfNat, Stage.next]
                  | succ n₅ =>
                      cases n₅ with
                      | zero => simp [stageOfNat, Stage.next]
                      | succ _ => simp [stageOfNat, Stage.next]

lemma stageOfNat_add_five (k : ℕ) :
    stageOfNat (5 + k) = Stage.beyond := by
  refine Nat.rec ?base ?step k
  · simp [stageOfNat]
  · intro k ih
    have h := stageOfNat_succ (n := 5 + k)
    have hsucc :
        stageOfNat (Nat.succ (5 + k)) = Stage.beyond :=
      by
        simpa [Nat.succ_eq_add_one, Nat.add_comm, Nat.add_left_comm,
          Nat.add_assoc, Stage.next, ih] using h
    simpa [Nat.succ_eq_add_one, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hsucc

/-- Coarsely classify a dial parameter by its dimension. -/
def stage (P : DialParam α) : Stage :=
  stageOfNat P.dimension

lemma process_pos (P : DialParam α) :
    ⊥ < ((P.dial.core.process : P.dial.core.Omega) : α) :=
  Reentry.process_pos (R := P.dial.core)

lemma counter_pos (P : DialParam α) :
    ⊥ < ((P.dial.core.counterProcess : P.dial.core.Omega) : α) :=
  Reentry.counter_pos (R := P.dial.core)

lemma process_le_of_pos (P : DialParam α) {x : P.dial.core.Omega}
    (hx : ⊥ < (x : α)) :
    P.dial.core.process ≤ x :=
  Reentry.process_le_of_pos (R := P.dial.core) hx

@[simp] lemma euler_boundary_coe (P : DialParam α) :
    ((P.dial.core.eulerBoundary : P.dial.core.Omega) : α)
      = P.dial.core.primordial := by
  simp [Reentry.eulerBoundary_eq_process, Reentry.process_coe]

lemma euler_boundary_process (P : DialParam α) :
    P.dial.core.eulerBoundary = P.dial.core.process :=
  Reentry.eulerBoundary_eq_process (R := P.dial.core)

@[simp] lemma stage_base (R : Reentry α) :
    (base R).stage = Stage.boolean := rfl

@[simp] lemma stage_elevate (P : DialParam α) :
    (P.elevate).stage = (P.stage).next := by
  unfold stage
  have := stageOfNat_succ (n := P.dimension)
  simpa [elevate_dimension, Nat.succ_eq_add_one] using this

def booleanParam (R : Reentry α) : DialParam α :=
  base R

def heytingParam (R : Reentry α) : DialParam α :=
  ladder R 1

def mvParam (R : Reentry α) : DialParam α :=
  ladder R 2

def effectParam (R : Reentry α) : DialParam α :=
  ladder R 3

def orthomodularParam (R : Reentry α) : DialParam α :=
  ladder R 4

@[simp] lemma booleanParam_stage (R : Reentry α) :
    (booleanParam (α := α) R).stage = Stage.boolean := rfl

@[simp] lemma heytingParam_stage (R : Reentry α) :
    (heytingParam (α := α) R).stage = Stage.heyting := by
  unfold stage
  simp [heytingParam, ladder_dimension, stageOfNat]

@[simp] lemma mvParam_stage (R : Reentry α) :
    (mvParam (α := α) R).stage = Stage.mv := by
  unfold stage
  simp [mvParam, ladder_dimension, stageOfNat]

@[simp] lemma effectParam_stage (R : Reentry α) :
    (effectParam (α := α) R).stage = Stage.effect := by
  unfold stage
  simp [effectParam, ladder_dimension, stageOfNat]

@[simp] lemma orthomodularParam_stage (R : Reentry α) :
    (orthomodularParam (α := α) R).stage = Stage.orthomodular := by
  unfold stage
  simp [orthomodularParam, ladder_dimension, stageOfNat]

@[simp] lemma booleanParam_core (R : Reentry α) :
    (booleanParam (α := α) R).dial.core = R := rfl

@[simp] lemma heytingParam_core (R : Reentry α) :
    (heytingParam (α := α) R).dial.core = R := by
  simp [heytingParam, ladder_core]

@[simp] lemma mvParam_core (R : Reentry α) :
    (mvParam (α := α) R).dial.core = R := by
  simp [mvParam, ladder_core]

@[simp] lemma effectParam_core (R : Reentry α) :
    (effectParam (α := α) R).dial.core = R := by
  simp [effectParam, ladder_core]

@[simp] lemma orthomodularParam_core (R : Reentry α) :
    (orthomodularParam (α := α) R).dial.core = R := by
  simp [orthomodularParam, ladder_core]

end DialParam

end Modal
end Logic
end HeytingLean
