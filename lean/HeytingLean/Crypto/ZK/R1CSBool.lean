import HeytingLean.Crypto.Prog
import HeytingLean.Crypto.VM
import HeytingLean.Crypto.Compile
import HeytingLean.Crypto.BoolLens
import HeytingLean.Crypto.ZK.BoolArith
import HeytingLean.Crypto.ZK.R1CS
import HeytingLean.Crypto.ZK.Support

open scoped BigOperators

namespace HeytingLean
namespace Crypto
namespace ZK
namespace R1CSBool

open BoolLens
open Finset

/-- Builder state used while translating a boolean trace to R1CS. -/
structure Builder where
  nextVar : Var := 0
  assign : Var → ℚ := fun _ => 0
  constraints : List Constraint := []

namespace Builder

/-- Allocate a fresh variable with the provided witness value. -/
def fresh (st : Builder) (value : ℚ) : Builder × Var :=
  let idx := st.nextVar
  let assign' : Var → ℚ := fun j => if j = idx then value else st.assign j
  ({ nextVar := idx + 1, assign := assign', constraints := st.constraints }, idx)

/-- Append a constraint to the builder. -/
def addConstraint (st : Builder) (c : Constraint) : Builder :=
  { st with constraints := c :: st.constraints }

end Builder

/-- Booleanity constraint ensuring `v ∈ {0,1}`. -/
def boolConstraint (v : Var) : Constraint :=
  { A := LinComb.single v 1
    B := ⟨-1, [(v, 1)]⟩
    C := LinComb.ofConst 0 }

/-- Support of the booleanity constraint is the singleton `{v}`. -/
@[simp] lemma boolConstraint_support (v : Var) :
    Constraint.support (boolConstraint v) = ({v} : Finset Var) := by
  classical
  simp [Constraint.support, boolConstraint]

/-- Booleanity constraint evaluates to `a v * (a v - 1) = 0`. -/
@[simp] lemma boolConstraint_satisfied (assign : Var → ℚ) (v : Var) :
    Constraint.satisfied assign (boolConstraint v) ↔
      assign v * (assign v - 1) = 0 := by
  classical
  have hB :
      (⟨-1, [(v, 1)]⟩ : LinComb).eval assign = assign v - 1 := by
    simp [LinComb.eval, sub_eq_add_neg, add_comm]
  simp [Constraint.satisfied, boolConstraint, LinComb.eval_single,
    LinComb.eval_ofConst, hB, sub_eq_add_neg]

/-- Constraint enforcing `v = constant`. -/
def eqConstConstraint (v : Var) (value : ℚ) : Constraint :=
  { A := LinComb.single v 1
    B := LinComb.ofConst 1
    C := LinComb.ofConst value }

/-- Support of the equality-to-constant constraint is `{v}`. -/
@[simp] lemma eqConstConstraint_support (v : Var) (value : ℚ) :
    Constraint.support (eqConstConstraint v value) = ({v} : Finset Var) := by
  classical
  simp [Constraint.support, eqConstConstraint]

/-- Satisfying `eqConstConstraint` is definitionally the equality `assign v = value`. -/
@[simp] lemma eqConstConstraint_satisfied (assign : Var → ℚ)
    (v : Var) (value : ℚ) :
    Constraint.satisfied assign (eqConstConstraint v value) ↔
      assign v = value := by
  classical
  simp [Constraint.satisfied, eqConstConstraint]

/-- Constraint enforcing `lhs = rhs` using a single multiplicative slot. -/
def eqConstraint (lhs rhs : LinComb) : Constraint :=
  { A := lhs, B := LinComb.ofConst 1, C := rhs }

/-- Result of compiling a Boolean form to R1CS. -/
structure Compiled where
  system : System
  assignment : Var → ℚ
  output : Var

private def recordBoolean (builder : Builder) (var : Var) : Builder :=
  Builder.addConstraint builder (boolConstraint var)

private def pushConst (builder : Builder) (value : ℚ) :
    Builder × Var := by
  classical
  let (builder', v) := Builder.fresh builder value
  let builder'' := Builder.addConstraint builder' (eqConstConstraint v value)
  exact (recordBoolean builder'' v, v)

/-- Relation connecting a Boolean stack with its assigned R1CS variables. -/
def Matches (builder : Builder) (stack : Stack) (vars : List Var) : Prop :=
  List.Forall₂ (fun b v => boolToRat b = builder.assign v) stack vars

/-- Every variable stored on the stack is strictly below the current `nextVar`. -/
def Bounded (builder : Builder) (vars : List Var) : Prop :=
  ∀ v, v ∈ vars → v < builder.nextVar

def Builder.system (builder : Builder) : System :=
  { constraints := builder.constraints }

def SupportOK (builder : Builder) : Prop :=
  System.support (Builder.system builder) ⊆ Finset.range builder.nextVar

def StrongInvariant (builder : Builder) (stack : Stack) (vars : List Var) : Prop :=
  Matches builder stack vars ∧
    Bounded builder vars ∧
    SupportOK builder ∧
    System.satisfied builder.assign (Builder.system builder)

namespace StrongInvariant

lemma matches_ {builder : Builder} {stack : Stack} {vars : List Var}
    (h : StrongInvariant builder stack vars) : Matches builder stack vars :=
  h.1

lemma bounded_ {builder : Builder} {stack : Stack} {vars : List Var}
    (h : StrongInvariant builder stack vars) : Bounded builder vars :=
  h.2.1

lemma support_ {builder : Builder} {stack : Stack} {vars : List Var}
    (h : StrongInvariant builder stack vars) : SupportOK builder :=
  h.2.2.1

lemma satisfied_ {builder : Builder} {stack : Stack} {vars : List Var}
    (h : StrongInvariant builder stack vars) :
    System.satisfied builder.assign (Builder.system builder) :=
  h.2.2.2

lemma toInvariant {builder : Builder} {stack : Stack} {vars : List Var}
    (h : StrongInvariant builder stack vars) :
    Matches builder stack vars ∧ Bounded builder vars :=
  ⟨matches_ h, bounded_ h⟩

end StrongInvariant

private lemma range_subset_succ (n : ℕ) :
    Finset.range n ⊆ Finset.range (n + 1) := by
  intro v hv
  have hvlt : v < n := Finset.mem_range.mp hv
  exact Finset.mem_range.mpr (Nat.lt_succ_of_lt hvlt)

private lemma singleton_subset_range {n v : ℕ} (hv : v < n) :
    ({v} : Finset Var) ⊆ Finset.range n := by
  intro w hw
  have hw' : w = v := Finset.mem_singleton.mp hw
  subst hw'
  exact Finset.mem_range.mpr hv

namespace Builder

@[simp] lemma system_constraints (st : Builder) :
    (Builder.system st).constraints = st.constraints := rfl

@[simp] lemma system_fresh (st : Builder) (value : ℚ) :
    Builder.system (fresh st value).1 = Builder.system st := rfl

@[simp] lemma system_addConstraint (st : Builder) (c : Constraint) :
    Builder.system (addConstraint st c) =
      { (Builder.system st) with constraints := c :: (Builder.system st).constraints } := rfl

@[simp] lemma system_recordBoolean (st : Builder) (v : Var) :
    Builder.system (recordBoolean st v) =
      { (Builder.system st) with constraints := boolConstraint v :: (Builder.system st).constraints } := rfl

@[simp] lemma addConstraint_assign (st : Builder) (c : Constraint) :
    (addConstraint st c).assign = st.assign := rfl

@[simp] lemma addConstraint_nextVar (st : Builder) (c : Constraint) :
    (addConstraint st c).nextVar = st.nextVar := rfl

@[simp] lemma recordBoolean_assign (st : Builder) (v : Var) :
    (recordBoolean st v).assign = st.assign := by
  unfold recordBoolean
  simp

@[simp] lemma recordBoolean_nextVar (st : Builder) (v : Var) :
    (recordBoolean st v).nextVar = st.nextVar := by
  unfold recordBoolean
  simp

@[simp] lemma fresh_nextVar (st : Builder) (value : ℚ) :
    (fresh st value).1.nextVar = st.nextVar + 1 := by
  simp [fresh]

@[simp] lemma fresh_assign_self (st : Builder) (value : ℚ) :
    (fresh st value).1.assign (fresh st value).2 = value := by
  classical
  dsimp [fresh]
  simp

@[simp] lemma fresh_assign_lt {st : Builder} {value : ℚ} {w : Var}
    (hw : w < st.nextVar) :
    (fresh st value).1.assign w = st.assign w := by
  classical
  dsimp [fresh]
  have : w ≠ st.nextVar := Nat.ne_of_lt hw
  simp [this]

@[simp] lemma fresh_snd (st : Builder) (value : ℚ) :
    (fresh st value).2 = st.nextVar := rfl

lemma fresh_preserve_bounded {st : Builder} {value : ℚ} {vars : List Var}
    (h : Bounded st vars) :
    Bounded (fresh st value).1 vars := by
  classical
  refine fun v hv => ?_
  have hvlt := h v hv
  have : v < st.nextVar + 1 := Nat.lt_succ_of_lt hvlt
  simpa [fresh] using this

lemma fresh_preserve_support {st : Builder} {value : ℚ}
    (h : SupportOK st) :
    SupportOK (fresh st value).1 := by
  intro v hv
  have hvOld : v ∈ System.support (Builder.system st) := by
    simpa using hv
  have hvRange : v ∈ Finset.range st.nextVar := h hvOld
  have hvRangeSucc : v ∈ Finset.range (st.nextVar + 1) :=
    range_subset_succ st.nextVar hvRange
  simpa [fresh] using hvRangeSucc

lemma fresh_agreesOn_range (st : Builder) (value : ℚ) :
    AgreesOn (Finset.range st.nextVar) st.assign (fresh st value).1.assign := by
  intro v hv
  have hvlt : v < st.nextVar := Finset.mem_range.mp hv
  have := fresh_assign_lt (st := st) (value := value) (w := v) (hw := hvlt)
  simpa using this.symm

lemma fresh_preserve_satisfied_mem {st : Builder} {value : ℚ}
    (hSupport : SupportOK st)
    (hSat : System.satisfied st.assign (Builder.system st)) :
    System.satisfied (fresh st value).1.assign
        (Builder.system (fresh st value).1) := by
  classical
  intro c hc
  have hSatOld :
      System.satisfied (fresh st value).1.assign (Builder.system st) :=
    (System.satisfied_ext
        (sys := Builder.system st)
        (a := st.assign)
        (a' := (fresh st value).1.assign)
        (dom := Finset.range st.nextVar)
        (hSupp := hSupport)
        (hAgree := fresh_agreesOn_range st value)).1 hSat
  have hcOld : c ∈ (Builder.system st).constraints := by
    simpa using hc
  have := hSatOld hcOld
  simpa using this

lemma addConstraint_preserve_support {st : Builder} {c : Constraint}
    (hSupport : SupportOK st)
    (hc : Constraint.support c ⊆ Finset.range st.nextVar) :
    SupportOK (addConstraint st c) := by
  classical
  intro v hv
  have hvUnion :
      v ∈ System.support (Builder.system st) ∪ Constraint.support c := by
    simpa [Builder.system_addConstraint] using hv
  have hCases := Finset.mem_union.mp hvUnion
  cases hCases with
  | inl hvOld =>
      have := hSupport hvOld
      simpa [Builder.addConstraint_nextVar] using this
  | inr hvNew =>
      have := hc hvNew
      simpa [Builder.addConstraint_nextVar] using this

lemma addConstraint_preserve_satisfied_mem {st : Builder} {c : Constraint}
    (hSat : System.satisfied st.assign (Builder.system st))
    (hc : Constraint.satisfied st.assign c) :
    System.satisfied (addConstraint st c).assign
        (Builder.system (addConstraint st c)) := by
  intro d hd
  have hdCons : d = c ∨ d ∈ st.constraints := by
    simpa [Builder.system_addConstraint] using hd
  cases hdCons with
  | inl hdc =>
      subst hdc
      simpa [Builder.addConstraint_assign] using hc
  | inr hdOld =>
      have := hSat hdOld
      simpa [Builder.addConstraint_assign] using this

end Builder

lemma addConstraint_preserve_matches {builder : Builder} {stack vars}
    (h : Matches builder stack vars) (c : Constraint) :
    Matches (Builder.addConstraint builder c) stack vars := by
  simpa [Matches] using h

lemma addConstraint_preserve_bounded {builder : Builder} {vars : List Var}
    (h : Bounded builder vars) (c : Constraint) :
    Bounded (Builder.addConstraint builder c) vars := by
  simpa [Bounded] using h

lemma recordBoolean_preserve_matches {builder : Builder} {stack vars} (v : Var)
    (h : Matches builder stack vars) :
    Matches (recordBoolean builder v) stack vars := by
  unfold recordBoolean
  simpa [Matches] using h

lemma recordBoolean_preserve_bounded {builder : Builder} {vars : List Var} (v : Var)
    (h : Bounded builder vars) :
    Bounded (recordBoolean builder v) vars := by
  unfold recordBoolean
  simpa [Bounded] using h

lemma recordBoolean_preserve_support {builder : Builder} {v : Var}
    (hSupport : SupportOK builder) (hv : v < builder.nextVar) :
    SupportOK (recordBoolean builder v) := by
  classical
  have hSubset :
      Constraint.support (boolConstraint v) ⊆ Finset.range builder.nextVar := by
    intro w hw
    have hw' : w = v := by
      simpa [boolConstraint_support] using hw
    subst hw'
    exact Finset.mem_range.mpr hv
  have :=
    Builder.addConstraint_preserve_support
      (st := builder) (c := boolConstraint v) hSupport hSubset
  simpa [recordBoolean, Builder.system_recordBoolean]
    using this

lemma recordBoolean_preserve_satisfied_mem {builder : Builder} {v : Var}
    (hSat : System.satisfied builder.assign (Builder.system builder))
    (hv : Constraint.satisfied builder.assign (boolConstraint v)) :
    System.satisfied (recordBoolean builder v).assign
        (Builder.system (recordBoolean builder v)) := by
  classical
  unfold System.satisfied at hSat ⊢
  intro c hc
  have hc' : c = boolConstraint v ∨ c ∈ builder.constraints := by
    simpa [recordBoolean, Builder.system_recordBoolean] using hc
  -- `recordBoolean` does not change the assignment except for adding the constraint.
  have hAssign :
      (recordBoolean builder v).assign = builder.assign := by
    simp [recordBoolean]
  cases hc' with
  | inl hEq =>
      subst hEq
      -- Constraint satisfied by assumption `hv`.
      simpa [hAssign] using hv
  | inr hMem =>
      have := hSat hMem
      simpa [hAssign] using this

namespace BuilderPreserve

lemma fresh_agreesOn_support {b : Builder} {val : ℚ}
    (hOK : SupportOK b) :
    ∀ v ∈ System.support (Builder.system b),
      b.assign v = (Builder.fresh b val).1.assign v := by
  intro v hv
  have hvRange : v ∈ Finset.range b.nextVar := hOK hv
  have hvlt : v < b.nextVar := Finset.mem_range.mp hvRange
  have hEq := Builder.fresh_assign_lt (st := b) (value := val) (w := v) (hw := hvlt)
  simpa using hEq.symm

lemma fresh_preserve_satisfied {b : Builder} {val : ℚ}
    (hOK : SupportOK b)
    (hSat : System.satisfied b.assign (Builder.system b)) :
    System.satisfied (Builder.fresh b val).1.assign
        (Builder.system (Builder.fresh b val).1) :=
  Builder.fresh_preserve_satisfied_mem (st := b) (value := val) hOK hSat

lemma addConstraint_preserve_satisfied {b : Builder} {c : Constraint}
    (hTail : System.satisfied b.assign (Builder.system b))
    (hHead : Constraint.satisfied b.assign c) :
    System.satisfied (Builder.addConstraint b c).assign
        (Builder.system (Builder.addConstraint b c)) :=
  Builder.addConstraint_preserve_satisfied_mem
    (st := b) (c := c) hTail hHead

lemma recordBoolean_preserve_satisfied {b : Builder} {v : Var}
    (hSat : System.satisfied b.assign (Builder.system b))
    (hv : Constraint.satisfied b.assign (boolConstraint v)) :
    System.satisfied (recordBoolean b v).assign
        (Builder.system (recordBoolean b v)) :=
  _root_.HeytingLean.Crypto.ZK.R1CSBool.recordBoolean_preserve_satisfied_mem
    (builder := b) (v := v) hSat hv

end BuilderPreserve

@[simp] lemma matches_nil (builder : Builder) :
    Matches builder [] [] := List.Forall₂.nil

lemma matches_cons_head {builder : Builder} {b : Bool} {stack : Stack}
    {v : Var} {vars : List Var}
    (h : Matches builder (b :: stack) (v :: vars)) :
    boolToRat b = builder.assign v := by
  cases h with
  | cons hHead _ => simpa using hHead

lemma matches_cons_tail {builder : Builder} {b : Bool} {stack : Stack}
    {v : Var} {vars : List Var}
    (h : Matches builder (b :: stack) (v :: vars)) :
    Matches builder stack vars := by
  cases h with
  | cons _ hTail => simpa using hTail

lemma matches_tail_tail {builder : Builder} {b₁ b₂ : Bool} {stack : Stack}
    {v₁ v₂ : Var} {vars : List Var}
    (h : Matches builder (b₁ :: b₂ :: stack) (v₁ :: v₂ :: vars)) :
    Matches builder stack vars := by
  have hTail₁ :=
    matches_cons_tail (builder := builder) (b := b₁)
      (stack := b₂ :: stack) (v := v₁) (vars := v₂ :: vars) h
  exact matches_cons_tail (builder := builder) (b := b₂)
      (stack := stack) (v := v₂) (vars := vars) hTail₁

lemma matches_length_eq {builder : Builder} {stack : Stack} {vars : List Var}
    (h : Matches builder stack vars) :
    stack.length = vars.length := by
  induction h with
  | nil => simp
  | cons _ _ ih => simp [ih]

lemma matches_fresh_preserve {builder : Builder} {stack : Stack}
    {vars : List Var} {value : ℚ}
    (hM : Matches builder stack vars)
    (hB : Bounded builder vars) :
    Matches (Builder.fresh builder value).1 stack vars := by
  classical
  revert hB
  induction hM with
  | nil => intro; simp [Matches]
  | @cons b v stack vars hHead hTail ih =>
      intro hB
      have hvlt : v < builder.nextVar := hB v (by simp)
      have hBtail : Bounded builder vars := by
        intro w hw; exact hB w (by simp [hw])
      have hTail' := ih hBtail
      have hAssign := Builder.fresh_assign_lt (st := builder)
        (value := value) (w := v) (hw := hvlt)
      refine List.Forall₂.cons ?_ hTail'
      simpa [hAssign] using hHead

lemma Bounded.tail {builder : Builder} {v : Var} {vars : List Var}
    (h : Bounded builder (v :: vars)) :
    Bounded builder vars := by
  intro w hw
  exact h w (by simp [hw])

lemma Bounded.tail_tail {builder : Builder} {v₁ v₂ : Var} {vars : List Var}
    (h : Bounded builder (v₁ :: v₂ :: vars)) :
    Bounded builder vars :=
  Bounded.tail
    (builder := builder)
    (v := v₂) (vars := vars)
    (Bounded.tail (builder := builder)
      (v := v₁) (vars := v₂ :: vars) h)

/-- Convenience invariant bundling matches, bounds, and stack length. -/
def Invariant (builder : Builder) (stack : Stack) (vars : List Var) : Prop :=
  Matches builder stack vars ∧ Bounded builder vars ∧ stack.length = vars.length

namespace Invariant

lemma tail {builder : Builder} {stack : Stack} {vars : List Var}
    {b : Bool} {v : Var}
    (h : Invariant builder (b :: stack) (v :: vars)) :
    Invariant builder stack vars :=
  ⟨matches_cons_tail h.1, Bounded.tail h.2.1,
    by
      have := h.2.2
      simp [List.length_cons] at this
      exact this⟩

lemma tail₂ {builder : Builder} {stack : Stack} {vars : List Var}
    {b₁ b₂ : Bool} {v₁ v₂ : Var}
    (h : Invariant builder (b₁ :: b₂ :: stack) (v₁ :: v₂ :: vars)) :
    Invariant builder stack vars :=
  tail (builder := builder)
    (stack := stack) (vars := vars) (b := b₂) (v := v₂)
    (tail (builder := builder)
      (stack := b₂ :: stack) (vars := v₂ :: vars)
      (b := b₁) (v := v₁) h)

end Invariant

lemma pushConst_invariant {builder : Builder} {stack : Stack}
    {vars : List Var} {value : ℚ} {b : Bool}
    (hInv : Invariant builder stack vars)
    (hvalue : value = boolToRat b) :
    let result := pushConst builder value
    Invariant result.1 (b :: stack) (result.2 :: vars) := by
  classical
  obtain ⟨hMatches, hBounded, hLen⟩ := hInv
  dsimp [pushConst]
  cases hFresh : Builder.fresh builder value with
  | mk builder₁ v =>
      have hv_idx : v = builder.nextVar := by
        simpa [hFresh] using Builder.fresh_snd (st := builder) (value := value)
      have hNext₁ : builder₁.nextVar = builder.nextVar + 1 := by
        simpa [hFresh] using Builder.fresh_nextVar (st := builder) (value := value)
      have hMatches₁ : Matches builder₁ stack vars := by
        have := matches_fresh_preserve (builder := builder) (value := value)
          (stack := stack) (vars := vars) hMatches hBounded
        simpa [hFresh] using this
      have hBounded₁ : Bounded builder₁ vars := by
        have := Builder.fresh_preserve_bounded (st := builder)
          (value := value) (vars := vars) hBounded
        simpa [hFresh] using this
      have hAssign₁ : builder₁.assign v = value := by
        have := Builder.fresh_assign_self (st := builder) (value := value)
        simpa [hFresh] using this
      let builder₂ := Builder.addConstraint builder₁ (eqConstConstraint v value)
      have hMatches₂ : Matches builder₂ stack vars := by
        simpa [builder₂] using addConstraint_preserve_matches hMatches₁ _
      have hBounded₂ : Bounded builder₂ vars := by
        simpa [builder₂] using addConstraint_preserve_bounded hBounded₁ _
      let builder₃ := recordBoolean builder₂ v
      have hMatches₃ : Matches builder₃ stack vars := by
        simpa [builder₃] using
          recordBoolean_preserve_matches (builder := builder₂)
            (stack := stack) (vars := vars) (v := v) hMatches₂
      have hBounded₃ : Bounded builder₃ vars := by
        simpa [builder₃] using
          recordBoolean_preserve_bounded (builder := builder₂)
            (vars := vars) (v := v) hBounded₂
      have hHead : boolToRat b = builder₃.assign v := by
        subst builder₃
        simp [builder₂, recordBoolean, hAssign₁, hvalue]
      have hNext₃ : builder₃.nextVar = builder₁.nextVar := by
        simp [builder₃, builder₂, Builder.recordBoolean_nextVar, Builder.addConstraint_nextVar]
      have hMatches_new : Matches builder₃ (b :: stack) (v :: vars) :=
        List.Forall₂.cons hHead hMatches₃
      have hBounded_new : Bounded builder₃ (v :: vars) := by
        intro w hw
        rcases List.mem_cons.mp hw with hw | hw
        · subst hw
          have : builder.nextVar < builder.nextVar + 1 := Nat.lt_succ_self _
          simpa [hv_idx, hNext₁, hNext₃] using this
        · exact hBounded₃ w hw
      have hMatches_new' : Matches builder₃ (b :: stack) (builder.nextVar :: vars) :=
        by simpa [hv_idx] using hMatches_new
      have hBounded_new' : Bounded builder₃ (builder.nextVar :: vars) :=
        by simpa [hv_idx] using hBounded_new
      have hLen_new : (b :: stack).length = (builder.nextVar :: vars).length := by
        simpa [hv_idx, hLen]
      have hGoal : Invariant builder₃ (b :: stack) (builder.nextVar :: vars) :=
        And.intro hMatches_new' (And.intro hBounded_new' hLen_new)
      have hResult : pushConst builder value = (builder₃, v) := by
        simp [pushConst, hFresh, builder₂, builder₃]
      simpa [Invariant, hResult] using hGoal

lemma pushConst_strong {builder : Builder} {stack : Stack}
    {vars : List Var} {value : ℚ} {b : Bool}
    (hStrong : StrongInvariant builder stack vars)
    (hvalue : value = boolToRat b) :
    let result := pushConst builder value
    StrongInvariant result.1 (b :: stack) (result.2 :: vars) := by
  classical
  obtain ⟨hMatches, hBounded, hSupport, hSat⟩ := hStrong
  dsimp [pushConst]
  cases hFresh : Builder.fresh builder value with
  | mk builder₁ v =>
      have hv_idx : v = builder.nextVar := by
        simpa [hFresh] using
          Builder.fresh_snd (st := builder) (value := value)
      have hNext₁ : builder₁.nextVar = builder.nextVar + 1 := by
        simpa [hFresh] using
          Builder.fresh_nextVar (st := builder) (value := value)
      have hv_lt_next : v < builder₁.nextVar := by
        have : builder.nextVar < builder.nextVar + 1 := Nat.lt_succ_self _
        simpa [hv_idx, hNext₁] using this
      have hMatches₁ : Matches builder₁ stack vars := by
        have := matches_fresh_preserve
          (builder := builder) (value := value)
          (stack := stack) (vars := vars) hMatches hBounded
        simpa [hFresh] using this
      have hBounded₁ : Bounded builder₁ vars := by
        have := Builder.fresh_preserve_bounded
          (st := builder) (value := value) (vars := vars) hBounded
        simpa [hFresh] using this
      have hSupport₁ :
          SupportOK builder₁ := by
        have := Builder.fresh_preserve_support
          (st := builder) (value := value) hSupport
        simpa [hFresh] using this
      have hSat₁ :
          System.satisfied builder₁.assign (Builder.system builder₁) := by
        intro c hc
        have hc' :
            c ∈ (Builder.system (Builder.fresh builder value).1).constraints := by
          simpa [Builder.system, hFresh] using hc
        have hSatFresh :
            System.satisfied (Builder.fresh builder value).1.assign
              (Builder.system (Builder.fresh builder value).1) :=
          Builder.fresh_preserve_satisfied_mem
            (st := builder) (value := value) hSupport hSat
        have := hSatFresh (c := c) hc'
        simpa [Builder.system, hFresh] using this
      have hAssign₁ : builder₁.assign v = value := by
        have := Builder.fresh_assign_self
          (st := builder) (value := value)
        simpa [hFresh] using this
      let builder₂ := Builder.addConstraint builder₁ (eqConstConstraint v value)
      have hMatches₂ : Matches builder₂ stack vars := by
        simpa [builder₂] using
          addConstraint_preserve_matches hMatches₁ _
      have hBounded₂ : Bounded builder₂ vars := by
        simpa [builder₂] using
          addConstraint_preserve_bounded hBounded₁ _
      have hSupport₂ :
          SupportOK builder₂ := by
        have hSubset :
            Constraint.support (eqConstConstraint v value) ⊆
              Finset.range builder₁.nextVar := by
          simpa [eqConstConstraint_support] using
            (singleton_subset_range
              (n := builder₁.nextVar) (v := v) hv_lt_next)
        have := Builder.addConstraint_preserve_support
          (st := builder₁)
          (c := eqConstConstraint v value)
          hSupport₁ hSubset
        simpa [builder₂] using this
      have hSat₂ :
          System.satisfied builder₂.assign (Builder.system builder₂) := by
        have hEqConstraint :
            Constraint.satisfied builder₁.assign (eqConstConstraint v value) :=
          (eqConstConstraint_satisfied
              (assign := builder₁.assign) (v := v) (value := value)).2
            hAssign₁
        intro c hc
        have hc' :
            c ∈ (Builder.system
              (Builder.addConstraint builder₁ (eqConstConstraint v value))).constraints := by
          simpa [Builder.system, builder₂] using hc
        have hSatAdd :
            System.satisfied
              (Builder.addConstraint builder₁ (eqConstConstraint v value)).assign
              (Builder.system (Builder.addConstraint builder₁ (eqConstConstraint v value))) :=
          Builder.addConstraint_preserve_satisfied_mem
            (st := builder₁)
            (c := eqConstConstraint v value)
            hSat₁ hEqConstraint
        have := hSatAdd (c := c) hc'
        simpa [Builder.system, builder₂] using this
      let builder₃ := recordBoolean builder₂ v
      have hMatches₃ : Matches builder₃ stack vars := by
        simpa [builder₃] using
          recordBoolean_preserve_matches
            (builder := builder₂)
            (stack := stack) (vars := vars) (v := v) hMatches₂
      have hBounded₃ : Bounded builder₃ vars := by
        simpa [builder₃] using
          recordBoolean_preserve_bounded
            (builder := builder₂) (v := v) hBounded₂
      have hv_lt_next₂ : v < builder₂.nextVar := by
        simpa [builder₂, Builder.addConstraint_nextVar] using hv_lt_next
      have hSupport₃ :
          SupportOK builder₃ :=
        recordBoolean_preserve_support
          (builder := builder₂) (v := v) hSupport₂ hv_lt_next₂
      have hAssign₂ : builder₂.assign v = value := by
        simpa [builder₂, Builder.addConstraint_assign] using hAssign₁
      have hBoolEq :
          builder₂.assign v * (builder₂.assign v - 1) = 0 := by
        simpa [hAssign₂, hvalue] using ZK.boolToRat_sq_sub b
      have hBoolConstraint :
          Constraint.satisfied builder₂.assign (boolConstraint v) :=
        (boolConstraint_satisfied (assign := builder₂.assign) (v := v)).2
          hBoolEq
      have hSat₃ :
          System.satisfied builder₃.assign (Builder.system builder₃) :=
        recordBoolean_preserve_satisfied_mem
          (builder := builder₂) (v := v) hSat₂ hBoolConstraint
      have hAssign₂_bool : builder₂.assign v = boolToRat b := by
        simpa [hvalue] using hAssign₂
      have hAssign₃ : builder₃.assign v = boolToRat b := by
        have : builder₃.assign v = builder₂.assign v := by
          simp [builder₃, builder₂, recordBoolean]
        exact this.trans hAssign₂_bool
      have hNext₂ : builder₂.nextVar = builder₁.nextVar := by
        simp [builder₂]
      have hNext₃ : builder₃.nextVar = builder₁.nextVar := by
        simp [builder₃, builder₂, Builder.recordBoolean_nextVar,
          Builder.addConstraint_nextVar]
      have hHead :
          boolToRat b = builder₃.assign builder.nextVar := by
        simpa [hv_idx] using hAssign₃.symm
      have hMatches_final :
          Matches builder₃ (b :: stack) (builder.nextVar :: vars) :=
        List.Forall₂.cons hHead hMatches₃
      have hBounded_final :
          Bounded builder₃ (builder.nextVar :: vars) := by
        intro w hw
        rcases List.mem_cons.mp hw with hw | hw
        · subst hw
          have hv_lt_builder₃ : v < builder₃.nextVar := by
            simpa [hNext₃] using hv_lt_next
          simpa [hv_idx] using hv_lt_builder₃
        · exact hBounded₃ w hw
      have hSupport_new :
          SupportOK builder₃ := hSupport₃
      have hSat_new :
          System.satisfied builder₃.assign (Builder.system builder₃) :=
        hSat₃
      have hResult :
          pushConst builder value = (builder₃, v) := by
        simp [pushConst, hFresh, builder₂, builder₃]
      have hStrong_new :
          StrongInvariant builder₃ (b :: stack) (builder.nextVar :: vars) :=
        ⟨hMatches_final,
          ⟨hBounded_final, ⟨hSupport_new, hSat_new⟩⟩⟩
      have :
          StrongInvariant
            (recordBoolean
              (Builder.addConstraint builder₁ (eqConstConstraint builder.nextVar value))
              builder.nextVar)
            (b :: stack) (builder.nextVar :: vars) := by
        simpa [builder₃, builder₂, hv_idx] using hStrong_new
      exact this

lemma applyAnd_invariant {builder : Builder} {x y : Bool}
    {before : Stack} {vx vy : Var} {vars : List Var}
    (hInv : Invariant builder (x :: y :: before) (vx :: vy :: vars)) :
    Invariant
      (recordBoolean
        (Builder.addConstraint (Builder.fresh builder (boolToRat (y && x))).1
          { A := LinComb.single vx 1
            B := LinComb.single vy 1
            C := LinComb.single (Builder.fresh builder (boolToRat (y && x))).2 1 })
        (Builder.fresh builder (boolToRat (y && x))).2)
      ((y && x) :: before)
      ((Builder.fresh builder (boolToRat (y && x))).2 :: vars) := by
  classical
  let z : Bool := y && x
  let fres := Builder.fresh builder (boolToRat z)
  let builder1 := fres.1
  let vz := fres.2
  let builder2 := Builder.addConstraint builder1
    { A := LinComb.single vx 1
      B := LinComb.single vy 1
      C := LinComb.single vz 1 }
  let builder3 := recordBoolean builder2 vz
  change Invariant builder3 (z :: before) (vz :: vars)
  obtain ⟨hMatches, hBounded, hLen⟩ := hInv
  have hMatchesTail := matches_cons_tail hMatches
  have hMatchesRest : Matches builder before vars :=
    matches_cons_tail hMatchesTail
  have hBoundedTail : Bounded builder (vy :: vars) :=
    Bounded.tail hBounded
  have hBoundedRest : Bounded builder vars :=
    Bounded.tail hBoundedTail
  have hLenRest : before.length = vars.length :=
    matches_length_eq hMatchesRest
  have hv_idx : vz = builder.nextVar := by
    simp [vz, fres]
  have hNext₁ : builder1.nextVar = builder.nextVar + 1 := by
    simp [builder1, fres]
  have hMatches1 : Matches builder1 (x :: y :: before) (vx :: vy :: vars) := by
    have := matches_fresh_preserve (builder := builder) (value := boolToRat z)
      (stack := x :: y :: before) (vars := vx :: vy :: vars) hMatches hBounded
    simpa [builder1, fres] using this
  have hBounded1 : Bounded builder1 (vx :: vy :: vars) := by
    have := Builder.fresh_preserve_bounded (st := builder)
      (value := boolToRat z) (vars := vx :: vy :: vars) hBounded
    simpa [builder1, fres] using this
  have hAssign1 : builder1.assign vz = boolToRat z := by
    have := Builder.fresh_assign_self (st := builder) (value := boolToRat z)
    simpa [builder1, vz, fres] using this
  set builder2 := Builder.addConstraint builder1
    { A := LinComb.single vx 1
      B := LinComb.single vy 1
      C := LinComb.single vz 1 }
  have hMatches2 : Matches builder2 (x :: y :: before) (vx :: vy :: vars) := by
    simpa [builder2] using addConstraint_preserve_matches hMatches1 _
  have hBounded2 : Bounded builder2 (vx :: vy :: vars) := by
    simpa [builder2] using addConstraint_preserve_bounded hBounded1 _
  set builder3 := recordBoolean builder2 vz
  have hMatches3 : Matches builder3 (x :: y :: before) (vx :: vy :: vars) := by
    simpa [builder3] using
      recordBoolean_preserve_matches (builder := builder2)
        (stack := x :: y :: before) (vars := vx :: vy :: vars) (v := vz) hMatches2
  have hBounded3 : Bounded builder3 (vx :: vy :: vars) := by
    simpa [builder3] using
      recordBoolean_preserve_bounded (builder := builder2)
        (vars := vx :: vy :: vars) (v := vz) hBounded2
  have hMatchesRest3 : Matches builder3 before vars :=
    matches_tail_tail hMatches3
  have hBoundedRest3 : Bounded builder3 vars :=
    Bounded.tail_tail hBounded3
  have hAssign3 : builder3.assign vz = boolToRat z := by
    subst builder3
    simp [builder2, recordBoolean, hAssign1]
  have hHead : boolToRat z = builder3.assign vz := by
    simpa using hAssign3.symm
  have hMatches_new : Matches builder3 (z :: before) (vz :: vars) :=
    List.Forall₂.cons hHead hMatchesRest3
  have hNext₂ : builder2.nextVar = builder1.nextVar := by
    simp [builder2]
  have hNext₃ : builder3.nextVar = builder1.nextVar := by
    simp [builder3, builder2, Builder.recordBoolean_nextVar, Builder.addConstraint_nextVar]
  have hBounded_new : Bounded builder3 (vz :: vars) := by
    intro w hw
    rcases List.mem_cons.mp hw with hw | hw
    · subst hw
      have : builder.nextVar < builder.nextVar + 1 := Nat.lt_succ_self _
      simpa [hv_idx, hNext₁, hNext₃] using this
    · exact hBoundedRest3 w hw
  have hLen_new : (z :: before).length = (vz :: vars).length := by
    simp [List.length_cons, hLenRest]
  exact And.intro hMatches_new (And.intro hBounded_new hLen_new)

private def compileStep {n : ℕ} (ρ : Env n)
    (instr : Instr n) (before after : Stack)
    (stackVars : List Var) (builder : Builder) :
    Builder × List Var := by
  classical
  cases instr with
  | pushTop =>
      let (builder', v) := pushConst builder 1
      exact (builder', v :: stackVars)
  | pushBot =>
      let (builder', v) := pushConst builder 0
      exact (builder', v :: stackVars)
  | pushVar idx =>
      let val := boolToRat (ρ idx)
      let (builder', v) := pushConst builder val
      exact (builder', v :: stackVars)
  | applyAnd =>
      match before, after, stackVars with
      | _x :: _y :: _, z :: _, vx :: vy :: rest =>
          let (builder1, vz) := Builder.fresh builder (boolToRat z)
          let builder2 :=
            Builder.addConstraint builder1
              { A := LinComb.single vx 1
                B := LinComb.single vy 1
                C := LinComb.single vz 1 }
          let builder3 := recordBoolean builder2 vz
          exact (builder3, vz :: rest)
      | _, _, _ =>
          exact (builder, stackVars)
  | applyOr =>
      match before, after, stackVars with
      | x :: y :: _, z :: _, vx :: vy :: rest =>
          let mulVal := boolToRat (y && x)
          let (builder1, vmul) := Builder.fresh builder mulVal
          let builder2 :=
            Builder.addConstraint builder1
              { A := LinComb.single vy 1
                B := LinComb.single vx 1
                C := LinComb.single vmul 1 }
          let builder3 := recordBoolean builder2 vmul
          let (builder4, vz) := Builder.fresh builder3 (boolToRat z)
          let linear : LinComb := { const := 0, terms := [(vx, 1), (vy, 1), (vmul, -1)] }
          let builder5 :=
            Builder.addConstraint builder4
              (eqConstraint (LinComb.single vz 1) linear)
          let builder6 := recordBoolean builder5 vz
          exact (builder6, vz :: rest)
      | _, _, _ =>
          exact (builder, stackVars)
  | applyImp =>
      match before, after, stackVars with
      | x :: y :: _, z :: _, vx :: vy :: rest =>
          let mulVal := boolToRat (y && x)
          let (builder1, vmul) := Builder.fresh builder mulVal
          let builder2 :=
            Builder.addConstraint builder1
              { A := LinComb.single vy 1
                B := LinComb.single vx 1
                C := LinComb.single vmul 1 }
          let builder3 := recordBoolean builder2 vmul
          let (builder4, vz) := Builder.fresh builder3 (boolToRat z)
          let linear : LinComb := { const := 1, terms := [(vy, -1), (vmul, 1)] }
          let builder5 :=
            Builder.addConstraint builder4
              (eqConstraint (LinComb.single vz 1) linear)
          let builder6 := recordBoolean builder5 vz
          exact (builder6, vz :: rest)
      | _, _, _ =>
          exact (builder, stackVars)

private def compileSteps {n : ℕ} (ρ : Env n)
    (prog : Program n) (trace : List Stack)
    (stackVars : List Var) (builder : Builder) :
    Builder × List Var :=
  match prog, trace with
  | [], _ => (builder, stackVars)
  | _, [] => (builder, stackVars)
  | instr :: prog', before :: trace' =>
      match trace' with
      | [] => (builder, stackVars)
      | after :: traceTail =>
          let (builder', stackVars') :=
            compileStep ρ instr before after stackVars builder
          compileSteps ρ prog' (after :: traceTail) stackVars' builder'

/-- Compile a Boolean form/environment pair into R1CS constraints and witness. -/
def compile {n : ℕ} (φ : Form n) (ρ : Env n) : Compiled := by
  classical
  let prog := Form.compile φ
  let trace := BoolLens.traceFrom ρ prog []
  let (builder, stackVars) :=
    compileSteps (ρ := ρ) prog trace [] {}
  let outputVar := stackVars.headD 0
  exact
    { system := { constraints := builder.constraints.reverse }
      assignment := builder.assign
      output := outputVar }

end R1CSBool
end ZK
end Crypto
end HeytingLean
