import HeytingLean.Crypto.BoolLens
import HeytingLean.Crypto.ZK.BoolArith
import HeytingLean.Crypto.ZK.R1CS

namespace HeytingLean
namespace Crypto
namespace ZK

open BoolLens

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

/-- Constraint enforcing `v = constant`. -/
def eqConstConstraint (v : Var) (value : ℚ) : Constraint :=
  { A := LinComb.single v 1
    B := LinComb.ofConst 1
    C := LinComb.ofConst value }

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

namespace Builder

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

lemma fresh_preserve_bounded {st : Builder} {value : ℚ} {vars : List Var}
    (h : Bounded st vars) :
    Bounded (fresh st value).1 vars := by
  classical
  refine fun v hv => ?_
  have hvlt := h v hv
  have : v < st.nextVar + 1 := Nat.lt_succ_of_lt hvlt
  simpa [fresh] using this

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

end ZK
end Crypto
end HeytingLean
