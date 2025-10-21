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
def fresh (st : Builder) (value : ℚ) : Builder × Var := by
  classical
  let idx := st.nextVar
  let assign' : Var → ℚ :=
    fun j => if j = idx then value else st.assign j
  let st' : Builder :=
    { nextVar := idx + 1, assign := assign', constraints := st.constraints }
  exact (st', idx)

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
