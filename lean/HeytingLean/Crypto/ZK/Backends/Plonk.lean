import HeytingLean.Crypto.Prog
import HeytingLean.Crypto.BoolLens
import HeytingLean.Crypto.ZK.IR
import HeytingLean.Crypto.ZK.R1CS
import HeytingLean.Crypto.ZK.R1CSBool
import HeytingLean.Crypto.ZK.R1CSSoundness
import HeytingLean.Crypto.ZK.Support
import HeytingLean.Crypto.ZK.PlonkIR

namespace HeytingLean
namespace Crypto
namespace ZK
namespace Backends

open BoolLens
open R1CSBool

/-- System used by the PLONK-style backend wrapper: reuse the R1CS system and
    carry the output wire id. -/
structure PlonkSys where
  system : Plonk.System
  output : Var

def PlonkAssign := Var → ℚ

/-- Compile a BoolLens program into the wrapped system by delegating to the
    existing R1CS builder pipeline. -/
def plonkCompile {n : ℕ} (ρ : Env n) (prog : HeytingLean.Crypto.Prog.Program n) : PlonkSys :=
  let trace := BoolLens.traceFrom ρ prog []
  let result := R1CSBool.compileSteps (ρ := ρ) prog trace [] {}
  let builder := result.1
  let stackVars := result.2
  let gates : List Plonk.Gate :=
    (builder.constraints.reverse).map (fun c => { Plonk.Gate . A := c.A, B := c.B, C := c.C })
  -- identity permutation over 0..maxVar
  let maxVar := (builder.nextVar)
  let copyPerm : List Nat := List.range maxVar
  let sys : Plonk.System := { gates := gates, copyPermutation := copyPerm }
  let out : Var := stackVars.headD 0
  { system := sys, output := out }

/-- Satisfaction relation delegates to the underlying R1CS system. -/
def plonkSatisfies (s : PlonkSys) (a : PlonkAssign) : Prop :=
  let r1 := Plonk.System.toR1CS s.system
  R1CS.System.satisfied a r1

/-- Public output: decode the Boolean result from the assignment and output var. -/
def plonkPublic (s : PlonkSys) (a : PlonkAssign) : List ℚ :=
  [a s.output]

/-- Backend instance specialising the generic interface. -/
def PlonkBackend : ZK.IR.Backend ℚ :=
  { Sys := PlonkSys
  , Assign := PlonkAssign
  , compile := fun ρ prog => plonkCompile ρ prog
  , satisfies := fun s a => plonkSatisfies s a
  , public := fun s a => plonkPublic s a }

/-- Soundness: the canonical assignment satisfies the compiled system and
    matches the BoolLens evaluation on the output. -/
theorem plonk_sound {n : ℕ} (φ : HeytingLean.Crypto.Form n) (ρ : Env n) :
  let p := HeytingLean.Crypto.Form.compile φ
  let s := plonkCompile ρ p
  let res := R1CSBool.compileSteps (ρ := ρ) p (BoolLens.traceFrom ρ p []) [] {}
  let builder := res.1
  let assign : PlonkAssign := builder.assign
  plonkSatisfies s assign ∧ plonkPublic s assign = [if BoolLens.eval φ ρ then 1 else 0] := by
  classical
  intro p s res builder assign
  -- Satisfaction on original order
  have hSatOrig : R1CS.System.satisfied builder.assign { constraints := builder.constraints } := by
    -- from the canonical helper on empty start
    
    have := (R1CSBool.compileTraceToR1CSFromEmpty_satisfied (ρ := ρ) (prog := p))
    -- simplify goal shape
    simpa using this
  -- Flip to reversed order
  have hSat : R1CS.System.satisfied builder.assign { constraints := builder.constraints.reverse } := by
    have hIff := ZK.System.satisfied_reverse (assign := builder.assign) (cs := builder.constraints)
    exact hIff.2 hSatOrig
  -- Output alignment by `compile_output_eval`
  have hOutEval := R1CSSoundness.compile_output_eval (φ := φ) (ρ := ρ)
  have hSysEq : s.system = { constraints := builder.constraints.reverse } := rfl
  have hOutIdx : s.output = (R1CSBool.compile ρ φ).output := rfl
  refine And.intro ?hsat ?hout
  · simpa [plonkSatisfies, hSysEq] using hSat
  · -- `compile_output_eval` states equality of boolToRat(eval) and assignment at output
    -- adapt it to a list with a single element
    have : plonkPublic s builder.assign = [builder.assign s.output] := by
      simp [plonkPublic]
    have hOutVal : builder.assign s.output = (if BoolLens.eval φ ρ then 1 else 0) := by
      -- from compile_output_eval after rewriting the output id
      simpa [hOutIdx, R1CSBool.compile] using hOutEval
    simpa [this, hOutVal]

/-- Completeness: existence of a satisfying assignment that matches the BoolLens evaluation. -/
theorem plonk_complete {n : ℕ} (φ : HeytingLean.Crypto.Form n) (ρ : Env n) :
  let p := HeytingLean.Crypto.Form.compile φ
  let s := plonkCompile ρ p
  ∃ as, plonkSatisfies s as ∧ plonkPublic s as = [if BoolLens.eval φ ρ then 1 else 0] := by
  classical
  intro p s
  -- use the canonical builder assignment as the witness
  let res := R1CSBool.compileSteps (ρ := ρ) p (BoolLens.traceFrom ρ p []) [] {}
  let builder := res.1
  refine ⟨builder.assign, ?_⟩
  have := plonk_sound (φ := φ) (ρ := ρ)
  -- unpack the result and finish
  simpa [plonkCompile, plonkSatisfies, plonkPublic] using this

end Backends
end ZK
end Crypto
end HeytingLean
