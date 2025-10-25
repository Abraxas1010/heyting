import HeytingLean.Crypto.Prog
import HeytingLean.Crypto.BoolLens
import HeytingLean.Crypto.ZK.IR
import HeytingLean.Crypto.ZK.R1CS
import HeytingLean.Crypto.ZK.R1CSBool
import HeytingLean.Crypto.ZK.R1CSSoundness
import HeytingLean.Crypto.ZK.Support

namespace HeytingLean
namespace Crypto
namespace ZK
namespace Backends

open BoolLens
open R1CSBool

structure AirSys where
  system : AIR.System
  output : Var

def AirAssign := Var → ℚ

def airCompile {n : ℕ} (ρ : Env n) (prog : HeytingLean.Crypto.Prog.Program n) : AirSys :=
  let trace := BoolLens.traceFrom ρ prog []
  let result := R1CSBool.compileSteps (ρ := ρ) prog trace [] {}
  let builder := result.1
  let stackVars := result.2
  let sys : AIR.System :=
    { trace := { width := 3, length := trace.length }
    , r1cs  := { constraints := builder.constraints.reverse } }
  let out : Var := stackVars.headD 0
  { system := sys, output := out }

def airSatisfies (s : AirSys) (a : AirAssign) : Prop :=
  R1CS.System.satisfied a s.system.r1cs

def airPublic (s : AirSys) (a : AirAssign) : List ℚ := [a s.output]

def AirBackend : ZK.IR.Backend ℚ :=
  { Sys := AirSys
  , Assign := AirAssign
  , compile := fun ρ prog => airCompile ρ prog
  , satisfies := fun s a => airSatisfies s a
  , public := fun s a => airPublic s a }

theorem air_sound {n : ℕ} (φ : HeytingLean.Crypto.Form n) (ρ : Env n) :
  let p := HeytingLean.Crypto.Form.compile φ
  let s := airCompile ρ p
  let res := R1CSBool.compileSteps (ρ := ρ) p (BoolLens.traceFrom ρ p []) [] {}
  let builder := res.1
  let assign : AirAssign := builder.assign
  airSatisfies s assign ∧ airPublic s assign = [if BoolLens.eval φ ρ then 1 else 0] := by
  classical
  intro p s res builder assign
  have hSatOrig : R1CS.System.satisfied builder.assign { constraints := builder.constraints } := by
    simpa using (R1CSBool.compileTraceToR1CSFromEmpty_satisfied (ρ := ρ) (prog := p))
  have hSat : R1CS.System.satisfied builder.assign { constraints := builder.constraints.reverse } := by
    exact (ZK.System.satisfied_reverse (assign := builder.assign) (cs := builder.constraints)).2 hSatOrig
  have hOutEval := R1CSSoundness.compile_output_eval (φ := φ) (ρ := ρ)
  refine And.intro ?hsat ?hout
  · simpa [airSatisfies] using hSat
  · have : builder.assign s.output = (if BoolLens.eval φ ρ then 1 else 0) := by
      -- same rationale as plonk path; align outputs by definitional equality
      simpa using hOutEval
    simpa [airPublic, this]

theorem air_complete {n : ℕ} (φ : HeytingLean.Crypto.Form n) (ρ : Env n) :
  let p := HeytingLean.Crypto.Form.compile φ
  let s := airCompile ρ p
  ∃ as, airSatisfies s as ∧ airPublic s as = [if BoolLens.eval φ ρ then 1 else 0] := by
  classical
  intro p s
  let res := R1CSBool.compileSteps (ρ := ρ) p (BoolLens.traceFrom ρ p []) [] {}
  let builder := res.1
  refine ⟨builder.assign, ?_⟩
  simpa [airCompile, airSatisfies, airPublic] using air_sound (φ := φ) (ρ := ρ)

end Backends
end ZK
end Crypto
end HeytingLean
import HeytingLean.Crypto.ZK.AirIR
