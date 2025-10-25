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

structure BulletSys where
  system : Bullet.System
  output : Var

def BulletAssign := Var → ℚ

def bulletCompile {n : ℕ} (ρ : Env n) (prog : HeytingLean.Crypto.Prog.Program n) : BulletSys :=
  let trace := BoolLens.traceFrom ρ prog []
  let result := R1CSBool.compileSteps (ρ := ρ) prog trace [] {}
  let builder := result.1
  let stackVars := result.2
  -- create labels for each variable index as a placeholder commitment list
  let labels := (List.range builder.nextVar).map (fun i => Bullet.Commitment.mk s!"v{i}")
  let sys : Bullet.System := { commitments := labels, r1cs := { constraints := builder.constraints.reverse } }
  let out : Var := stackVars.headD 0
  { system := sys, output := out }

def bulletSatisfies (s : BulletSys) (a : BulletAssign) : Prop :=
  R1CS.System.satisfied a s.system.r1cs

def bulletPublic (s : BulletSys) (a : BulletAssign) : List ℚ := [a s.output]

def BulletBackend : ZK.IR.Backend ℚ :=
  { Sys := BulletSys
  , Assign := BulletAssign
  , compile := fun ρ prog => bulletCompile ρ prog
  , satisfies := fun s a => bulletSatisfies s a
  , public := fun s a => bulletPublic s a }

theorem bullet_sound {n : ℕ} (φ : HeytingLean.Crypto.Form n) (ρ : Env n) :
  let p := HeytingLean.Crypto.Form.compile φ
  let s := bulletCompile ρ p
  let res := R1CSBool.compileSteps (ρ := ρ) p (BoolLens.traceFrom ρ p []) [] {}
  let builder := res.1
  let assign : BulletAssign := builder.assign
  bulletSatisfies s assign ∧ bulletPublic s assign = [if BoolLens.eval φ ρ then 1 else 0] := by
  classical
  intro p s res builder assign
  have hSatOrig : R1CS.System.satisfied builder.assign { constraints := builder.constraints } := by
    simpa using (R1CSBool.compileTraceToR1CSFromEmpty_satisfied (ρ := ρ) (prog := p))
  have hSat : R1CS.System.satisfied builder.assign { constraints := builder.constraints.reverse } := by
    exact (ZK.System.satisfied_reverse (assign := builder.assign) (cs := builder.constraints)).2 hSatOrig
  have hOutEval := R1CSSoundness.compile_output_eval (φ := φ) (ρ := ρ)
  refine And.intro ?hsat ?hout
  · simpa [bulletSatisfies] using hSat
  · have : builder.assign s.output = (if BoolLens.eval φ ρ then 1 else 0) := by
      simpa using hOutEval
    simpa [bulletPublic, this]

theorem bullet_complete {n : ℕ} (φ : HeytingLean.Crypto.Form n) (ρ : Env n) :
  let p := HeytingLean.Crypto.Form.compile φ
  let s := bulletCompile ρ p
  ∃ as, bulletSatisfies s as ∧ bulletPublic s as = [if BoolLens.eval φ ρ then 1 else 0] := by
  classical
  intro p s
  let res := R1CSBool.compileSteps (ρ := ρ) p (BoolLens.traceFrom ρ p []) [] {}
  let builder := res.1
  refine ⟨builder.assign, ?_⟩
  simpa [bulletCompile, bulletSatisfies, bulletPublic] using bullet_sound (φ := φ) (ρ := ρ)

end Backends
end ZK
end Crypto
end HeytingLean
import HeytingLean.Crypto.ZK.BulletIR
