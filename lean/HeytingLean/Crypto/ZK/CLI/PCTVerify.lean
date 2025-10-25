import Lean
import Lean.Data.Json
import HeytingLean.Crypto.Form
import HeytingLean.Crypto.ZK.R1CS
import HeytingLean.Crypto.ZK.Support
import HeytingLean.Crypto.ZK.Export
import HeytingLean.Crypto.ZK.R1CSBool
import HeytingLean.Crypto.BoolLens
import HeytingLean.Crypto.ZK.CLI.PCTR1CS

namespace HeytingLean
namespace Crypto
namespace ZK
namespace CLI
namespace PCTVerify

open IO
open Lean
open BoolLens
open R1CSBool

def main (args : List String) : IO UInt32 := do
  match args with
  | [formPath, envPath, r1csPath, witnessPath] =>
      let formRaw ← FS.readFile formPath
      let envRaw  ← FS.readFile envPath
      let r1csRaw ← FS.readFile r1csPath
      let witRaw  ← FS.readFile witnessPath
      let formJson ← match Json.parse formRaw with | .ok j => pure j | .error err => eprintln err; return 1
      let envJson  ← match Json.parse envRaw  with | .ok j => pure j | .error err => eprintln err; return 1
      let sysJson  ← match Json.parse r1csRaw with | .ok j => pure j | .error err => eprintln err; return 1
      let asJson   ← match Json.parse witRaw  with | .ok j => pure j | .error err => eprintln err; return 1
      let n := match envJson.getArr? with | .ok a => a.size | .error _ => 0
      let formJ ← match CLI.PCTR1CS.FormJ.fromJsonE formJson with | .ok f => pure f | .error err => eprintln err; return 1
      let some φ := CLI.PCTR1CS.FormJ.toForm? n formJ | do eprintln s!"Form contains var ≥ n={n}"; return 1
      let ρ ← match CLI.PCTR1CS.parseEnvE n envJson with | .ok r => pure r | .error err => eprintln err; return 1
      let some sys := Export.jsonToSystem sysJson | do eprintln "Bad system"; return 1
      let some arr := Export.jsonToAssignment asJson | do eprintln "Bad assignment"; return 1
      -- domain check: ensure assignment covers all vars in system support
      let sup := ZK.System.support sys
      let maxVar := (sup.sup id).getD 0
      if arr.size ≤ maxVar then
        eprintln s!"Witness too small, needs ≥ {maxVar+1}, got {arr.size}"
        return 1
      let assign := Export.assignmentOfArray arr
      -- Check satisfaction
      let okSat : Bool :=
        Id.run do
          let mut ok := true
          for c in sys.constraints do
            if ¬ ZK.Constraint.satisfied assign c then
              ok := false
          return ok
      if ¬ okSat then
        eprintln "R1CS not satisfied by witness"
        return 1
      -- Check output alignment against recomputed compiled output
      let compiled := R1CSBool.compile φ ρ
      let expected := compiled.output
      let actual := assign expected
      let want := if BoolLens.eval φ ρ then (1 : ℚ) else 0
      if actual ≠ want then
        eprintln s!"Output mismatch: witness[{expected}]={actual}, expected {want}"
        return 1
      println "ok"
      return 0
  | _ =>
      eprintln "Usage: lake exe pct_verify <form.json> <env.json> <r1cs.json> <witness.json>"
      return 1

end PCTVerify
end CLI
end ZK
end Crypto
end HeytingLean
