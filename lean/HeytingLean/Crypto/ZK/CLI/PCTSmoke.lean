import Lean
import Lean.Data.Json
import HeytingLean.Crypto.ZK.CLI.PCTR1CS
import HeytingLean.Crypto.ZK.Export
import HeytingLean.Crypto.ZK.R1CSBool
import HeytingLean.Crypto.BoolLens

namespace HeytingLean
namespace Crypto
namespace ZK
namespace CLI
namespace PCTSmoke

open IO
open Lean
open BoolLens
open R1CSBool

def main (_args : List String) : IO UInt32 := do
  let formPath : System.FilePath := "lean/Examples/PCT/form_and_imp.json"
  let envPath  : System.FilePath := "lean/Examples/PCT/env_2vars.json"
  let formRaw ← FS.readFile formPath
  let envRaw  ← FS.readFile envPath
  let formJson ← match Json.parse formRaw with | .ok j => pure j | .error err => eprintln err; return 1
  let envJson  ← match Json.parse envRaw  with | .ok j => pure j | .error err => eprintln err; return 1
  let n := match envJson.getArr? with | .ok a => a.size | .error _ => 0
  let formJ ← match CLI.PCTR1CS.FormJ.fromJsonE formJson with | .ok f => pure f | .error err => eprintln err; return 1
  let some φ := CLI.PCTR1CS.FormJ.toForm? n formJ | do eprintln s!"Form contains var ≥ n={n}"; return 1
  let ρ ← match CLI.PCTR1CS.parseEnvE n envJson with | .ok r => pure r | .error err => eprintln err; return 1
  let compiled := R1CSBool.compile φ ρ
  -- In-memory check: output alignment (witness output equals BoolLens.eval)
  let want := if BoolLens.eval φ ρ then (1 : ℚ) else 0
  let got := compiled.assignment compiled.output
  if got ≠ want then
    eprintln s!"smoke: output mismatch. got={got}, want={want}"
    return 1
  println "smoke: ok"
  return 0

end PCTSmoke
end CLI
end ZK
end Crypto
end HeytingLean
