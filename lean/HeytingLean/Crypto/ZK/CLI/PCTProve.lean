import Lean
import Lean.Data.Json
import HeytingLean.Crypto.Form
import HeytingLean.Crypto.ZK.R1CSBool
import HeytingLean.Crypto.ZK.Export
import HeytingLean.Crypto.BoolLens
import HeytingLean.Crypto.ZK.CLI.PCTR1CS

namespace HeytingLean
namespace Crypto
namespace ZK
namespace CLI
namespace PCTProve

open IO
open Lean
open BoolLens
open R1CSBool

def writeFile (path : System.FilePath) (content : String) : IO Unit := do
  FS.writeFile path content

def main (args : List String) : IO UInt32 := do
  match args with
  | [formPath, envPath, outDir] =>
      let formRaw ← FS.readFile formPath
      let envRaw  ← FS.readFile envPath
      let formJson ← match Json.parse formRaw with
        | .ok j => pure j
        | .error err => eprintln err; return 1
      let envJson ← match Json.parse envRaw with
        | .ok j => pure j
        | .error err => eprintln err; return 1
      -- infer n
      let n := match envJson.getArr? with | .ok a => a.size | .error _ => 0
      let formJ ← match CLI.PCTR1CS.FormJ.fromJsonE formJson with
        | .ok f => pure f
        | .error err => eprintln err; return 1
      let some φ := CLI.PCTR1CS.FormJ.toForm? n formJ | do eprintln s!"Form contains var ≥ n={n}"; return 1
      let ρ ← match CLI.PCTR1CS.parseEnvE n envJson with
        | .ok r => pure r
        | .error err => eprintln err; return 1
      let compiled := R1CSBool.compile φ ρ
      -- Write R1CS and witness
      let r1csJson := Export.systemToJson compiled.system |>.compress
      -- compute maximum var index referenced in the system
      let maxVar := compiled.system.constraints.foldl (init := 0) (fun m c =>
        let step := fun acc (ts : List (Var × ℚ)) => ts.foldl (fun a p => Nat.max a p.fst) acc
        let m1 := step 0 c.A.terms
        let m2 := step m1 c.B.terms
        let m3 := step m2 c.C.terms
        Nat.max m m3)
      let numVars := maxVar + 1
      let witnessJson := Export.assignmentToJson compiled.assignment numVars |>.compress
      let metaJ := Json.mkObj
        [ ("outputVar", Json.num compiled.output)
        , ("eval", Json.str (toString (BoolLens.eval φ ρ)))
        , ("backend", Json.str "r1cs")
        , ("field", Json.str "prime")
        , ("modulus", Json.str "21888242871839275222246405745257275088548364400416034343698204186575808495617")
        ] |>.compress
      let outR1cs := System.FilePath.mk outDir / "r1cs.json"
      let outWitness := System.FilePath.mk outDir / "witness.json"
      let outMeta := System.FilePath.mk outDir / "meta.json"
      FS.createDirAll (System.FilePath.mk outDir)
      writeFile outR1cs r1csJson
      writeFile outWitness witnessJson
      writeFile outMeta metaJ
      println s!"wrote {outR1cs} {outWitness} {outMeta}"
      return 0
  | _ =>
      eprintln "Usage: lake exe pct_prove <form.json> <env.json> <outdir>"
      return 1

end PCTProve
end CLI
end ZK
end Crypto
end HeytingLean
