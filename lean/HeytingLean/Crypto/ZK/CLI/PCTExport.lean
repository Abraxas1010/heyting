import Lean
import Lean.Data.Json
import HeytingLean.Crypto.BoolLens
import HeytingLean.Crypto.ZK.R1CSBool
import HeytingLean.Crypto.ZK.Export
import HeytingLean.Crypto.ZK.CLI.PCTR1CS

namespace HeytingLean
namespace Crypto
namespace ZK
namespace CLI
namespace PCTExport

open IO
open Lean
open BoolLens
open R1CSBool

/--
Usage: lake exe pct_export <backend> <form.json> <env.json> <outdir>
Currently supported backend: "r1cs". Others are placeholders.
-/
private def findOpt (opts : List String) (prefix : String) : Option String :=
  match opts.find? (fun s => s.startsWith prefix) with
  | none => none
  | some s =>
      let parts := s.splitOn "="
      if parts.length = 2 then some parts.get! 1 else none

private def parseCsv (s : String) : List String :=
  s.splitOn ","

def main (args : List String) : IO UInt32 := do
  match args with
  | backend :: formPath :: envPath :: outDir :: opts =>
      -- reuse PCTR1CS decoders
      let formRaw ← FS.readFile formPath
      let envRaw  ← FS.readFile envPath
      let formJson ← match Json.parse formRaw with | .ok j => pure j | .error err => eprintln err; return 1
      let envJson  ← match Json.parse envRaw  with | .ok j => pure j | .error err => eprintln err; return 1
      let n := match envJson.getArr? with | .ok a => a.size | .error _ => 0
      let formJ ← match CLI.PCTR1CS.FormJ.fromJsonE formJson with | .ok f => pure f | .error err => eprintln err; return 1
      let some φ := CLI.PCTR1CS.FormJ.toForm? n formJ | do eprintln s!"Form contains var ≥ n={n}"; return 1
      let ρ ← match CLI.PCTR1CS.parseEnvE n envJson with | .ok r => pure r | .error err => eprintln err; return 1
      match backend with
      | "r1cs" =>
          let compiled := R1CSBool.compile φ ρ
          -- Write R1CS and witness (identical to pct_prove path)
          let r1csJson := Export.systemToJson compiled.system |>.compress
          -- compute numVars as in Export.compiledToJson
          let maxVar := compiled.system.constraints.foldl (init := 0) (fun m c =>
            let step := fun acc (ts : List (Var × ℚ)) => ts.foldl (fun a p => Nat.max a p.fst) acc
            let m1 := step 0 c.A.terms
            let m2 := step m1 c.B.terms
            let m3 := step m2 c.C.terms
            Nat.max m m3)
          let numVars := maxVar + 1
          let witnessJson := Export.assignmentToJson compiled.assignment numVars |>.compress
          let meta := Json.mkObj
            [ ("backend", Json.str "r1cs")
            , ("outputVar", Json.num compiled.output)
            , ("eval", Json.str (toString (BoolLens.eval φ ρ)))
            ] |>.compress
          let outR1cs := System.FilePath.mk outDir / "r1cs.json"
          let outWitness := System.FilePath.mk outDir / "witness.json"
          let outMeta := System.FilePath.mk outDir / "meta.json"
          FS.createDirAll (System.FilePath.mk outDir)
          FS.writeFile outR1cs r1csJson
          FS.writeFile outWitness witnessJson
          FS.writeFile outMeta meta
          println s!"export wrote {outR1cs} {outWitness} {outMeta}"
          return 0
      | "plonk" =>
          -- parse optional: --plonk-gates=name1,name2
          let gates :=
            match findOpt opts "--plonk-gates" with
            | some v => (parseCsv v).map (fun nm => ZK.Plonk.Gate.mk nm)
            | none => [ZK.Plonk.Gate.mk "custom"]
          let sys : ZK.Plonk.System := { gates := gates }
          let out := Export.plonkSystemToJson sys |>.compress
          FS.createDirAll (System.FilePath.mk outDir)
          FS.writeFile (System.FilePath.mk outDir / "plonk.json") out
          FS.writeFile (System.FilePath.mk outDir / "meta.json") (Json.compress (Json.mkObj [("backend", Json.str "plonk")]))
          println s!"export wrote {System.FilePath.mk outDir / "plonk.json"}"
          return 0
      | "air" =>
          -- parse optional: --air-width=n --air-length=m
          let width := (findOpt opts "--air-width").bind String.toNat?
          let length := (findOpt opts "--air-length").bind String.toNat?
          let w := width.getD 0
          let l := length.getD 0
          let sys : ZK.AIR.System := { trace := { width := w, length := l } }
          let out := Export.airSystemToJson sys |>.compress
          FS.createDirAll (System.FilePath.mk outDir)
          FS.writeFile (System.FilePath.mk outDir / "air.json") out
          FS.writeFile (System.FilePath.mk outDir / "meta.json") (Json.compress (Json.mkObj [("backend", Json.str "air")]))
          println s!"export wrote {System.FilePath.mk outDir / "air.json"}"
          return 0
      | "bullet" =>
          -- parse optional: --bullet-labels=a,b,c
          let labels :=
            match findOpt opts "--bullet-labels" with
            | some v => (parseCsv v)
            | none => ["C"]
          let commitments := labels.map (fun lab => ZK.Bullet.Commitment.mk lab)
          let sys : ZK.Bullet.System := { commitments := commitments }
          let out := Export.bulletSystemToJson sys |>.compress
          FS.createDirAll (System.FilePath.mk outDir)
          FS.writeFile (System.FilePath.mk outDir / "bullet.json") out
          FS.writeFile (System.FilePath.mk outDir / "meta.json") (Json.compress (Json.mkObj [("backend", Json.str "bullet")]))
          println s!"export wrote {System.FilePath.mk outDir / "bullet.json"}"
          return 0
      | _ =>
          -- Optional: attempt to call an external verifier if requested
          if (opts.any (· = "--run-verifier")) then
            let cmd := (findOpt opts "--verifier").getD (← IO.getEnv "PCT_VERIFIER_CMD").getD ""
            if cmd = "" then
              eprintln s!"no verifier configured (pass --verifier=cmd or set PCT_VERIFIER_CMD)"
              return 3
            else
              -- Build arguments: backend + primary JSON path(s)
              let args := match backend with
                | "r1cs"   => ["r1cs", (System.FilePath.mk outDir / "r1cs.json").toString, (System.FilePath.mk outDir / "witness.json").toString]
                | "plonk"  => ["plonk", (System.FilePath.mk outDir / "plonk.json").toString]
                | "air"    => ["air", (System.FilePath.mk outDir / "air.json").toString]
                | "bullet" => ["bullet", (System.FilePath.mk outDir / "bullet.json").toString]
                | _        => [backend]
              let child ← IO.Process.spawn { cmd := cmd, args := args.toArray }
              let code ← child.wait
              match code with
              | 0 => return 0
              | _ =>
                  eprintln s!"verifier '{cmd}' failed with exit code {code}"
                  return (UInt32.ofNat code.toNat)
          else
            eprintln s!"backend '{backend}' not implemented; use 'r1cs'|'plonk'|'air'|'bullet'"
            return 2
  | _ =>
      eprintln "Usage: lake exe pct_export <backend> <form.json> <env.json> <outdir>"
      return 1

end PCTExport
end CLI
end ZK
end Crypto
end HeytingLean
