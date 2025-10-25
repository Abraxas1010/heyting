import Lean
import Lean.Data.Json

namespace HeytingLean
namespace Crypto
namespace ZK
namespace CLI
namespace PCTMutate

open IO
open Lean

def toggle (s : String) : String :=
  if s = "0" then "1" else if s = "1" then "0" else s

def main (args : List String) : IO UInt32 := do
  match args with
  | [metaPath, witnessPath, outPath] =>
      let metaRaw ← FS.readFile metaPath
      let witRaw  ← FS.readFile witnessPath
      let meta ← match Json.parse metaRaw with | .ok j => pure j | .error err => eprintln err; return 1
      let wit  ← match Json.parse witRaw  with | .ok j => pure j | .error err => eprintln err; return 1
      let outVarJson ← meta.getObjVal? "outputVar"
      let outVar ← match outVarJson with | .ok j => j.getNat? | .error _ => .error "no outputVar"
      let outIdx ← match outVar with | .ok n => pure n | .error err => eprintln err; return 1
      let arr ← match wit.getArr? with | .ok a => pure a | .error err => eprintln err; return 1
      if outIdx ≥ arr.size then
        eprintln s!"index {outIdx} out of range (size={arr.size})"
        return 1
      -- rebuild array with toggled entry at outIdx
      let arr' := arr.mapIdx (fun i j =>
        if i = outIdx then
          let s := match j.getStr? with | .ok ss => ss | .error _ => ""
          Json.str (toggle s)
        else j)
      let outJson := Json.arr arr'
      FS.writeFile outPath (Json.compress outJson)
      return 0
  | _ =>
      eprintln "Usage: lake exe pct_mutate <meta.json> <witness.json> <out.json>"
      return 1

end PCTMutate
end CLI
end ZK
end Crypto
end HeytingLean

