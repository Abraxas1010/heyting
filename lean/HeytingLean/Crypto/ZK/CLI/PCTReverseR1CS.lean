import Lean
import Lean.Data.Json
import HeytingLean.Crypto.ZK.Export

namespace HeytingLean
namespace Crypto
namespace ZK
namespace CLI
namespace PCTReverseR1CS

open IO
open Lean
open ZK
open Export

def main (args : List String) : IO UInt32 := do
  match args with
  | [r1csPath, outPath] =>
      let raw ← FS.readFile r1csPath
      let j ← match Json.parse raw with | .ok jj => pure jj | .error err => eprintln err; return 1
      let some sys := jsonToSystem j | do eprintln "Bad R1CS JSON"; return 1
      -- reverse constraints
      let sys' : System := { constraints := sys.constraints.reverse }
      let out := systemToJson sys' |>.compress
      FS.writeFile outPath out
      return 0
  | _ =>
      eprintln "Usage: lake exe pct_reverse_r1cs <r1cs.json> <out.json>"
      return 1

end PCTReverseR1CS
end CLI
end ZK
end Crypto
end HeytingLean

