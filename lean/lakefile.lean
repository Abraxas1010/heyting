import Lake
open Lake DSL

package «HeytingLean» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.24.0"

@[default_target]
lean_lib «HeytingLean» where

lean_exe «heytinglean» where
  root := `Main

lean_exe «generateDepGraph» where
  root := `Tools.GenerateDepGraph
  supportInterpreter := true

lean_exe «pct_r1cs» where
  root := `HeytingLean.Crypto.ZK.CLI.PCTR1CS

lean_exe «pct_prove» where
  root := `HeytingLean.Crypto.ZK.CLI.PCTProve

lean_exe «pct_verify» where
  root := `HeytingLean.Crypto.ZK.CLI.PCTVerify
