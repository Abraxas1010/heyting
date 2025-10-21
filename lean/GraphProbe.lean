import HeytingLean.ProofWidgets.LoFViz.Proof.Core
import Lean.Elab.Command

open Lean Elab Command
open HeytingLean.ProofWidgets.LoFViz.Proof

#eval (graphJsonOfConstant ``Nat.mul_comm : CommandElabM Json)
