import HeytingLean.ProofWidgets.LoFViz.Proof.Core
import Lean.Elab.Command

open Lean Elab Command
open HeytingLean.ProofWidgets.LoFViz.Proof

/-- Emit the JSON representation of the `Nat.mul_comm` proof widget. -/
#eval (graphJsonOfConstant ``Nat.mul_comm : CommandElabM Json)
