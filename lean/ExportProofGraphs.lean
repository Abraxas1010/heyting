import Lean
import HeytingLean.ProofWidgets.LoFViz.Proof.Core
import HeytingLean.ProofWidgets.LoFViz.Proof.Graph

open Lean Elab Command HeytingLean.ProofWidgets.LoFViz
open HeytingLean.ProofWidgets.LoFViz.Proof
open HeytingLean.ProofWidgets.LoFViz.Proof.Graph

private def trimGraph (g : ProofGraph) (nodeLimit : Nat := 80) : ProofGraph := Id.run do
  let nodes := g.nodes.toList.take nodeLimit
  let keepIds := nodes.map (·.id)
  let keepSet : Std.HashSet NodeId := Std.HashSet.ofList keepIds
  let edges := g.edges.filter fun e => keepSet.contains e.src && keepSet.contains e.dst
  let root := match g.root with
    | some r => if keepSet.contains r then some r else none
    | none => none
  { g with nodes := nodes.toArray, edges := edges, root }

#eval show CommandElabM Unit from do
  let names := [``Nat.mul_comm, ``Nat.add_comm, ``Nat.mul_left_comm]
  for name in names do
    let bundle ← bundleOfConstant name
    let trimmed := trimGraph bundle.graph 80
    let doc := Json.compress (ProofGraph.toJson trimmed)
    logInfo m!"{name.toString}\n{doc}"
