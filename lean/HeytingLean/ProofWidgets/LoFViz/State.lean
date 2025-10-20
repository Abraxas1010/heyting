import Lean

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz

open Lean

/-- Primitive interactions exposed in the UI. -/
inductive Primitive
  | unmark
  | mark
  | reentry
  deriving DecidableEq, Inhabited, Repr, BEq, ToJson, FromJson

/-- Dimensional dial stages. -/
inductive DialStage
  | s0_ontic
  | s1_symbolic
  | s2_circle
  | s3_sphere
  deriving DecidableEq, Inhabited, Repr, BEq, ToJson, FromJson

/-- Visual renderer modes. -/
inductive VisualMode
  | boundary
  | euler
  | hypergraph
  | fiber
  | string
  | split
  deriving DecidableEq, Inhabited, Repr, BEq, ToJson, FromJson

/-- Bridge lens selection. -/
inductive Lens
  | logic
  | tensor
  | graph
  | clifford
  deriving DecidableEq, Inhabited, Repr, BEq, ToJson, FromJson

/-- High-level event kind emitted by the widget. -/
inductive EventKind
  | primitive
  | dial
  | lens
  | mode
  deriving DecidableEq, Inhabited, Repr, BEq, ToJson, FromJson

/-- Event payload forwarded from the widget via RPC. -/
structure Event where
  kind         : EventKind
  primitive?   : Option Primitive := none
  dialStage?   : Option DialStage := none
  lens?        : Option Lens := none
  mode?        : Option VisualMode := none
  clientVersion : String
  sceneId       : String
  deriving Inhabited, Repr, ToJson, FromJson

/-- Internal journal entry for primitives. -/
structure JournalEntry where
  primitive : Primitive
  timestamp : Nat
  deriving Inhabited, Repr, ToJson, FromJson

/-- State tracked per scene. -/
structure State where
  sceneId    : String
  journal    : Array JournalEntry
  dialStage  : DialStage
  lens       : Lens
  mode       : VisualMode
  nextStamp  : Nat
  deriving Inhabited, Repr, ToJson, FromJson

instance : ToString DialStage :=
  ⟨fun
    | .s0_ontic    => "S0_ontic"
    | .s1_symbolic => "S1_symbolic"
    | .s2_circle   => "S2_circle"
    | .s3_sphere   => "S3_sphere"⟩

instance : ToString VisualMode :=
  ⟨fun
    | .boundary   => "Boundary"
    | .euler      => "Euler"
    | .hypergraph => "Hypergraph"
    | .fiber      => "FiberBundle"
    | .string     => "StringDiagram"
    | .split      => "Split"⟩

instance : ToString Lens :=
  ⟨fun
    | .logic    => "Logic"
    | .tensor   => "Tensor"
    | .graph    => "Graph"
    | .clifford => "Clifford"⟩

/-- Default empty state for a new scene. -/
def initialState (sceneId : String := "default") : State :=
  { sceneId
    journal := #[]
    dialStage := .s0_ontic
    lens := .logic
    mode := .boundary
    nextStamp := 0 }

namespace Stepper

/-- Apply a primitive event, appending to the journal. -/
def applyPrimitive (s : State) (p : Primitive) : State :=
  let entry : JournalEntry := { primitive := p, timestamp := s.nextStamp }
  { s with
      journal := s.journal.push entry
      nextStamp := s.nextStamp + 1 }

/-- Interpret an incoming event, updating the state. -/
def applyEvent (s : State) (evt : Event) : MetaM State :=
  match evt.kind with
  | .primitive =>
      match evt.primitive? with
      | some p => pure <| applyPrimitive s p
      | none   => pure s
  | .dial =>
      pure <| match evt.dialStage? with
              | some θ => { s with dialStage := θ }
              | none   => s
  | .lens =>
      pure <| match evt.lens? with
              | some ℓ => { s with lens := ℓ }
              | none   => s
  | .mode =>
      pure <| match evt.mode? with
              | some m => { s with mode := m }
              | none   => s

end Stepper

end LoFViz
end ProofWidgets
end HeytingLean
