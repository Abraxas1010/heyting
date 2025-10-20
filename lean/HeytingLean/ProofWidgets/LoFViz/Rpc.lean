import Lean
import ProofWidgets
import HeytingLean.ProofWidgets.LoFViz.State
import HeytingLean.ProofWidgets.LoFViz.Kernel
import HeytingLean.ProofWidgets.LoFViz.Render.Router
import HeytingLean.ProofWidgets.LoFViz.Render.Types

open Std
open Lean
open Lean Server

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz

/-- In-memory cache for scene state. This persists for the lifetime of the Lean server. -/
initialize sceneCache : IO.Ref (Std.HashMap String State) ← IO.mkRef {}

/-- Load a specific scene, falling back to the initial state. -/
def loadState (sceneId : String) : RequestM State := do
  let cache : Std.HashMap String State ← sceneCache.get
  if h : sceneId ∈ cache then
    pure <| cache.get sceneId h
  else
    pure <| initialState sceneId

/-- Persist the updated scene state in the cache. -/
def saveState (st : State) : RequestM Unit := do
  sceneCache.modify fun m => m.insert st.sceneId st

/-- RPC entry point consumed by the widget. -/
@[server_rpc_method]
def apply (evt : Event) : RequestM (RequestTask Render.ApplyResponse) :=
  RequestM.asTask do
    let s ← loadState evt.sceneId
    let s' := Stepper.applyEvent s evt
    let kernel := KernelData.fromState s'
    let rendered := Render.route s' kernel
    let _ ← saveState s'
    pure
      { render :=
          { sceneId := s'.sceneId
            stage := toString s'.dialStage
            lens := toString s'.lens
            svg := rendered.svg
            hud := rendered.hud }
        proof := rendered.certificates }

end LoFViz
end ProofWidgets
end HeytingLean
