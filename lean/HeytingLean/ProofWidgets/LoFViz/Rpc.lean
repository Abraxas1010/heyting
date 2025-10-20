import Lean
import HeytingLean.ProofWidgets.LoFViz.State
import HeytingLean.ProofWidgets.LoFViz.Kernel
import HeytingLean.ProofWidgets.LoFViz.Render.Router
import HeytingLean.ProofWidgets.LoFViz.Render.Types

open Std

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz

open Lean Meta

/-- In-memory cache for scene state. This persists for the lifetime of the Lean server. -/
initialize sceneCache : IO.Ref (Std.HashMap String State) ← IO.mkRef {}

/-- Load a specific scene, falling back to the initial state. -/
def loadState (sceneId : String) : MetaM State := do
  let cache ← liftIO sceneCache.get
  pure <| cache.findD sceneId (initialState sceneId)

/-- Persist the updated scene state in the cache. -/
def saveState (st : State) : MetaM Unit := do
  liftIO <| sceneCache.modify fun m => m.insert st.sceneId st

/-- RPC entry point consumed by the widget. -/
@[widget.rpc_method]
def apply (evt : Event) : MetaM Render.ApplyResponse := do
  let s ← loadState evt.sceneId
  let s' ← Stepper.applyEvent s evt
  let kernel := KernelData.fromState s'
  let rendered ← Render.route s' kernel
  saveState s'
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
