import Lean
import HeytingLean.ProofWidgets.LoFViz.State
import HeytingLean.ProofWidgets.LoFViz.Kernel

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz
namespace Render

open Lean

/-- Minimal HUD payload delivered with the SVG render. -/
structure Hud where
  dialStage : DialStage
  lens      : Lens
  mode      : VisualMode
  notes     : Array String := #[]
  deriving Inhabited, Repr, ToJson

/-- Raw render result emitted by a renderer. -/
structure BridgeResult where
  svg          : String
  hud          : Hud
  certificates : CertificateBundle
  deriving Inhabited, Repr

/-- Summary returned to the widget. -/
structure RenderSummary where
  sceneId : String
  stage   : String
  lens    : String
  svg     : String
  hud     : Hud
  deriving Inhabited, Repr, ToJson

/-- Full RPC response payload. -/
structure ApplyResponse where
  render : RenderSummary
  proof  : CertificateBundle
  deriving Inhabited, Repr, ToJson

end Render
end LoFViz
end ProofWidgets
end HeytingLean
