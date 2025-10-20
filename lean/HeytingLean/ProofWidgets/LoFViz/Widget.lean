import ProofWidgets

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz

/-- Props accepted by the LoF visual widget. -/
structure WidgetProps where
  sceneId : String := "demo"
  clientVersion : String := "0.1.0"
  deriving _root_.Lean.Server.RpcEncodable

/-- JavaScript-backed widget component powered by the bundled React client. -/
@[widget_module]
def LoFVisualAppWidget : _root_.ProofWidgets.Component WidgetProps where
  javascript := include_str "LoFVisualApp.js"

end LoFViz
end ProofWidgets
end HeytingLean
