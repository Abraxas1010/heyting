import Lean
import ProofWidgets
import HeytingLean.ProofWidgets.LoFViz

-- Interactive LoF visualization widget backed by `HeytingLean.ProofWidgets.LoFViz.apply`.

#widget HeytingLean.ProofWidgets.LoFViz.LoFVisualAppWidget with
  ({ sceneId := "demo", clientVersion := "0.1.0" }
    : HeytingLean.ProofWidgets.LoFViz.WidgetProps)
