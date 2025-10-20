import ProofWidgets
import HeytingLean.ProofWidgets.LoFViz.Rpc

/-- Interactive LoF visualization widget backed by `HeytingLean.ProofWidgets.LoFViz.apply`. -/
#widget
  (rpc := ``HeytingLean.ProofWidgets.LoFViz.apply)
  (component := "LoFVisualApp")
  (%json| {
    "sceneId": "demo",
    "clientVersion": "0.1.0"
  })
