import HeytingLean.ProofWidgets.LoFViz.State
import HeytingLean.ProofWidgets.LoFViz.Kernel
import HeytingLean.ProofWidgets.LoFViz.Render.Types

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz
namespace Render

/-- Build an SVG view showing the base LoF manifold with three bridge fibres. -/
def fiberSvg (kernel : KernelData) : String :=
  let base :=
    "<svg viewBox='0 0 420 240' xmlns='http://www.w3.org/2000/svg'>
      <rect x='2' y='2' width='416' height='236' rx='24' fill='#0f172a' stroke='#1e293b' stroke-width='3'/>
      <defs>
        <linearGradient id='fiberGrad' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color='#22d3ee'/>
          <stop offset='100%' stop-color='#4338ca'/>
        </linearGradient>
        <marker id='arrowTip' markerWidth='10' markerHeight='10' refX='10' refY='5' orient='auto'>
          <path d='M0,0 L10,5 L0,10 z' fill='#22d3ee'/>
        </marker>
      </defs>"
  let baseManifold :=
    "<circle cx='210' cy='150' r='60' fill='rgba(15,118,110,0.25)' stroke='#0f766e' stroke-width='4'/>
     <text x='210' y='156' text-anchor='middle' font-family='monospace' font-size='13' fill='#ccfbf1'>LoF Core (Ω_R)</text>"
  let fiberBoxes :=
    "<rect x='40' y='42' width='110' height='70' rx='16' fill='rgba(2,132,199,0.18)' stroke='#0284c7' stroke-width='2'/>
     <rect x='170' y='20' width='110' height='70' rx='16' fill='rgba(234,179,8,0.18)' stroke='#d97706' stroke-width='2'/>
     <rect x='300' y='42' width='110' height='70' rx='16' fill='rgba(217,70,239,0.18)' stroke='#c026d3' stroke-width='2'/>"
  let fiberLabels :=
    "<text x='95' y='78' text-anchor='middle' font-family='monospace' font-size='12' fill='#bae6fd'>Tensor fiber</text>
     <text x='225' y='55' text-anchor='middle' font-family='monospace' font-size='12' fill='#fde68a'>Graph fiber</text>
     <text x='355' y='78' text-anchor='middle' font-family='monospace' font-size='12' fill='#f5d0fe'>Clifford fiber</text>"
  let arrows :=
    "<path d='M95 112 L185 140' stroke='url(#fiberGrad)' stroke-width='4' fill='none' marker-end='url(#arrowTip)' opacity='0.8'/>
     <path d='M225 90 L225 120' stroke='url(#fiberGrad)' stroke-width='4' fill='none' marker-end='url(#arrowTip)' opacity='0.8'/>
     <path d='M355 112 L245 140' stroke='url(#fiberGrad)' stroke-width='4' fill='none' marker-end='url(#arrowTip)' opacity='0.8'/>"
  let footer :=
    s!"<text x='210' y='228' text-anchor='middle' font-family='monospace' font-size='12' fill='#94a3b8'>{kernel.summary}</text>"
  base ++ baseManifold ++ fiberBoxes ++ fiberLabels ++ arrows ++ footer ++ "</svg>"

/-- Render the fiber bundle portal. -/
def renderFiber (state : State) (kernel : KernelData) : MetaM BridgeResult := do
  let hud : Hud :=
    { dialStage := state.dialStage
      lens := state.lens
      mode := state.mode
      notes := kernel.notes ++ kernel.fiberNotes }
  pure { svg := fiberSvg kernel, hud, certificates := kernel.certificates }

end Render
end LoFViz
end ProofWidgets
end HeytingLean
