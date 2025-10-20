import Lean
import HeytingLean.ProofWidgets.LoFViz.State
import HeytingLean.ProofWidgets.LoFViz.Kernel
import HeytingLean.ProofWidgets.LoFViz.Render.Types
import HeytingLean.ProofWidgets.LoFViz.Render.Boundary
import HeytingLean.ProofWidgets.LoFViz.Render.Hypergraph
import HeytingLean.ProofWidgets.LoFViz.Render.Fiber
import HeytingLean.ProofWidgets.LoFViz.Render.String

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz
namespace Render

open Lean

/-- Simple SVG placeholder for modes not yet implemented. -/
def demoSvg (state : State) (kernel : KernelData) : BridgeResult :=
  let label := toString state.mode
  let svg :=
    s!"<svg viewBox='0 0 360 200' xmlns='http://www.w3.org/2000/svg'>
        <rect x='2' y='2' width='356' height='196' rx='20' fill='#111827' stroke='#1f2937' stroke-width='3'/>
        <text x='180' y='90' fill='#e5e7eb' text-anchor='middle' font-family='monospace' font-size='18'>{label}</text>
        <text x='180' y='125' fill='#9ca3af' text-anchor='middle' font-family='monospace' font-size='12'>{kernel.summary}</text>
      </svg>"
  { svg
    hud :=
      { dialStage := state.dialStage
        lens := state.lens
        mode := state.mode
        notes := kernel.notes }
    certificates := kernel.certificates }

/-- Combined split-mode SVG showing boundary and hypergraph side-by-side. -/
def splitSvg (kernel : KernelData) : String :=
  let leftActive := kernel.currentIsActive
  let rightActive := kernel.aggregate.reentries > 0
  s!"<svg viewBox='0 0 740 220' xmlns='http://www.w3.org/2000/svg'>
      <rect x='2' y='2' width='736' height='216' rx='24' fill='#0b1120' stroke='#1e293b' stroke-width='3'/>
      <g transform='translate(0,0)'>
        <rect x='24' y='22' width='320' height='176' rx='18' fill='rgba(14,116,144,0.15)' stroke='#0ea5e9' stroke-width='2'/>
        <text x='184' y='48' text-anchor='middle' font-family='monospace' font-size='15' fill='#38bdf8'>Boundary View</text>
        <circle cx='184' cy='120' r='64' fill='rgba(56,189,248,0.18)' stroke='#38bdf8' stroke-width='4'/>
        <circle cx='184' cy='120' r='36' fill='{if leftActive then "#38bdf8" else "rgba(56,189,248,0.12)"}' stroke='#0ea5e9' stroke-width='4'/>
        <text x='184' y='180' text-anchor='middle' font-family='monospace' font-size='12' fill='#e0f2fe'>
          current: {if leftActive then "⊤" else "⊥"}
        </text>
      </g>
      <g transform='translate(360,0)'>
        <rect x='24' y='22' width='320' height='176' rx='18' fill='rgba(139,92,246,0.15)' stroke='#8b5cf6' stroke-width='2'/>
        <text x='184' y='48' text-anchor='middle' font-family='monospace' font-size='15' fill='#c4b5fd'>Hypergraph View</text>
        <line x1='60' y1='160' x2='300' y2='80' stroke='{if rightActive then "#f97316" else "#475569"}' stroke-width='4' opacity='0.8'/>
        <circle cx='84' cy='150' r='22' fill='rgba(168,85,247,0.2)' stroke='#c084fc' stroke-width='3'/>
        <text x='84' y='152' text-anchor='middle' font-family='monospace' font-size='11' fill='#f5d0fe'>prev</text>
        <circle cx='184' cy='110' r='26' fill='rgba(56,189,248,0.2)' stroke='#38bdf8' stroke-width='3'/>
        <text x='184' y='112' text-anchor='middle' font-family='monospace' font-size='11' fill='#bae6fd'>current</text>
        <circle cx='284' cy='80' r='22' fill='rgba(34,197,94,0.2)' stroke='#22c55e' stroke-width='3'/>
        <text x='284' y='82' text-anchor='middle' font-family='monospace' font-size='11' fill='#bbf7d0'>process</text>
        <text x='184' y='180' text-anchor='middle' font-family='monospace' font-size='12' fill='#ddd6fe'>
          re-entry edges: {kernel.aggregate.reentries}
        </text>
      </g>
      <text x='370' y='210' text-anchor='middle' font-family='monospace' font-size='12' fill='#94a3b8'>{kernel.summary}</text>
    </svg>"

/-- Route a render request for the selected mode. -/
def route (state : State) (kernel : KernelData) : MetaM BridgeResult :=
  match state.mode with
  | .boundary   => renderBoundary state kernel
  | .euler      => renderBoundary state kernel
  | .hypergraph => renderHypergraph state kernel
  | .fiber      => renderFiber state kernel
  | .string     => renderString state kernel
  | .split      =>
      let hud : Hud :=
        { dialStage := state.dialStage
          lens := state.lens
          mode := state.mode
          notes :=
            kernel.notes ++
              #["Left panel: boundary containment.",
                "Right panel: hypergraph merges journal ordering."] }
      pure { svg := splitSvg kernel, hud, certificates := kernel.certificates }
  | _           => pure <| demoSvg state kernel

end Render
end LoFViz
end ProofWidgets
end HeytingLean
