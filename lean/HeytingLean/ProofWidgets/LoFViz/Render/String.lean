import HeytingLean.ProofWidgets.LoFViz.State
import HeytingLean.ProofWidgets.LoFViz.Kernel
import HeytingLean.ProofWidgets.LoFViz.Render.Types

namespace HeytingLean
namespace ProofWidgets
namespace LoFViz
namespace Render

open Lean

@[inline] def eventLabel : Primitive → String
  | .unmark => "Unmark"
  | .mark   => "Mark"
  | .reentry => "Re-entry"

@[inline] def eventColor : Primitive → String
  | .unmark => "#f97316"
  | .mark   => "#22c55e"
  | .reentry => "#a855f7"

/-- Produce SVG snippets for the primitive timeline. -/
def buildTimeline (state : State) : String :=
  let events := state.journal.toList.enum
  let baseLine :=
    "<path d='M40 140 C120 100, 240 100, 320 140' fill='none' stroke='#64748b' stroke-width='3' stroke-dasharray='6 6' opacity='0.6'/>"
  let notches :=
    events.map fun (idx, entry) =>
      let x := toString (80 * idx + 60)
      let color := eventColor entry.primitive
      let label := eventLabel entry.primitive
      s!"<g>
          <line x1='{x}' y1='70' x2='{x}' y2='180' stroke='{color}' stroke-width='4' opacity='0.85'/>
          <circle cx='{x}' cy='120' r='18' fill='{color}' opacity='0.35' stroke='{color}' stroke-width='3'/>
          <text x='{x}' y='118' text-anchor='middle' fill='#e2e8f0' font-family='monospace' font-size='11'>{label}</text>
        </g>"
  baseLine ++ String.intercalate "" notches

/-- Build the string diagram SVG summarising process/counter-process. -/
def stringSvg (kernel : KernelData) (state : State) : String :=
  let header :=
    "<svg viewBox='0 0 360 220' xmlns='http://www.w3.org/2000/svg'>
      <rect x='2' y='2' width='356' height='216' rx='24' fill='#0f172a' stroke='#1f2937' stroke-width='3'/>"
  let processStrand :=
    "<path d='M40 80 C120 40, 240 40, 320 80' fill='none' stroke='#38bdf8' stroke-width='4' opacity='0.9'/>"
  let counterStrand :=
    "<path d='M40 180 C140 210, 220 210, 320 180' fill='none' stroke='#f87171' stroke-width='4' opacity='0.7'/>"
  let crossings :=
    if state.journal.any (·.primitive = .reentry) then
      "<path d='M40 80 C120 140, 240 140, 320 80' fill='none' stroke='#a855f7' stroke-width='3' opacity='0.65'/>"
    else ""
  let timeline := buildTimeline state
  let caption :=
    s!"<text x='180' y='32' text-anchor='middle' font-family='monospace' font-size='16' fill='#e2e8f0'>Process Strings</text>
        <text x='180' y='210' text-anchor='middle' font-family='monospace' font-size='12' fill='#94a3b8'>{kernel.summary}</text>"
  header ++ processStrand ++ counterStrand ++ crossings ++ timeline ++ caption ++ "</svg>"

/-- Render the string diagram view. -/
def renderString (state : State) (kernel : KernelData) : MetaM BridgeResult := do
  let hudNotes :=
    kernel.notes ++
      #[
        "Process strand (blue) follows the primordial witness.",
        "Counter-process (red) mirrors the complementary fixed point.",
        if state.journal.any (·.primitive = .reentry) then
          "Purple strand marks re-entry application."
        else "No re-entry step recorded yet."
      ]
  let hud : Hud :=
    { dialStage := state.dialStage
      lens := state.lens
      mode := state.mode
      notes := hudNotes }
  pure { svg := stringSvg kernel state, hud, certificates := kernel.certificates }

end Render
end LoFViz
end ProofWidgets
end HeytingLean
