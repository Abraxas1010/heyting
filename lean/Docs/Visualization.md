# LoF Visualization Scaffold

The visualization subsystem (see `visualization_system_plan.md`) now has a functional nucleus-backed core:

- `ProofWidgets/LoFViz/State.lean` tracks the scene journal, dial, lens, and mode.
- `ProofWidgets/LoFViz/Kernel.lean` interprets the journal in the canonical `Set Unit` LoF nucleus, emits kernel summaries, and computes certificate bundles (adjunction/RT/classicalisation flags).
- `ProofWidgets/LoFViz/Render/Boundary.lean` produces a nucleus-backed Boundary/Euler SVG using the computed kernel.
- `ProofWidgets/LoFViz/Render/Hypergraph.lean` renders the re-entry preorder as a hypergraph with edges guided by the nucleus aggregates.
- `ProofWidgets/LoFViz/Render/Fiber.lean` visualises the bridge transports as a bundle of fibers anchored to the LoF core.
- `ProofWidgets/LoFViz/Render/String.lean` renders process/counter-process strands over the primitive timeline.
- `ProofWidgets/LoFViz/Rpc.lean` exposes the `LoFViz.apply` RPC and now persists scene state via the proof-widget persistent store.
- `ProofWidgets/widget/src/LoFVisualApp.tsx` wires the UI shell to the RPC and renders proof badges.

All primary modes (Boundary, Euler, Hypergraph, Fiber, String) are backed by nucleus-derived renderers. The Split mode provides a comparative dashboard; future work can enrich it with live sub-mode selection.
