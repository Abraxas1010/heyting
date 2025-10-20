# LoF Visualization System Implementation Plan (Server-First, Proof-Backed Rendering)

This plan turns the narrative in `HeytingLean/TBD/visualization_system.md` into an execution-ready roadmap.  Deliverables are Lean RPC/rendering modules that compute every visual artefact and certificates, plus a thin widget frontend that only renders server-issued SVG + HUD metadata.

## Architecture Snapshot

- **Lean renderers (`lean/ProofWidgets/LoF`):** Maintain the LoF journal, build the constructive kernel `(R, Ω_R, θ)`, run bridge transports, and return `BridgeResult` records containing:
  - SVG payload for the selected visual mode.
  - Proof badges (RT, adjunction, effect/OML, classicalisation, etc.).
  - HUD metadata (dial stage, lens labels, oscillation traces, etc.).
- **RPC (`ProofWidgets/LoF/Rpc.lean`):** Expose a single `[widget.rpc] def apply` that consumes `LoFEvent`s (`Primitive`, `Dial`, `Lens`, `Mode`) and replies with the updated render bundle.
- **Widget shell (`widget/src/LoFVisualApp.tsx` + helpers):** Emit `LoFEvent` JSON, embed returned SVG, and display proof badges. No mathematics or approximations run on the client.

## Tooling & Dependencies

- Lean 4 + `mathlib4` (already in use) + `proofwidgets`.
- Frontend: existing widget build (Vite + React). No additional packages beyond SVG helpers.
- Internal rendering DSL reuses combinators in `ProofWidgets/Canvas`.

## Execution Roadmap

### 0. Repository Scaffolding *(status: ✅ infrastructure in place)*
- Lean modules for `HeytingLean/ProofWidgets/LoFViz/` (state machine, kernel, renderers, RPC, tests) are live under the new namespace.
- Widget sources (`ProofWidgets/widget/src/{LoFVisualApp,types,proofBadges}.tsx`) now ship with the repo and export the `"LoFVisualApp"` component consumed by `VisualizationDemo.lean`.
- Developers can import `HeytingLean.ProofWidgets.LoFViz` for the umbrella module or tree-shake individual renderers.

### 1. Lean State & Journal *(status: ✅ baseline state machine; persistence TBD)*
1. `Primitive`, `DialStage`, `Lens`, `VisualMode`, and RPC `Event` types exist with JSON derivations.
2. `State` tracks the primitive journal (`Array JournalEntry`), dial/lens/mode, and a monotone timestamp counter.
3. `Stepper.applyEvent : State → Event → MetaM State` handles primitive appends and mode/lens/dial switches.
4. TODO: add explicit serialization utilities plus a persistence hook once we replace the in-memory cache (see Stage 4).

### 2. Kernel Construction *(status: ✅ first structured nucleus)*
1. `KernelData.fromState` replays the journal into an explicit finite-set nucleus (regions `α…δ`), preserving a stack, cardinalities, and re-entry closures.
2. `KernelData` exposes `nucleus`, `implication`, `meet`, `breathingAngle`, and human-readable subset renderings.
3. Certificates now check adjunction (`prev ⊆ (current ⇒ nucleus current)`), nucleus stability (RT-1/RT-2), and surface a classicalisation flag from the dial stage.
4. Future refinements: extend the region lattice beyond the toy model and introduce bridge helpers (`himp`, `mvAdd`, `effectAdd?`, etc.) that discharge bona fide Lean proofs.

### 3. Renderers *(status: ✅ server-driven visuals with HUD data)*
Each mode returns SVG strings + HUD metadata derived from the enriched kernel:

| Mode              | Current implementation                                                                                     | Follow-ups                          |
|-------------------|-------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `Boundary`/`Euler`| Radii and fills respond to nucleus cardinality; subtitles carry breathing summaries.                       | Upgrade to `ProofWidgets.Svg` primitives for composition. |
| `Hypergraph`      | Node labels render concrete subsets; edges highlight live dependencies and re-entry.                       | Integrate Alexandroff order once the full lattice lands.  |
| `Fiber`           | HUD badges report lens-specific invariants (closure, boundedness, parity).                                  | Connect to genuine bridge transports.                     |
| `String`          | Timeline overlays remain journal-driven; process/counter annotations pulled from state.                     | Extend with bridge-aware animations.                      |
| `Split`           | Combines route outputs while preserving proof metadata.                                                     | Allow user-configurable pairings beyond Boundary/Hypergraph. |

### 4. RPC Wiring *(status: ✅ minimal in-memory RPC; persistence + error handling pending)*
1. `Render.route` delegates to the renderer modules; `BridgeResult`/`RenderSummary` structures defined in `Render/Types.lean`.
2. `[widget.rpc_method] def apply` maintains an `IO.Ref` cache keyed by `sceneId`, runs the state stepper, rebuilds kernel data, and returns `{ render, proof }`.
3. Next steps:
   - Swap to `ProofWidgets.PersistentStore` (or similar) for multi-process robustness.
   - Emit richer diagnostics / `diagnostics` traces when certificates flag failures.

### 5. Frontend Shell *(status: ✅ wired to RPC)*
1. `LoFVisualApp.tsx` emits Lean events, displays server SVG, and streams HUD/proof badges with loading affordances.
2. Shared `types.ts` mirrors Lean JSON encodings; `proofBadges.tsx` formats certificate status/message text.
3. The widget registers against `HeytingLean.ProofWidgets.LoFViz.apply`, so `VisualizationDemo.lean` loads cleanly in InfoView.

### 6. Testing & Validation *(status: 🚧 minimal Lean sanity lemmas)*
1. `Tests.lean` covers summarised text properties (`KernelData` notes/messages length).
2. Outstanding testing work:
   - Property tests around journal transitions and certificate expectations.
   - Renderer snapshot/golden tests once SVG stabilises.
   - Frontend integration tests (Playwright/Cypress) and CI wiring for `lake build -- -DwarningAsError=true`.

### 7. Documentation & Handover *(status: ⏳ not started)*
1. Still need `Docs/Visualization.md` describing the end-to-end flow.
2. Update `README.md` with widget launch instructions.
3. Draft contributor notes for extending modes, lenses, or certificates.

## Milestones

1. **M1 – State Machine & Kernel** (events, journal, kernel certificates).
2. **M2 – Core Renderers** (`Boundary`, `Euler`, `Hypergraph`).
3. **M3 – Bridge Visuals** (`FiberBundle`, lens toggles with RT badges).
4. **M4 – Process Strings & Split View**.
5. **M5 – RPC + Widget Integration**.
6. **M6 – Testing, Golden SVGs, Documentation, Demo clip**.

## Success Criteria

- Every `LoFEvent` is played in Lean, never on the client.
- Renderers emit proof-backed SVG with RT/adjunction/effect/OML/classicalisation flags.
- Widget UI stays math-free and reacts purely to RPC responses.
- Tests cover kernel invariants, renderer proof metadata, and golden visuals.
- Documentation aligns devs on how to interpret/extend the system.
