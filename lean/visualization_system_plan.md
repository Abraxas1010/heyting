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

### 0. Repository Scaffolding *(status: ⏳ planned)*
- Create `lean/ProofWidgets/LoF/` with submodules:
  - `State.lean` (journal & kernel state machine)
  - `Kernel.lean` (derives `(R, Ω_R, θ)` from the journal)
  - `Render/Types.lean` (`BridgeResult`, proof badge structures)
  - `Render/Boundary.lean`, `Hypergraph.lean`, `FiberBundle.lean`, `String.lean`
  - `Render/Router.lean`
  - `Rpc.lean`
- Add widget entries:
  - `widget/src/types.ts` (LoFEvent, VisualMode, Lens, LoFPrimitive)
  - `widget/src/LoFVisualApp.tsx` (UI shell)
  - `widget/src/proofBadges.tsx` (HUD renderer)

### 1. Lean State & Journal *(status: ⏳ planned)*
1. Implement `LoFPrimitive`/`LoFEvent` mirroring the TypeScript types.
2. Define `LoFJournal` as a list of primitive entries with timestamps and dial/lens transitions.
3. Provide `Stepper.applyEvent : LoFState → LoFEvent → MetaM LoFState` maintaining:
   - current dial stage (S0–S3),
   - active lens (`Logic`/`Tensor`/`Graph`/`Clifford`),
   - view mode (`Boundary`/`Euler`/`Hypergraph`/`Fiber`/`String`/`Split`),
   - cached `KernelData` (below).
4. Implement serialization helpers for persistence (Lean JSON instances).

### 2. Kernel Construction *(status: ⏳ planned)*
1. Build `Kernel.fromJournal (j : LoFJournal) : MetaM KernelData` that:
   - Interprets the sequence of `Mark`/`Unmark`/`Reentry` as LoF regions `(a,b,…)`.
   - Computes the nucleus `R`, fixed-point lattice `Ω_R`, Euler boundary, and breathing angle `θ`.
   - Supplies transport handles for Tensor/Graph/Clifford bridges using existing encode/decode contracts.
2. Provide convenience functions:
   - `Kernel.himp`, `Kernel.mvAdd`, `Kernel.effectAdd?`, `Kernel.orthocomplement`.
   - `Kernel.certificates : KernelData → CertificateBundle` (adjunction, RT-1/RT-2, effect definedness, OML, classicalisation).

### 3. Renderers *(status: ⏳ planned)*
For each visual mode implement `renderX : KernelData → MetaM BridgeResult`.

| Mode              | Renderer responsibilities                                                                                             | Proof badges          |
|-------------------|------------------------------------------------------------------------------------------------------------------------|-----------------------|
| `Boundary`/`Euler`| • Draw nested SVG paths for regions `(a,b,R a,R (¬a ∨ b))`.<br>• Encode breathing cycles via θ timeline.               | adjunction, EM/¬¬, RT |
| `Hypergraph`      | • Build nodes/edges from re-entry preorder & Alexandroff opens.<br>• Mark loops for re-entry.                          | adjunction, RT (graph)|
| `FiberBundle`     | • Draw base + 3 fibers (Tensor/Graph/Clifford) with arrows labelled `encode/decode` & RT statuses.                     | RT-1/RT-2 per lens    |
| `String`          | • Serialize process/counter-process as string diagram (cuts, braids, collapses).                                      | residuation triangle  |
| `Split`           | • Compose two renderer outputs side by side (call `route` twice).                                                     | union of both         |

All SVG generation should rely on pure functions producing `ProofWidgets.Svg.Svg`. Attach HUD metadata (dial stage, active lens, classicalisation flag, oscillation trace coordinates).

### 4. RPC Wiring *(status: ⏳ planned)*
1. Implement `Render.route : VisualMode → KernelData → MetaM BridgeResult`.
2. Implement `[widget.rpc] def apply`:
   - Load/save state from `ProofWidgets.PersistentStore`.
   - Apply the event, recompute kernel, route renderer.
   - Return JSON serializable structure with `{ render := RenderResult, proof := CertificateBundle }`.
3. Add unit tests in `ProofWidgets/LoF/Tests.lean` covering:
   - Primitive sequences leading to known kernels.
   - Dial & lens switches.
   - Renderer snapshot hashes for golden cases.

### 5. Frontend Shell *(status: ⏳ planned)*
1. Create `LoFVisualApp.tsx` with controls:
   - Primitive buttons (`Unmark`, `Mark`, `Re-entry`).
   - Mode selector (Boundary/Euler/Hypergraph/Fiber/String/Split).
   - Lens selector (Logic/Tensor/Graph/Clifford).
   - Dial slider/toggle (S0–S3).
2. Use the existing widget RPC hook to call `"LoF.apply"`.
3. Inject returned SVG via `dangerouslySetInnerHTML`, render proof badges using metadata.
4. Implement split view layout (two SVG containers) driven solely by server payload.

### 6. Testing & Validation *(status: ⏳ planned)*
1. Extend `Tests/Compliance.lean` with visualization-oriented property tests:
   - `renderBoundary` returns adjunction badge `true` when `Kernel.certificates.adjunction`.
   - `renderFiberBundle` reports RT statuses matching bridge contracts.
2. Introduce golden SVG tests:
   - Store hashed outputs under `tests/golden/lof/*.svg`.
   - Use Lean tests to compare against baseline for deterministic states.
3. Frontend verification:
   - Add Playwright (or existing Cypress) script to load each mode and assert proof badge text.
4. CI: ensure `lake build -- -Dno_sorry -DwarningAsError=true` and `npm test` remain mandatory.

### 7. Documentation & Handover *(status: ⏳ planned)*
1. Author `Docs/Visualization.md` summarizing:
   - Event → renderer → proof badge flow.
   - Mode meanings and how to interpret HUD outputs.
2. Update `README.md` with instructions to launch the widget.
3. Provide onboarding notes for future contributors (how to add a new mode or lens).

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
