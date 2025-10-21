# LoF Generative Proof & Visualization Plan (Core-First, Server-Driven)

This plan turns the narrative in `HeytingLean/TBD/visualization_system.md` into an execution-ready roadmap. Deliverables begin with LoF primitives and the re-entry nucleus, add bridge transports (tensor, graph, Clifford, etc.), produce certified kernel data, and only then surface renderer/RPC/UX layers. Visuals are consumers, not the centrepiece.

## Architecture Snapshot

- **LoF core (`HeytingLean/LoF`, `HeytingLean/Logic`):** Formalise the re-entry nucleus, Heyting core `Ω_R`, and staged MV/effect/OML overlays that bridges must transport.
- **Bridge layer (`HeytingLean/Bridges`, `HeytingLean/Contracts`):** Provide round-trip contracts for tensor tuples, graph carriers, Clifford bivectors, etc., each commuting with stage semantics and collapsing to the Heyting core.
- **ProofWidgets backend (`lean/HeytingLean/ProofWidgets/LoFViz`):** Replay journal primitives, build `KernelData`, emit certificate bundles, and package renderer-ready models that stay in sync with the algebra.
- **RPC + Widget (`ProofWidgets/LoF/Rpc.lean`, `widget/src/LoFVisualApp.tsx`):** Stream JSON events into Lean, surface proof-backed results, and keep the frontend math-free.
- **Documentation & Blueprint (`Docs/LoFBlueprint/*.html`, `Docs/Visualization.md`):** Track the generative spine, bridge cards, and testing commitments so every subsystem is covered.

## Tooling & Dependencies

- Lean 4 + `mathlib4` (already in use) + `proofwidgets`.
- Frontend: existing widget build (Vite + React). No additional packages beyond SVG helpers.
- Internal rendering DSL reuses combinators in `ProofWidgets/Canvas`.

## Execution Roadmap

### 0. Repository Scaffolding *(status: ✅ infrastructure in place)*
- Lean namespaces for the LoF core (`HeytingLean/LoF`), staged logic overlays (`HeytingLean/Logic`), bridge layer (`HeytingLean/Bridges`), and visualization backend (`HeytingLean/ProofWidgets/LoFViz`) are all present under `lean/`.
- Widget sources (`ProofWidgets/widget/src/{LoFVisualApp,types,proofBadges}.tsx`) ship with the repo and export the `"LoFVisualApp"` component consumed by `VisualizationDemo.lean`.
- Developers can import `HeytingLean.ProofWidgets.LoFViz` for the umbrella backend or tree-shake individual subsystems.

### 1. LoF State & Journal *(status: ✅ baseline state machine; persistence TBD)*
1. `Primitive`, `DialStage`, `Lens`, `VisualMode`, and RPC `Event` types exist with JSON derivations so every client event is replayed in Lean.
2. `State` tracks the journal, dial/lens/mode selections, and a monotone timestamp counter; lens values already mirror bridge families (`logic`, `tensor`, `graph`, `clifford`).
3. `Stepper.applyEvent : State → Event → State` is pure, making journal replay deterministic for proofs/tests.
4. TODO: add persistence (e.g. `ProofWidgets.PersistentStore`) and export/import utilities so bridge proofs can analyse historical journals.

### 2. Kernel & Heyting Core *(status: ✅ toy nucleus + certificates)*
1. `KernelData.fromState` replays the journal into the canonical toy nucleus (regions `α…δ`) capturing mark/unmark stack discipline and re-entry closure.
2. The module exposes Heyting operations (`nucleus`, `implication`, `meet`) plus HUD summaries and breathing telemetry.
3. `CertificateBundle` reports adjunction, RT₁/RT₂, and classicalisation flags. Next: extend the nucleus beyond the toy lattice and expose Stage semantics operations (`mvAdd`, `effectAdd?`, etc.) directly from `KernelData`.

### 3. Bridge Integrations *(status: ✅ tensor/graph/clifford transports proved; visual surfacing pending)*
1. Bridge carriers (`Bridges/Tensor.lean`, `Bridges/Graph.lean`, `Bridges/Clifford.lean`) plus round-trip contracts are live; they collapse back to the Heyting core via coordinate/projector maps.
2. Stage semantics overlays commute through each bridge (see `stageMvAdd_encode`, `stageEffectAdd_encode`, etc. in the bridge modules).
3. Action items:
   - Surface bridge telemetry in `KernelData` / render models (e.g. tensor bounds, graph adjacency proofs, Clifford parity) so HUDs reflect the proofs.
   - Extend lens selection and blueprint docs to cover future bridges (geometry, category, probabilistic) as their scaffolding arrives.

### 4. Render Models & Visual Output *(status: ✅ renderers share certified models; bridge data WIP)*
Each mode now consumes structured models derived from `KernelData`:

| Mode              | Current implementation                                                                                  | Follow-ups                                                   |
|-------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `Boundary`/`Euler`| `Render/Models.lean` provides `BoundaryModel`; radii/flags reflect nucleus cardinals with coupling lemmas. | Elevate radius logic to Heyting lemmas + stage-aware badges. |
| `Hypergraph`      | `HypergraphModel` renders re-entry dependencies, sharing activity flags with boundary.                   | Add Alexandroff-order overlays once lattice extension lands. |
| `Fiber`           | HUD badges surface synthetic lens invariants.                                                            | Replace with real bridge invariants (tensor boundedness, etc.). |
| `String`          | Journal timeline (mark/unmark/re-entry) with process/counter annotations.                                | Animate bridge state changes post transport proofs.          |
| `Split`           | Side-by-side boundary/hypergraph panels reusing shared models.                                           | Offer custom pairings and bridge telemetry badges.           |

### 5. RPC & Widget *(status: ✅ minimal RPC; robustness + harness WIP)*
1. `Render.route` dispatches to renderers; `Render/Types.lean` defines `BridgeResult`/`RenderSummary`.
2. `[widget.rpc_method] def apply` maintains an `IO.Ref` cache keyed by `sceneId`, runs the stepper, rebuilds `KernelData`, and returns `{ render, proof }`.
3. TODO:
   - Swap to persistent storage for multi-session resilience.
   - Emit structured diagnostics when certificates fail; document schemas for tooling.
   - Build an integration harness that drives the JS widget against Lean RPC.

### 6. Tests & Monitoring *(status: ⚙️ kernel tests live; coverage needs expansion)*
1. `HeytingLean/ProofWidgets/LoFViz/Tests.lean` checks kernel summaries, renderer model invariants, and boundary/hypergraph coupling.
2. CI baseline remains `lake build -- -Dno_sorry -DwarningAsError=true`; keep lean on warnings.
3. TODO:
   - Add golden SVG regression suite once bridge-backed visuals stabilise.
   - Implement property tests around journal/bridge transport invariants.
   - Add RPC + widget integration tests and dashboard/CI visibility.

### 7. Documentation & Blueprint *(status: ⚙️ generative spine live; bridge cards pending)*
1. `Docs/Visualization.md` summarises the LoFViz subsystem; update alongside new bridge proofs.
2. `Docs/LoFBlueprint/{index,global,bridges,bridge-*,renderers,rpc-widget,proof-layer,tests}.html` provide decision cards plus a global dependency-graph blueprint plan inspired by Infinity Cosmos.
3. TODO:
   - Add hover tooltips and cross-links within the blueprint.
   - Publish bridge-focused cards once transport lemmas land.
   - Implement the global dependency graph using the blueprint SDK (hover summaries, clickable nodes opening cards, status legend).

## Proof-to-Visualization Expansion Plan (LoF Operators First)

This companion roadmap describes how we ingest an arbitrary Lean proof (either an existing theorem or a pasted proof term), normalise it into the three core LoF operators (`Mark`, `Unmark`, `Re-entry`), and emit all visualisations (boundary, hypergraph, fibre, string, split) in a single static frame. No interactive controls are required for this flow.

### A. Proof Ingestion *(status: ✅ basic constant loader implemented)*
1. Current implementation supports **existing compiled theorems** only. `Proof.Normalized.ofConstant` fetches the statement + proof term and records a LoF journal.
2. Ad-hoc/pasted proof scripts remain future work (requires elaboration pipeline + error reporting).
3. `Proof.Handle` caches statement + provenance string for downstream display.

### B. LoF Normalisation *(status: ✅ heuristic walker in place)*
1. `visitExpr` performs a fuelled WHNF walk mapping lambdas/foralls/lets to `Mark`/`Unmark` and self recursion to `Re-entry`.
2. Journals default to `[Mark, Re-entry, Unmark]` if no structure is detected (prevents empty feeds).
3. Extension hooks (per-tactic mapping, richer annotations) still TODO.

### C. Visual Bundle Generation *(status: ✅ static HTML dashboard)*
1. Journals are replayed through `applyJournal`, producing a frozen `State` + `KernelData`.
2. All renderers run locally (no RPC), generating an array of `(VisualMode × BridgeResult)`.
3. `VisualizationBundle.toHtml` converts the outputs into plain `ProofWidgets.Html` (SVG injected via `dangerouslySetInnerHTML`).
4. No caching or persistence yet; everything recomputes per invocation.

### D. Static UI Frame *(status: ⚠️ rudimentary HTML only)*
1. The dashboard renders directly in the InfoView (`#html Proof.htmlOfConstant …`).
2. No selection UI or side-by-side grid yet; future work includes richer layout + proof picker.
3. Interactive widget controls remain for the manual mode (no automatic transition support yet).

## Visualization Gap Assessment *(status: 🟥 needs rebuild)*

- **Current Surface** — The static HTML export produced by `RenderNatMulCommHtml.lean` only shows the boundary visualization, LoF journal, and a minimal proof-graph summary. All other planned views (hypergraph, Euler/tensor overlays, causal summaries, fibre/string/split) are missing.
- **Planned Features to Restore**
  1. Render additional views within the export (hypergraph layout, Euler/tensor dashboards, causal dependency digests, etc.).
  2. Add proof gallery + upload/paste workflow with validation and user feedback.
  3. Port layered DAG / force-directed layout logic from the widget prototype into the static page.
- **Action Items**
  - Rebuild the static export before resuming widget development.
  - Update documentation to describe the new export workflow once rebuilt.

### E. Testing & Tooling *(status: ⏳ planned)*
1. Regression snapshots: given a proof handle, verify that the generated journal matches expectations and that renderer SVG hashes remain stable.
2. Property checks on the LoF translation (e.g. journal never empty for non-trivial proofs, `Re-entry` count matches recursion depth).
3. CLI utility `lake exe render-proof <name>` to generate bundle JSON for offline analysis.

### F. Milestones
1. **P1** – Existing theorem ingestion **(done for constants)**.
2. **P2** – Heuristic normaliser **(done)**; need tactic-specific extensions.
3. **P3** – Bundle + HTML rendering **(basic version done)**.
4. **P4** – Dedicated proof dashboard UI (pending).
5. **P5** – Regression harness / CLI tooling (pending).

## Milestones

1. **M1 – State Machine & Kernel** (events, journal, kernel certificates).
2. **M2 – Core Renderers** (`Boundary`, `Euler`, `Hypergraph`).
3. **M3 – Bridge Visuals** (`FiberBundle`, lens toggles with RT badges).
4. **M4 – Process Strings & Split View**.
5. **M5 – RPC + Widget Integration**.
6. **M6 – Testing, Golden SVGs, Documentation, Demo clip**.

## Success Criteria

- Every `LoFEvent` is replayed server-side and feeds certified `KernelData`.
- Tensor/graph/Clifford (and future) bridges satisfy round-trip + Stage semantics transport lemmas.
- Renderers emit proof-backed SVG with RT/adjunction/effect/OML/classicalisation and bridge-specific flags.
- Widget UI stays math-free and reacts purely to RPC responses.
- Tests cover kernel invariants, bridge transports, renderer metadata, and golden visuals.
- Documentation + blueprint pages stay current so contributors can navigate core → bridges → visuals.
