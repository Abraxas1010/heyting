# Integrated Ontological & Lean Formalization Plan

## Ontological Snapshot

- Distinction-as-Re-entry realised via nucleus structures in `HeytingLean/LoF/PrimaryAlgebra.lean` and `HeytingLean/LoF/Nucleus.lean`, exposing complementary fixed points (`process`, `counterProcess`) and Euler-boundary lemmas.
- Euler Boundary implemented as the least nontrivial fixed point; positivity, equality with `process`, and interaction with `counterProcess` are proven.
- Modal breathing ladder (`Logic/ModalDial.lean`) links the `θ` parameter to dial stages (0D→3D).
- Stage transport (`Logic/StageSemantics.lean`) packages MV/effect/orthomodular operations and provides bridge transport lemmas; lint cleanup is underway so that Tensor/Graph/Clifford bridges and compliance lemmas align cleanly with the new `Option.map (fromCore ·)` transport while retaining commuting lemmas with `logicalShadow`.
- Compliance tests cover the stage transport and shadow-commutation guarantees; documentation tasks remain.

## Codebase Audit *(April 2025)*

- `lake build` (Lean 4) is back to green after the transport refactor; there are still no `sorry`/`admit` placeholders or custom `axiom` declarations in compiled Lean files (narrative `TBD/` notes still contain `sorry` stubs).
- Stage helpers and shadow-commutation lemmas for Tensor/Graph/Clifford have been updated to the new transport signature, and the compliance suite exercises the revised Option-handling facts.
- Recommended CI command: `lake build -- -Dno_sorry -DwarningAsError=true` to keep “compiled = proven”.

## Objective
- Ground the metastructure from `lean/lean.md` in Lean so that the re-entry nucleus, Heyting core, transitional ladder, and cross-lens translations become machine-checked definitions, instances, and theorems.
- Produce a coherent Lean codebase whose modules line up with the logical (LoF), algebraic (residuated/effect), geometric (Clifford/orthomodular), and computational (tensor/graph) lenses while preserving the contracts (RT-1/RT-2, TRI-1/TRI-2, DN).

## Guiding Principles
- Treat the re-entry operator as a nucleus/interior map and make its fixed points the constructive logic (Ω_R). Every other component (modalities, lenses, limits) should be phrased as structure transported along, or compatible with, this nucleus.
- Prefer reusing mathlib typeclasses (`frame`, `heyting_algebra`, `residuated_lattice`, `mv_algebra`, `effect_algebra`, `orthomodular_lattice`, `module`, `inner_product_space`, `submodule`) where available; introduce new typeclasses only when an existing one cannot be adapted cleanly.
- Keep proofs compositional: prove core algebraic facts once in the abstract LoF namespace and instantiate them for tensors, graphs, and Clifford/Hilbert spaces via dedicated functors/interior operators.

## Toolchain & Dependencies
- Target Lean 4 with `mathlib4` (preferred) or Lean 3 with classical mathlib if the project already uses it. Confirm `lakefile.toml`/`lean-toolchain` configuration before generating files.
- Inventory mathlib support:
  - `order/nucleus.lean` (or the Lean 3 analog) for interior operators and fixed-point sublattices.
  - `order/heyting` for Heyting algebra lemmas and residuation.
  - `algebra/order/residuated` and `data/mv_polynomial` or `measure_theory` modules for residuated lattices and MV-algebras (verify availability; otherwise stub structures locally).
  - `linear_algebra/orthogonal`, `analysis/inner_product_space/projection`, and `topology/algebraic_topology` pieces for closed subspaces, projectors, and orthomodularity.
  - `order/topology/interval` and `order/category` for Alexandroff topologies and interior operators on preorders.
- Flag gaps early (e.g. if `effect_algebra` is missing) and plan to either implement minimal definitions or search for community contributions.

## Proposed Directory Layout
```
lean/
  lakefile.toml / lake-manifest.json (Lake ≥5 default)
  LoF/
    PrimaryAlgebra.lean
    Nucleus.lean
    HeytingCore.lean
  Logic/
    ResiduatedLadder.lean
    ModalDial.lean
    StageSemantics.lean
  Quantum/
    Orthomodular.lean
    ProjectorNucleus.lean
  Bridges/
    Tensor.lean
    Graph.lean
    Clifford.lean
  Contracts/
    RoundTrip.lean
    Examples.lean
  Docs/
    README.md (generated from this plan once stabilized)
```
Adjust if the repository already uses a different naming convention; the key requirement is to keep LoF foundations separate from lens-specific files.

## Execution Roadmap

### 0. Environment Setup *(status: ✅ baseline in place, CI runs `lake build`)*
- Initialize or update the Lean project using `lake init` / `lake exe cache get`. *(Done – `lakefile.lean` and `lake-manifest.json` configured.)*
- Set up CI or a local script that runs `lake build` and `lake exe lint` (or `lean --make`) to ensure definitions compile continuously. *(Done – `.github/workflows/lean_action_ci.yml` now invokes `lake build`. Consider adding a lint target once proofs grow.)*

### 1. Primary Algebra (LoF) Foundation *(status: ✅ implemented)*
- `PrimaryAlgebra` now extends `Order.Frame`, and `Reentry` wraps mathlib nuclei (`lean/HeytingLean/LoF/PrimaryAlgebra.lean`, `.../Nucleus.lean`).
- Core lemmas (`idempotent`, `map_inf`, monotonicity) are in place; `Ω_R` realizes the fixed-point sublocale via `toSublocale`.
- **Remaining work:** expose additional helper lemmas for downstream use (`map_sup`, `map_bot`) if future lenses need them.

### 2. Heyting Core on Fixed Points *(status: ✅ core API + Boolean limit witness)*
- `lean/HeytingLean/LoF/HeytingCore.lean` provides `instHeytingOmega`, `heyting_adjunction`, `residuation`, double negation, and the explicit Boolean equivalence witness when `R = id`.
- **Remaining work:** surface auxiliary simp lemmas (`map_sup`, `map_bot`) on demand for downstream automation.

### 3. Residuated & Transitional Ladder *(status: ⚠️ partially complete)*
- Deduction/abduction/induction equivalence is formalized (`lean/HeytingLean/Logic/ResiduatedLadder.lean`); modal ladder increments exist (`lean/HeytingLean/Logic/ModalDial.lean`).
- **Remaining work:** integrate the MV/effect/orthomodular ladder parameters into the modal collapse/expansion lemmas and reuse those results when wiring lenses (see Research & Open Questions).

### 4. Modal Layer (Breathing Operators) *(status: ✅ scaffolding + dial ladder; ⚠️ richer laws pending)*
- `lean/HeytingLean/Logic/ModalDial.lean` includes `Dial`, the breathing lemmas, and the `DialParam.ladder` (0D→3D) monotone chain.
- **Remaining work:** state modal collapse/expansion laws, relate them to concrete dimensional semantics, and integrate those results with the Stage semantics module.

### 5. Lens-Specific Realizations *(status: ✅ bridge transport aligned)*
- Identity bridge plus tensor/graph/clifford carriers with round-trip proofs exist (`lean/HeytingLean/Bridges/...`, `Contracts/Examples.lean`).
- `lean/HeytingLean/Logic/StageSemantics.lean` supplies reusable MV/effect/orthomodular structures and bridge transport lemmas; `lean/HeytingLean/Logic/Trace.lean` introduces independence/trace-monoid tooling so bridge updates can be expressed via causal invariance; Tensor/Graph/Clifford modules expose the base `stage*` helpers and commuting lemmas with `logicalShadow`, now specialised for the updated transport.
- **Remaining work:**
  - Document the intended dial behaviours for the canonical ladder specialisations and reuse them in higher-order bridge proofs.
  - Tensor: replace tuples with the intended ordered carriers (e.g. `ℕ`/`ℤ`-indexed intensity vectors) and supply the compatibility proofs promised in the roadmap.
  - Graph: integrate Alexandroff/topological structure plus message-passing invariants so the bridge mirrors the ontological account.
  - Clifford: prepare for projector semantics by factoring carrier/projector data into the forthcoming `Quantum/` modules.

### 6. Cross-Lens Contracts *(status: ⚠️ base cases proven; stage interactions partly captured)*
- Identity contract + bridges’ `logicalShadow` lemmas cover RT-1 style properties; compliance tests now assert the stage-transport commutation facts provided in `StageSemantics`.
- **Remaining work:** use the trace-monoid concurrency infrastructure to generalize RT-1/RT-2 and TRI-1/TRI-2 across arbitrary dial stages for every bridge, automate the resulting proofs (`@[simp]`, custom tactics), and expand compliance tests with Boolean/MV/effect/orthomodular exemplars that exercise the new stage helpers.

### 7. Limits, Dialing, and Examples *(status: ⚠️ partial)*
- Dial ladder examples exist (`DialParam.ladder`). Contracts examples cover basic round-trip cases; stage helpers are available but not yet showcased.
- **Remaining work:** add Boolean limit + MV/effect/orthomodular examples (reusing the new stage helpers) and breathing-cycle scenarios demonstrating meet/join dominance.

### 8. Validation & Automation *(status: ⚠️ warnings outstanding)*
- `lake build` runs cleanly again; StageSemantics has been refactored for lint-friendliness, and bridge modules are being updated to eliminate `unused simp` and `simpa` warnings.
- **Next steps:** finish trimming the remaining bridge/compliance lint noise, add structured automation (`@[simp]`, `@[aesop?]`), expand test coverage (RT/TRI proofs, stage interactions, Boolean/MV/effect/orthomodular limits), and standardise running `lake build -- -Dno_sorry -DwarningAsError=true` (plus optional lint) after each milestone.

### 9. Documentation & Developer Support *(status: ⚠️ to-do)*
- Docstrings adorn new modules; full documentation export still pending.
- **Next steps:** mirror the updated roadmap into `Docs/README.md`, add narrative examples, and plan doc-gen scripts once the remaining algebraic layers are built.

### 10. Epistemic Laws from Re-entry *(status: ✅ implemented; documentation pending)*
- Occam/PSR/Dialectic modules (`Epistemic/Occam.lean`, `Logic/PSR.lean`, `Logic/Dialectic.lean`) are in place and the compliance suite exercises them alongside the canonical bridge specialisations.
- **Remaining work:** extend documentation/examples (Euler boundary narrative) and reference the operators in the higher-level design notes before publishing.

## Research & Open Questions
- Confirm whether mathlib already supplies `EffectAlgebra`/`MVAlgebra` in Lean 4; if missing, scope minimal definitions consistent with the project’s needs.
- Investigate availability of projector averages (`∫_G U g A U g⁻¹`) in mathlib; if absent, outline assumptions (compact group action, Haar measure) and decide whether to axiomatize or implement numerically later.
- Determine the best representation for the tensor interior (`Int`): e.g. define via order-closure on `[0,1]^n` or adapt `SetLike`.
- Evaluate existing orthomodular lattice support; if limited, plan to supply bespoke proofs for `closed_subspace`.
- Decide how stage-aware bridge helpers should behave for non-base dial parameters (what algebraic laws we guarantee at MV/effect/orthomodular stages) and document the chosen behaviour in a design note.
- Specify whether `logicalShadow` ought to commute with / preserve stage operations beyond the base dial (e.g. lax joins, residuation) and, once settled, encode the required lemmas or counterexamples.
- Settle how minimal-birthday witnesses are constructed algorithmically (well-founded minimisation vs. choice) so Occam/PSR modules can provide canonical reasons inside Lean’s constructive fragment.

## Immediate Action Items
- Sweep the lingering lint warnings (`unused simp` arguments, `unnecessary simpa`) across Stage semantics, bridges, and compliance tests so the transport lemmas stay tidy.
- Finalise the stage semantics decisions, implement the non-base dial laws, and mirror them in design notes plus compliance tests (Boolean/MV/effect/orthomodular exemplars).
- Produce `Docs/Ontology.md` summarising the philosophical ↔ Lean correspondence and link it from `Docs/README.md`.
- Update CI / developer docs to standardise on `lake build -- -Dno_sorry -DwarningAsError=true` (and optionally lint) after each major milestone so green builds guarantee all proofs are complete.
- Stand up the Occam/PSR/Dialectic modules with minimal-birthday proofs and regression tests demonstrating Euler-boundary behaviour across the new operators.

## Near-Term Subplan *(May 2025 sprint)*

1. **Transport polish & lint cleanup**
   - Normalise `simp` usage across `StageSemantics` and every bridge so transport lemmas compile without `unused simp` warnings.
   - Extract recurring Option-handling rewrites into helper lemmas to keep subsequent proofs short.

2. **Compliance enrichment**
   - Extend `Tests/Compliance.lean` with Boolean/MV/effect/orthomodular exemplars that exercise the updated transport helpers at each dial stage.
   - Add RT/TRI regression checks once the lint cleanup is merged, aiming to run the suite with `-DwarningAsError=true`.

3. **Documentation & follow-on prep**
   - Capture the transport API decisions in `Docs/README.md` and seed `Docs/Ontology.md` with a short Euler-boundary narrative.
   - Outline the remaining ladder semantics (modal collapse/expansion) so subsequent sprints can implement them without re-auditing the bridges.

## Cleanup Backlog

- Normalize `simp` usage across `Logic/StageSemantics.lean` and the bridge suites so transport lemmas compile without `unused simp` warnings. Extract recurring `Option.map` rewrites into helpers once higher-priority compliance work lands.
## Milestones
- **M1:** Primary algebra and nucleus compiled with Heyting core (`LoF/`).
- **M2:** Residuated ladder and modal dial completed with theorem statements (`Logic/`).
- **M3:** Geometry and orthomodular modules typecheck with nucleus proofs (`Quantum/`).
- **M4:** Tensor and graph bridges established with RT/TRI contracts (`Bridges/` and `Contracts/`).
- **M5:** Occam/PSR/Dialectic implemented with accompanying compliance proofs (`Epistemic/`, `Logic/`).
- **M6:** Examples and documentation demonstrate dial-a-logic scenarios, breathing cycles, and epistemic laws.

Maintaining these milestones ensures the Lean codebase evolves in sync with the conceptual framework described in `lean.md`.
