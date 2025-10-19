# Bridge Carrier Upgrade Plan

## Overview
- Goal: align each bridge with its roadmap carrier so transport proofs and compliance suites validate against the intended mathematical structures.
- Scope: document upgrade paths for tensor, graph, and Clifford bridges; capture dependencies, verification strategy, and sprint-ready tasks.
- Status: current bridges rely on placeholder carriers that satisfy round-trip (RT/TRI) contracts but lack the final data models described in the overall plan.

## Baseline Snapshot
- Unified helpers (`Contracts/RoundTrip.stageOccam`, stage collapse/expand wrappers) are in place and exercised in `Tests/Compliance.lean`.
- Bridges currently use simplified carriers to keep automation stable; Occam and PSR transports already land on these simplified carriers.
- No outstanding `sorry`/`admit` blocks block the upgrade, but lint cleanup remains in bridges/tests.

## Upgrade Targets

### Tensor Bridge — Intensity Vectors
- **Current carrier**: generic residuated tensor space with abstract intensity slots.
- **Target model**: concrete intensity vectors with norm control (bounded ℓ¹/ℓ² hybrid) matching the narrative in `Logic/ModalDial`.
- **Scope**:
  1. Introduce a `Structure` for bounded intensity vectors (consider `Finite` index sets, dependent tuples, or `LinearMap` wrappers).
  2. Rework transport lemmas so round-trip contracts operate on the concrete structure.
  3. Extend compliance tests with intensity-preserving scenarios (`collapse`, `expand`, Occam reachability).
- **Dependencies**:
  - mathlib linear algebra norms (`LinearAlgebra.normed`), finite support vectors.
  - Potential helper module for coercions between raw tuples and proof-friendly structures.
- **Risks / Mitigations**:
  - Norm handling may trigger additional proof obligations; plan tactic support (`by have`, `simp [map_eq]`) to keep goals manageable.
  - Ensure no performance regressions in regression tests by gating heavy proofs behind `simp` and `aesop`.

### Graph Bridge — Alexandroff Opens
- **Current carrier**: abstract graph stage with set-valued nodes.
- **Target model**: Alexandroff topology opens on the reachability preorder, enabling direct use of interior/closure operators.
- **Scope**:
  1. Formalise reachability preorder (`Reach ≤`) and the associated Alexandroff topology.
  2. Define the carrier as the lattice of upward-closed sets (opens), ensuring `heyting_algebra` instances align with nucleus fixed points.
  3. Update bridge contracts to transport via `Open` subsets; add compliance cases covering collapse/expand and Occam.
- **Invariants**:
  - Upward-closure proof per open: `∀ x y, reach x y → x ∈ U → y ∈ U`.
  - Compatibility with ladder collapse: `collapseAt` viewed as taking the interior (Alexandroff open kernel) of the corresponding subset.
  - Closure under finite meets/joins required by the Heyting operations (`∩` and interiorised `∪`).
  - Reachability traces in compliance must witness that Occam reductions stay within the open.
- **Dependencies**:
  - mathlib topology (`orderTopology`, `topologicalSpace.alexandrov`), existing `LoF` nucleus lemmas.
  - Reuse of stage collapse/expand automation once the open-set representation is in place.
- **Risks / Mitigations**:
  - Alexandroff constructions introduce universe-level bookkeeping; scope definitions inside a dedicated namespace to control levels.
  - Need additional simp lemmas for upward-closure proofs; schedule a small automation pass.

### Clifford Bridge — Projectors
- **Current carrier**: simplified Clifford algebra elements without explicit projector constraints.
- **Target model**: projector nets (idempotent, self-adjoint elements) forming an orthomodular lattice compatible with breathing operators.
- **Scope**:
  1. Implement projector type (idempotent + adjoint) leveraging existing `Quantum/` scaffolding.
  2. Show the projector lattice satisfies the required orthomodular contracts and lifts the LoF nucleus.
  3. Expand compliance with projector-specific validations (e.g., `collapseAtOmega` aligning with projector composition).
- **Invariants**:
  - Idempotent witness: `P ∘ P = P` and `P† = P` guaranteed at construction time.
  - Orthogonality tracking for breathing: `collapseAt` must preserve mutually orthogonal projectors, and `expandAt` may only introduce bias via admissible joins.
  - Nucleus compatibility: `R (projectorRange P) = projectorRange P` and similar statements tying transport decode/encode to the Heyting core.
  - Round-trip guards: compliance should record that Occam/PSR transports stay within the projector net and respect adjoint symmetry.
- **Dependencies**:
  - mathlib linear algebra (`matrix`, `innerProductSpace`) or existing Clifford utilities.
  - Potential need for lemmas about projector composition; identify missing mathlib support early.
- **Risks / Mitigations**:
  - Proof obligations around self-adjointness may be heavy; plan to cache key lemmas (`@[simp]`) once established.
  - Ensure the automation layer handles orthomodularity without repeated manual rewrites (`simp`, `aesop`).

## Integration Plan
- **Phasing**:
  1. Prepare data-structure modules (one per bridge) with lemmas but without switching transports (`Feature` flags or alternate namespaces).
  2. Update bridge transports once the related compliance suite passes.
  3. Remove legacy carriers after regression tests remain stable across a full CI run.
- **Testing**:
  - Extend `Tests/Compliance.lean` to cover new carrier invariants.
  - Add targeted regression tests per bridge (intensity preservation, open-set invariance, projector orthomodularity).
- **Automation**:
  - Introduce helper tactics (`simp` bundles, `aesop` rules) for common carrier-specific rewrites to keep proof scripts concise.

## Sprint Outlook
- **Next Sprint Ready**:
  1. Finalise data definitions for intensity vectors and Alexandroff opens; scaffold projector typeclass.
  2. Wire transports to new carriers behind feature flags; add corresponding compliance tests.
- **Deferred / Backlog**:
  - Full projector automation and advanced breathing-cycle examples.
  - Performance tuning once all carriers are in place.
  - Documentation refresh in `Docs/Semantics.md` summarising the upgraded carriers once implemented.

## Follow-Up Checklist
- [x] Create infrastructure modules (`Bridges/Tensor/Intensity.lean`, `Bridges/Graph/Alexandroff.lean`, `Bridges/Clifford/Projector.lean`).
- [x] Draft compliance extensions exercising the new carriers.
- [ ] Update docs (`Semantics.md`, `Ontology.md`) after carrier upgrades land.
- [ ] Re-run lint/automation sweeps post-upgrade to ensure proofs remain ergonomic.
