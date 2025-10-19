# Integrated Ontological & Lean Formalization Plan

This plan tracks the formalisation of the LoF re-entry nucleus, its Heyting core, and the transports across tensors, graphs, and Clifford/geometry lenses.

## Ontological Snapshot

- **Re-entry nucleus** (`LoF/Nucleus.lean`, `LoF/HeytingCore.lean`) — primary algebra, nucleus `R`, fixed points (`process`, `counterProcess`), Euler boundary, Heyting core instance.
- **Breathing ladder** (`Logic/ModalDial.lean`, `Logic/StageSemantics.lean`) — dial hierarchy, `collapseAt`/`expandAt`, ladder operators.
- **Dynamic reasoning layer** (`Logic/PSR.lean`, `Epistemic/Occam.lean`, `Logic/Dialectic.lean`) — residuated reasoning APIs (`breathe`, `birth`), Occam reductions, reachability lemmas.
- **Bridge transports** (`Bridges/Tensor.lean`, `Bridges/Graph.lean`, `Bridges/Clifford.lean`) — round-trip contracts, stage operations, compliance harness (`Tests/Compliance.lean`).

## Codebase Audit *(April 2025)*

- `lake build -- -Dno_sorry -DwarningAsError=true` is green.
- No compiled `sorry`/`admit`/custom axioms.
- Lint warnings restricted to tracked ToDos (unused section variables, `simp` vs `simpa`).

## Objective

Mechanise the LoF nucleus and Heyting core and reuse them across tensor, graph, and geometry lenses.  Maintain consistent round-trip/triad contracts and document invariants, classical limits, and quantum excursions.

## Guiding Principles

- Treat re-entry as a nucleus/interior operator; `Ω_R` is the constructive core.
- Build once in LoF, reuse via interior operators for other lenses.
- Prefer mathlib typeclasses (`heyting_algebra`, `mv_algebra`, etc.).
- Favour compositional proofs; add automation when behaviour stabilises.

## Toolchain & Dependencies

Lean 4 + `mathlib4` with nuclei (`order/nucleus`), Heyting/residuated lattices, MV/effect/orthomodular structures, tensor/graph tools, and Clifford/topological infrastructure.

## Directory Layout

```
lean/
  LoF/            -- nucleus, Heyting core
  Logic/          -- residuated ladder, breathing layer, PSR, dialectic
  Epistemic/      -- Occam and related epistemic laws
  Bridges/        -- tensor/graph/clifford transports
  Contracts/      -- round-trip abstractions & examples
  Docs/           -- plan, notes, proof docs
  Tests/          -- compliance & regression suites
```

## Execution Roadmap

### 0. Environment Setup *(status: ✅ complete)*

Project initialised; CI runs `lake build -- -Dno_sorry -DwarningAsError=true`.

### 1. Primary Algebra Foundation *(status: ✅ stable)*

Re-entry nucleus, helper lemmas (`map_sup`, `map_bot`, `map_himp`).

### 2. Heyting Core *(status: ✅ complete)*

Heyting instance, double-negation lemma, Boolean limit witness.

### 3. Residuated & Transitional Ladder *(status: ✅ proofs, 📌 docs/automation)*

Modal ladder and residuation equivalence complete. **Next:** document ladder semantics and add automation lemmas/tactics.

### 4. Modal Breathing Layer *(status: ✅ operators, 📌 dimensional narrative)*

`collapseAt`, `expandAt`, `breathe/birth` reachability in place. **Next:** connect to dimensional story in docs.

### 5. Bridge Realisations *(status: ✅ transports, 📌 carrier enrichment + transformer hook)*

Tensor/Graph/Clifford bridges share transports, and scaffolds (`Tensor/Intensity`, `Graph/Alexandroff`, `Clifford/Projector`) reuse core encode/decode/contract APIs with compliance coverage. **Next:** add openness/projector invariants, plan feature-flagged carrier rollout, and align the forthcoming transformer back-end spec (`TBD/transformer_architecture_integration.mc`) with the tensor logic bridge once the generators are ready.

### 6. Cross-Lens Contracts *(status: ✅ base cases, 📌 automation)*

RT-1/RT-2/TRI-1/TRI-2 proven for identity + transports. **Next:** expand automation (`@[simp]`, trace-monoid tooling) to reduce manual rewrites.

### 7. Limits, Dialling & Examples *(status: ✅ coverage, 📌 lint polish/examples)*

Compliance exercises Boolean/MV/effect/orthomodular examples. **Next:** resolve lint hints (`simp` vs `simpa`, unused section vars) and add richer breathing-cycle examples.

### 8. Validation & Automation *(status: ⚠️ automation/lint)*

Builds clean; continue automation + lint sweep across bridges/compliance.

### 9. Documentation & Developer Support *(status: ⚠️ outstanding)*

High-level docs pending: ladder/dimensional story, carrier rationale (`Docs/ProofNotes.md`).

### 10. Epistemic Laws *(status: ✅ logic, 📌 narrative)*

Occam/PSR/Dialectic implemented with tests. **Next:** document Euler-boundary narrative.

## Near-Term Sprint *(Q2 2025)*

1. **Document & Automate Ladder Dynamics**
   - Write dimensional semantics doc.
   - Add automation lemmas/tactics for ladder proofs.

2. **Enhance Bridge Carriers & Transformer Hook**
   - Specify openness/projector invariants for the Alexandroff/projector scaffolds.
   - Prepare feature-flag rollout once invariants/closure data are ready.
   - Sync the transformer architecture integration spec with the tensor bridge (define execution hooks/API expectations).

3. **Lint & Automation Sweep**
   - Resolve `simp` vs `simpa`, unused section variables.
   - Introduce automation (`@[simp]`, `aesop`) for repetitive rewrites.

4. **Documentation**
   - Integrate `Docs/ProofNotes.md` into developer docs.
   - Update `Docs/Semantics.md`, `Docs/Ontology.md` with carrier rationale/dimensional story.

## Cleanup Backlog

- Lint polish across bridges/compliance.
- Extend shared helpers for additional stage dynamics as needed.

## Outstanding TODOs

1. Document ladder collapse/expand + reachability in docs.
2. Introduce automation support for ladder/bridge proofs.
3. Switch transports to enriched carriers once invariants land (feature-flagged rollout).
4. Expand cross-lens contracts with trace-monoid automation.
5. Finish lint cleanup and add breathing-cycle examples.
6. Publish narrative docs/appendices.
7. Track mathlib gaps (effect/MV/orthomodular/projector support).

## Milestones

- **M1:** ✅ Primary algebra & Heyting core.
- **M2:** ✅ Residuated ladder & breathing infrastructure.
- **M3:** ✅ Epistemic laws (Occam/PSR/Dialectic) with reachability proofs.
- **M4:** ✅ Bridges aligned with shared transport helpers.
- **M5:** 📌 Carrier upgrades documented, invariants & rollout pending.
- **M6:** 📌 Publish docs/examples for dial-a-logic scenarios and breathing cycles.
