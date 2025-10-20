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

**Build contract:** Every verification (local and CI) MUST run `lake build -- -Dno_sorry -DwarningAsError=true`. No alternate build command counts as a valid test.

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

### 5. Bridge Realisations *(status: ✅ transports, 📌 carrier enrichment rollout)*

Tensor/Graph/Clifford bridges share transports, and scaffolds (`Tensor/Intensity`, `Graph/Alexandroff`, `Clifford/Projector`) reuse core encode/decode/contract APIs with compliance coverage. Stage automation (`stageCollapseAt_eq`, `stageExpandAt_eq`, `stageOccam_encode`) now lands uniformly across the carriers, so collapse/expand/Occam rewrites reduce to core nuclei with a single `simp`. **Next:** fold in the remaining carrier invariants—Alexandroff opens ✅, projector data pending—then feature-flag the enriched rollout and confirm compliance coverage across the new paths.

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

2. **Enhance Bridge Carriers**
   - Specify projector invariants for the Clifford scaffold (Alexandroff opens ✅).
   - Prepare feature-flag rollout once projector data land.
   - Update compliance harnesses so enriched carriers stay covered.

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

1. Adopt `Contracts.Examples.selectSuite` in the runtime selectors (surface flag combos + defaults).
2. Switch transports to enriched carriers once the  paths are vetted (feature-flagged rollout).
3. Expand cross-lens contracts with trace-monoid automation.
4. Finish lint cleanup and add breathing-cycle examples.
5. Publish narrative docs/appendices.
6. Track mathlib gaps (effect/MV/orthomodular/projector support).
7. Extend the enriched bridge packs beyond round-trip (automation, stage lemmas) and propagate into downstream integrations.

## Milestones

- **M1:** ✅ Primary algebra & Heyting core.
- **M2:** ✅ Residuated ladder & breathing infrastructure.
- **M3:** ✅ Epistemic laws (Occam/PSR/Dialectic) with reachability proofs.
- **M4:** ✅ Bridges aligned with shared transport helpers.
- **M5:** 📌 Carrier upgrades documented, invariants & rollout pending.
- **M6:** 📌 Publish docs/examples for dial-a-logic scenarios and breathing cycles.
