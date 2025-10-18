# Integrated Ontological & Lean Formalization Plan

## Ontological Snapshot

- **Re-entry nucleus** (`HeytingLean/LoF/Nucleus.lean`) delivers the primary algebra, complementary fixed points (`process`, `counterProcess`), Euler boundary, and the Heyting core instance (`LoF/HeytingCore.lean`).
- **Breathing ladder** (`Logic/ModalDial.lean`) supplies the dial hierarchy and the `collapseAt`/`expandAt` family; `Logic/StageSemantics.lean` exposes stage-level operators (`collapseAtOmega`, `expandAtOmega`, MV/effect/orthomodular structures).
- **Dynamic reasoning layer** (`Logic/PSR.lean`, `Epistemic/Occam.lean`, `Logic/Dialectic.lean`) now works through the `breathe`/`birth` reachability API, providing stability lemmas (`breathe_le_of_sufficient`, `sufficient_reachable`) and minimal-birthday Occam reductions.
- **Bridge transports** (Tensor/Graph/Clifford) share round-trip helpers via `Contracts/RoundTrip.stageOccam`, ensuring uniform Occam/PSR transport; compliance (`Tests/Compliance.lean`) exercises RT/TRI, stage collapse/expand, reachability, and Occam transport for every bridge.

## Codebase Audit *(April 2025)*

- `lake build -- -Dno_sorry -DwarningAsError=true` is green; no compiled `sorry`/`admit`/custom axioms.
- Stage helpers (`stageCollapseAt`, `stageExpandAt`, `stageOccam`) are aligned with bridge contracts; compliance covers Boolean/MV/effect/orthomodular exemplars plus PSR reachability lemmas.
- Shared helpers (`Contracts.stageOccam`, `Logic/PSR.reachable_iff_exists_breathe`) reduce duplication; lint warnings remain only where marked TODO.

## Objective

- Mechanise the re-entry nucleus, Heyting core, breathing ladder, and cross-lens translations in Lean so that ontological claims map to machine-checked theorems.
- Maintain a coherent codebase spanning logical (LoF), algebraic (residuated/effect), geometric (Clifford/orthomodular), and computational (tensor/graph) lenses while upholding the RT/TRI/double-negation contracts.

## Guiding Principles

- Treat re-entry as a nucleus/interior operator; fixed points (`Ω_R`) form the constructive core.
- Build once in the LoF layer; reuse via functors/interior operators for tensor/graph/geometry lenses.
- Prefer mathlib typeclasses (`heyting_algebra`, `mv_algebra`, `effect_algebra`, `orthomodular_lattice`, …); only add new abstractions if unavoidable.
- Keep proofs compositional, favour automation (`@[simp]`, tactics) once behaviour is stable.

## Toolchain & Dependencies

- Lean 4 + `mathlib4`.
- Key dependencies: nuclei (`order/nucleus`), Heyting/residuated lattices, MV/effect/orthomodular structures, inner-product/projector machinery (`linear_algebra`, `analysis`), Alexandroff/topological tools (`order/topology`).
- Flag gaps early (e.g., missing `EffectAlgebra` support) and scope minimal replacements if mathlib lacks them.

## Directory Layout

```
lean/
  lakefile.toml / lake-manifest.json
  LoF/              -- primary algebra, nucleus, Heyting core
  Logic/            -- residuated ladder, modal dial, stage semantics, PSR, dialectic
  Epistemic/        -- Occam and related epistemic laws
  Bridges/          -- tensor/graph/clifford transports
  Contracts/        -- round-trip abstractions & examples
  Quantum/          -- projector/orthomodular scaffolding (WIP)
  Docs/             -- narrative + generated documentation
  Tests/            -- compliance & regression suites
```

## Execution Roadmap

### 0. Environment Setup *(status: ✅ complete)*
Lean project initialised; CI runs `lake build -- -Dno_sorry -DwarningAsError=true`.

### 1. Primary Algebra Foundation *(status: ✅ stable)*
Re-entry nucleus + fixed-point sublocale (`Ω_R`) in place; helper lemmas (`map_sup`, `map_bot`, `map_himp`) available for downstream automation.

### 2. Heyting Core *(status: ✅ complete)*
Heyting instance, residuation, Boolean-limit witness implemented; double-negation/collapsing lemmas exposed.

### 3. Residuated & Transitional Ladder *(status: ✅ core proofs, 📌 docs pending)*
Deduction/abduction/induction equivalence (`Logic/ResiduatedLadder.lean`) and modal ladder definitions (`Logic/ModalDial.lean`) complete; ladder collapse/expand laws exercised via reachability. **Next:** document the ladder semantics and surface automation/tactics.

### 4. Modal Breathing Layer *(status: ✅ operators, 📌 dimensional narrative pending)*
`collapseAt`/`expandAt` monotonicity, `breathe`/`birth` reachability, and stage-level stability lemmas finished. **Next:** tie these laws explicitly to dimensional stories (0D→orthomodular) in Docs.

### 5. Bridge Realisations *(status: ✅ unified helpers, 📌 carrier upgrades pending)*
Tensor/Graph/Clifford bridges share transport (implication, Occam). **Next:** upgrade carriers to intended roadmap targets (tensor intensity vectors, graph Alexandroff opens, Clifford projectors) and extend helper catalogue for additional dynamics as needed.

### 6. Cross-Lens Contracts *(status: ✅ base cases, 📌 automation pending)*
RT-1/RT-2/TRI-1/TRI-2 verified for identity + bridges; compliance covers every dial level. **Next:** generalise via trace-monoid tooling and introduce automation (`@[simp]`, custom tactics) to avoid manual rewrites.

### 7. Limits, Dialling & Examples *(status: ✅ regression coverage, 📌 lint polish pending)*
Compliance exercises Boolean/MV/effect/orthomodular examples, breathing stability, and reachability lemmas. **Next:** finish lint cleanup (`simp` vs `simpa`, unused arguments) and add richer breathing-cycle examples.

### 8. Validation & Automation *(status: ⚠️ in progress)*
`lake build` clean; remaining work is automation/lint polish in bridges/compliance.

### 9. Documentation & Developer Support *(status: ⚠️ outstanding)*
Docstrings present; high-level docs (transport API decisions, dimensional semantics) still to be written.

### 10. Epistemic Laws *(status: ✅ implemented, 📌 narrative pending)*
Occam/PSR/Dialectic implemented with regression tests. **Next:** write the Euler-boundary narrative and integrate into Docs.

## Near-Term Subplan *(Sprint Focus)*

1. **Document & Automate Ladder Dynamics**  
   - Capture `collapseAt`/`expandAt` + reachability behaviour in Docs.  
   - Add automation/tactics for ladder collapse/expand proofs.

2. **Enhance Bridge Helpers & Carriers**  
   - Extend shared helpers (beyond `stageOccam`) for other dynamics.  
   - Begin roadmap carrier upgrades (tensor intensity, graph Alexandroff, Clifford projectors).

3. **Lint & Automation Sweep**  
   - Finish trimming lingering lint warnings across bridges/compliance/tests.  
   - Introduce structured automation (`@[simp]`, `aesop`) where repetitive rewrites remain.

## Cleanup Backlog

- Lint polish in bridges/compliance (`simp` vs `simpa`, unused arguments).  
- Shared helper catalogue for additional stage dynamics if required (e.g., `stageCollapse`, `stageExpand` wrappers).

## Outstanding TODO Summary

1. Document ladder collapse/expand + reachability and tie to dimensional semantics.  
2. Automate ladder/bridge proofs (simp+tactic support).  
3. Upgrade bridge carriers per roadmap targets.  
4. Extend cross-lens contracts with trace-monoid automation.  
5. Finish lint cleanup and add richer examples.  
6. Publish documentation (`Docs/README.md`, narrative appendices).  
7. Scope mathlib gaps (effect/MV/orthomodular/projector support).

## Milestones

- **M1:** ✅ Primary algebra & Heyting core.  
- **M2:** ✅ Residuated ladder + breathing infrastructure.  
- **M3:** ✅ Epistemic laws (Occam/PSR/Dialectic) with reachability proofs.  
- **M4:** ✅ Bridges aligned with shared transport helpers.  
- **M5:** 📌 Document ladder/bridge automation & upgrade carriers.  
- **M6:** 📌 Publish docs/examples showcasing dial-a-logic scenarios and breathing cycles.
