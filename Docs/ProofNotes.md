# Proof Notes & Contracts

This document records the mathematical guarantees that underpin the bridge scaffolds.  It is an
adaptation of the “Proof Notes & Contracts” brief and is organised by core → lens transport.

## 0. Preliminaries

- **Primary algebra (LoF)**: Spencer–Brown expressions modulo Calling/Crossing reduction.  Meet,
  join, and Boolean negation are written `∧`, `∨`, and `¬`; the lattice order is `≤`.
- **Nucleus** `R : PA → PA`: inflationary, idempotent, meet-preserving.  Intuition: re-entry
  stabilisation.

## 1. Heyting Core

Fixed-point subalgebra `Ω_R := { a ∈ PA | R a = a }`.  Operations:

```
a ∧₍R₎ b := a ∧ b
a ∨₍R₎ b := R (a ∨ b)
a ⇒₍R₎ b := R (¬ a ∨ b)
¬₍R₎ a   := R (¬ a)
```

### Key facts

1. `(Ω_R, ∧₍R₎, ∨₍R₎, ⇒₍R₎, ¬₍R₎, ⊤, ⊥)` is a Heyting algebra.
2. **Residuation**: `a ∧₍R₎ b ≤ c ↔ b ≤ a ⇒₍R₎ c`.
3. **Double negation**: `a ≤ ¬₍R₎ ¬₍R₎ a`, with equality iff `R = id` on the generated subalgebra.
4. Classical limit: `R = id` ⇒ the structure collapses to Boolean logic.

This is the single core used for the reasoning triad (deduction = meet, abduction/induction =
implication).

## 2. Transport to the Four Lenses

Each lens carries its own nucleus (interior/closure) satisfying the same three axioms so that
Heyting laws transport verbatim.

| Lens              | Interior      | Meet                  | Join                              | Implication                              | Negation                  |
| ----------------- | ------------- | --------------------- | --------------------------------- | ---------------------------------------- | ------------------------- |
| Logic core        | `R`           | `∧`                    | `R (∨)`                           | `R (¬ a ∨ b)`                            | `R (¬ a)`                 |
| Tensors           | `Int`         | pointwise `min`       | `Int (max)`                       | `Int (max (1 - χ a, χ b))`               | `Int (1 - χ a)`           |
| Graphs / opens    | Alex. `Int`   | `∩`                    | `Int (U ∪ V)`                     | `Int (Uᶜ ∪ V)`                           | `Int (Uᶜ)`                |
| Projectors/Clifford | `J`        | range intersection    | `J (span)`                        | `J (¬ A ∪ B)`                            | `J (¬ A)`                 |

Boolean behaviour is recovered when the interior is the identity; quantum/orthomodular excursions are
handled by projecting back with `J`.

## 3. Round-trip Contracts

Let `enc : Ω_R → lens` and `dec` be thresholding followed by the interior.

1. **RT-1**: `dec ∘ enc = id` on `Ω_R`.
2. **RT-2**: each lens operation equals the logical operation after applying the interior:
   `enc (a ⋄ b) = Int (enc a ⋄ enc b)`.

Compliance lemmas exercise these identities for the tensor intensity, graph Alexandroff, and Clifford
projector scaffolds.

## 4. Triad Contracts

For the core and every lens (after interiorisation):

```
Deduction:  C⋆ = A ∧₍R₎ B
Abduction:  B⋆ = A ⇒₍R₎ C
Induction:  A⋆ = B ⇒₍R₎ C

A ∧₍R₎ B ≤ C ↔ B ≤ A ⇒₍R₎ C ↔ A ≤ B ⇒₍R₎ C.
```

Integrity constraints and bias are enforced by meets in `Ω_R` (or their images under the lens
interior).

## 5. Counterexamples & Guardrails

- **Graphs/tensors (no interior)**: `U ∪ V` need not be open; residuation fails.  Fix: close unions.
- **Projectors (non-orthogonal span)**: distributivity can break; fix by interiorising with `J`.
- **Double negation strictness**: choose `R` that deletes a fringe; `a < ¬₍R₎ ¬₍R₎ a`, demonstrating
  constructive behaviour.

## 6. Dimension/Phase Family

Provide nuclei `{R_d}` (or `{J_d}`) with `R_1` strongly nontrivial and `R_d → id` as dimension grows.
This yields `Ω_{R_1} ⊆ Ω_{R_2} ⊆ …` and a controlled shift from Heyting to Boolean logic.

## 7. Deliverables & Status

- [x] Scaffolds linked to core transports (tensor/graph/projector).
- [x] Compliance lemmas covering new carriers.
- [x] Interior closure data (Alexandroff membership proofs, projector invariants) — Alexandroff opens ✅ and projector invariants now exercised via the feature-flag compliance suite.
- [ ] Lint polish (`simp` vs `simpa`, unused section variables).

This document serves as the mathematical reference for the bridge upgrade roadmap.  Implementation of
additional invariants can extend the corresponding sections.

## 8. Automation Targets

- **Ladder**: register `Dial.collapse_monotone`, `Dial.expand_monotone`, and the projection lemmas (`box_le_collapse`, `box_le_expand`) for `aesop`/`gcongr`; add `[simp]` or `[mono]` variants so `collapseAt`/`expandAt` chains avoid bespoke transitivity proofs.
- **Tensor**: factor `Point.ext`/`Point.eta` rewrites into `[simp]` bundles and a helper for coordinatewise nuclei application, enabling `simp` to discharge the round-trip proof in `Model.contract` and analogous lemmas.
- **Graph**: expose adjacency monotonicity as simp-available lemmas and add automation for `stage*` lifts so Occam/PSR transports reduce to core proofs without manual `Subtype.ext`.
- **Compliance**: curate a shared `simp` set enumerating the stage operations (`stageMvAdd`, `stageEffectAdd?`, `stageOrthocomplement`, `stageHimp`) across bridges, unlocking `simp`-first regressions instead of ad hoc rewrites.
