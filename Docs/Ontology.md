# HeytingLean Ontology Narrative

## Distinction-as-Re-entry

- `Reentry α` (lean/HeytingLean/LoF/Nucleus.lean) packages a nucleus on a primary algebra together with its primordial/counter fixed points.
- `Reentry α` (lean/HeytingLean/LoF/Nucleus.lean) packages a nucleus on a primary algebra together with its primordial/counter fixed points, now exposing handy rewrites (`map_sup`, `map_bot`) for future bridge automation.
- The Euler boundary `R.eulerBoundary` is defined by infimizing over positive fixed points, ensuring uniqueness up to the minimal process.
- Compliance covers disjointness (`process_inf_counter`) and positivity (`eulerBoundary_pos`).

## Ladder & Modal Breathing

- The residuated ladder (lean/HeytingLean/Logic/ResiduatedLadder.lean) now specializes deduction/abduction/induction to `R.eulerBoundary` and retains collapse behaviour.
- Modal dial parameters (lean/HeytingLean/Logic/ModalDial.lean) expose `euler_boundary_coe`, ensuring the breathing cycle respects the primordial boundary.
- θ-cycle semantics (`thetaCycle_zero_sum`) provide the oscillation witness connecting modal breathing back to the ontological narrative.
- Stage semantics transport (lean/HeytingLean/Logic/StageSemantics.lean) packages MV zero/commutativity and effect orthocomplement lemmas, so every ladder stage inherits the canonical MV/effect laws exercised by compliance.

## Bridges & Dial Examples

- Tensor/Graph/Clifford bridges encode the Euler boundary into concrete carriers (lean/HeytingLean/Bridges/*.lean).
- Compliance tests (`lean/HeytingLean/Tests/Compliance.lean`) assert the encoded outputs and θ-cycle arithmetic, and exercise Boolean collapses via `boolean_limit_verified`.

## Boolean Limit & Documentation

- Boolean equivalence (`booleanEquiv`, `boolean_limit`) form the R= id narrative, with compliance tests showing reconstruction under an identity nucleus.
- This document complements `Docs/STATUS.md` and the plan files (`lean/lean.md`, `lean/ontological_integration_plan.md`, `mathematical_formalism.md`) which have been updated to reference the new lemmas and examples.


## Breathing Example Narrative

To illustrate the breathing cycle concretely:

1. Choose any `Reentry α` and its modal dial parameter `P : DialParam α`.
2. The `thetaCycle_zero_sum` lemma supplies the oscillation cancellation witness.
3. Encoding the Euler boundary through the tensor/graph/clifford bridges yields concrete carriers matching the primordial process (see compliance tests `tensor_encode_euler`, `graph_encode_euler`, `clifford_encode_euler`).
4. With `Collapse` defined as `Dial.collapse`, the lemma `DialParam.euler_boundary_collapse` shows the first breathing step fixes the Euler boundary.
5. Finally, `boolean_limit_verified` demonstrates the Boolean limit in the identity-nucleus regime, capturing the classical endpoint of the ladder.

These steps provide a narrative recipe the team can follow during reviews or demonstrations, referencing compiled lemmas rather than prose alone.
