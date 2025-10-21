# Modal and Bridge Semantics

This document summarises how the Lean implementation realises the modal ladder and the
bridge semantics promised in the execution plan.

## Ladder semantics

* `HeytingLean/Logic/ModalDial.lean` defines modal dials (`Dial`) equipped with box/diamond
  nuclei and the arithmetic ladder `DialParam.ladder`.  Collapse and expansion are shown to be
  monotone and to agree with the ambient re-entry nucleus at every stage.
* `HeytingLean/Logic/StageSemantics.lean` promotes ladder operations to the Heyting core
  (`collapseAtOmega` / `expandAtOmega`) and supplies the transport helpers consumed by bridges.
* `HeytingLean/Logic/ResiduatedLadder.lean` packages deduction/abduction/induction so that the
  compliance suite can reason uniformly about residuated proofs across carriers.

## Bridge transports

* Each core bridge (`Tensor`, `Graph`, `Clifford`) implements `stageMvAdd`, `stageEffectAdd?`,
  `stageHimp`, and the ladder operators in terms of the shared stage semantics module.
* The enriched carriers layer additional invariants:
  * Tensor intensity carriers track that every coordinate lies in the fixed-point core and
    preserve that guarantee across stage operators.
  * Graph Alexandroff carriers ensure the chosen open set is closed under collapse, expand, and
    Occam.
  * Clifford projector carriers reuse the projector closure invariants so that stage operators
    remain inside the projected subspace.

## Contracts and runtime

* `HeytingLean/Contracts/RoundTrip.lean` specifies the encode/decode interface and supplies the
  `stageOccam` helper.  `Contracts/Examples.lean` instantiates the interface for the identity,
  tensor, graph, and Clifford bridges, and now assembles enriched bridge suites using the new
  invariants.
* `HeytingLean/Runtime/BridgeSuite.lean` selects carrier packs based on bridge flags, leaning on
  the enriched defaults described above.

Together these modules deliver the modal dynamics and bridge semantics that the roadmap lists
under “Document & Automate Ladder Dynamics” and “Enhance Bridge Carriers”.
