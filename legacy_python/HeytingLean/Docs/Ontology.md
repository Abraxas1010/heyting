# Ontological Snapshot

This repository formalises the LoF re-entry nucleus and the structures that arise from
its fixed-point core.  The Lean modules underneath the snapshot supplied in
`lean_formalization_plan.md` now have fully audited implementations.

## Re-entry nucleus

* `HeytingLean/LoF/Nucleus.lean` packages a `Reentry` structure consisting of the nucleus,
  the primordial and counter fixed points, Euler boundary, and the dominance lemmas that
  describe their order-theoretic interaction.
* `HeytingLean/Ontology/Primordial.lean` links the abstract nucleus to the ontological story,
  encoding the oscillation narrative (`thetaCycle`) and confirming the recursive zero law.

## Heyting core

* `HeytingLean/LoF/HeytingCore.lean` establishes the Heyting algebra on the fixed point
  sublocale `Ω_R`, records the adjunction (`heyting_adjunction`), and supplies basic
  equivalences such as the double-negation lemma and the Boolean limit equivalence.

## Dialectic layer

* `HeytingLean/Logic/Triad.lean` and `HeytingLean/Logic/ResiduatedLadder.lean` express
  deduction, abduction, and induction as the three faces of residuation and show that they
  coincide.  These lemmas feed directly into the compliance suite.
* `HeytingLean/Logic/Dialectic.lean` wraps the synthesis operator `synth` and proves it
  agrees with the Euler boundary when both endpoints are the primordial fixed point.

## Bridge carriers

* `HeytingLean/Bridges/Tensor.lean`, `HeytingLean/Bridges/Graph.lean`, and
  `HeytingLean/Bridges/Clifford.lean` implement the concrete tensor, graph, and Clifford
  carriers, each with encode/decode contracts and stage operations that commute with the
  round-trip data.
* Enriched variants live in `Bridges/Tensor/Intensity.lean`, `Bridges/Graph/Alexandroff.lean`,
  and `Bridges/Clifford/Projector.lean`.  These now record explicit invariants (fixed-point
  witnesses and projector closure) rather than the earlier `True` placeholders.

## Epistemic and runtime glue

* `HeytingLean/Epistemic/Occam.lean` introduces the iterated breathing operator, birth date,
  and Occam reduction, together with the minimality lemmas used in bridge transports.
* `HeytingLean/Runtime/BridgeSuite.lean` assembles tensor/graph/Clifford bridge packs using
  the feature flags described in the roadmap, while `HeytingLean/Tests/Compliance.lean`
  exercises the round-trip, ladder, and trace-commutation contracts.

The documentation above matches the roadmap commitments and provides the narrative that the
plan expected in `Docs/Ontology.md`.
