# Proof Notes

This note records auxiliary facts needed when extending the development.

## Fixed-point transportation

* `HeytingLean/LoF/Nucleus.lean` exposes `Reentry.map_himp_apply`, `Reentry.map_sup`, and the
  `apply_coe` lemmas which ensure that core elements remain fixed when they are transported into
  the ambient type.  These are the key ingredients when proving that enriched carriers stay
  stabilised.
* `HeytingLean/LoF/HeytingCore.lean` provides `heyting_adjunction`, letting proofs in the
  compliance suite shuttle between deduction and implication without re-deriving adjunction
  facts.

## Ladder automation

* Stage operations on `Ω_R` (`collapseAtOmega`, `expandAtOmega`) are definitionally equal to
  the base re-entry nucleus.  Lemmas such as
  `HeytingLean.Logic.Stage.DialParam.collapseAt_eq_reentry` and
  `expandAt_eq_reentry` enable straightforward simplification, which we leverage while
  establishing that enriched carriers respect collapse/expand invariants.

## Trace-permutation helpers

* The concurrency section of `HeytingLean/Tests/Compliance.lean` proves that bridge actions
  respect list permutations (`bridgeActWord_of_perm`).  These lemmas justify the trace-monoid
  automation that the roadmap calls for when checking cross-lens transport commutativity.

## Future extensions

When adding new carriers or epistemic layers:

1. Construct encode/decode data using `Contracts.RoundTrip`.
2. Prove the fixed-point invariants once and re-use the helper lemmas above to show that every
   stage operator preserves them.
3. Register the new carrier pack in `Contracts/Examples.selectSuite` so runtime features and
   compliance tests automatically pick up the extension.
