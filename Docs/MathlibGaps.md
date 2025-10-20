# Mathlib Gap Tracker

This note lists upstream features that would further simplify the bridge automation.

- Effect algebra elaboration for projector ranges (orthomodular support) – currently handled locally via `bridgeActWord_of_perm` permutations.
- MV-algebra stage helpers for additional ladder automation (collapse/expand monotonicity lemmas already registered locally).
- Potential additions around trace monoids (`List.Perm` utilities) to reduce bespoke permutations in `Tests/Compliance.lean`.

We will keep this list in sync with mathlib issues as they arise.
