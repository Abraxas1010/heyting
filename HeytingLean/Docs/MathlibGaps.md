# Mathlib Gaps

The current development relies only on Lean 4 core and `mathlib4`, but a few planned extensions
still require additional upstream lemmas.

1. **Alexandroff open-set automation.**  We currently construct the Alexandroff carrier using the
   universal open set.  Extending this to arbitrary specialisation orders would benefit from
   stronger support for closure properties of `SetLike` subtypes in order/topology.
2. **Projector arithmetic.**  The Clifford projector scaffold treats the projector axis as data.
   Further work on spectral decompositions will need additional `mathlib4` lemmas about
   self-adjoint idempotents in star-algebras (e.g. convenient reasoning about ranges and kernels).
3. **Tensor norm bookkeeping.**  The intensity carrier records ℓ¹/ℓ² bounds abstractly.  Once
   normed lattice facts for tuples land in `mathlib4`, we can specialise those proofs to replace
   the manual bound witnesses.

These items are tracked in the roadmap and inform the future TODO list.
