import HeytingLean.LoF.Nucleus

/-!
# Round-trip contracts

Abstract interface describing the encode/decode guarantees required in the roadmap.
-/

namespace HeytingLean
namespace Contracts

open HeytingLean.LoF

variable {α : Type u} [PrimaryAlgebra α]

/-- A round-trip contract packages encoding/decoding data for a given nucleus core. -/
structure RoundTrip (R : Reentry α) (β : Type v) where
  encode : R.Omega → β
  decode : β → R.Omega
  round : ∀ a, decode (encode a) = a

/-- The encoded representation is faithful up to the nucleus applied after decoding. -/
def interiorized (R : Reentry α) {β : Type v} (C : RoundTrip (R := R) β) :
    β → α :=
  fun b => R ((C.decode b : R.Omega) : α)

@[simp] lemma interiorized_id (R : Reentry α) {β} (C : RoundTrip (R := R) β) (a) :
    interiorized (R := R) C (C.encode a) = R (a : α) := by
  unfold interiorized
  simpa using
    congrArg (fun x : R.Omega => R (x : α)) (C.round a)

end Contracts
end HeytingLean
