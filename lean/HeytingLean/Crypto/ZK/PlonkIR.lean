import HeytingLean.Crypto.ZK.R1CS

namespace HeytingLean
namespace Crypto
namespace ZK
namespace Plonk

open ZK

/-- Use the existing linear combination type as PLONK gate slots. -/
abbrev LinComb := R1CS.LinComb

/-- A minimal gate: equality in the PLONK slot encoding `A·x = C·x` with `B=1`. -/
structure Gate where
  A : LinComb
  B : LinComb := R1CS.LinComb.ofConst 1
  C : LinComb

/-- A simplified PLONK system with gates and trivial copy permutation (identity).
    We keep a single permutation list representing variable indices seen. -/
structure System where
  gates : List Gate := []
  copyPermutation : List Nat := []

def System.toR1CS (sys : System) : R1CS.System :=
  { constraints := sys.gates.map (fun g => { R1CS.Constraint . A := g.A, B := g.B, C := g.C }) }

/-- Native PLONK satisfaction: in this simplified model it reduces to the R1CS
    satisfaction of the converted system. Copy/permutation checks are tracked
    by the `copyPermutation` field but are not enforced here (placeholder). -/
def System.satisfiedNative (assign : R1CS.Var → ℚ) (sys : System) : Prop :=
  (System.toR1CS sys).satisfied assign ∧ (copyConstraintSystem sys.copyPermutation).satisfied assign

@[simp]
theorem satisfiedNative_iff_r1cs (sys : System) (a : R1CS.Var → ℚ)
    (hId : sys.copyPermutation = List.range sys.copyPermutation.length) :
    sys.satisfiedNative a ↔ (System.toR1CS sys).satisfied a := by
  constructor
  · intro h; exact h.1
  · intro h
    have hCopy : (copyConstraintSystem sys.copyPermutation).satisfied a := by
      -- rewrite to identity and apply lemma
      have : (copyConstraintSystem sys.copyPermutation) = (copyConstraintSystem (List.range sys.copyPermutation.length)) := by
        simpa [hId]
      simpa [this] using copySatisfied_identity a sys.copyPermutation.length
    exact And.intro h hCopy

/-! Copy/permutation checks (identity case)

We model copy/permutation checks by adding equality constraints of the form
`x_i = x_{perm[i]}` as extra constraints. For the canonical identity
permutation `perm = [0,1,2,...,n-1]`, these are tautologies and do not change
the semantics. -/

def eqVarConstraint (i j : R1CS.Var) : R1CS.Constraint :=
  { A := R1CS.LinComb.single i 1
  , B := R1CS.LinComb.ofConst 1
  , C := R1CS.LinComb.single j 1 }

def copyPairs (perm : List Nat) : List (Nat × Nat) := (List.range perm.length).zip perm

def copyConstraintSystem (perm : List Nat) : R1CS.System :=
  { constraints := (copyPairs perm).map (fun ij => eqVarConstraint ij.1 ij.2) }

@[simp]
lemma eqVarConstraint_refl_satisfied (a : R1CS.Var → ℚ) (i : Nat) :
    R1CS.Constraint.satisfied a (eqVarConstraint i i) := by
  classical
  simp [R1CS.Constraint.satisfied, R1CS.LinComb.eval_single]

lemma copySatisfied_identity (a : R1CS.Var → ℚ) (n : Nat) :
    R1CS.System.satisfied a (copyConstraintSystem (List.range n)) := by
  classical
  intro c hc
  -- Constraints are of the form eqVarConstraint i i because zip(range,range)
  have : ∃ i, c = eqVarConstraint i i := by
    -- derive from membership shape
    unfold copyConstraintSystem at hc
    simp [copyConstraintSystem, copyPairs] at hc
    rcases hc with ⟨i, hi, rfl⟩
    -- `hi` witnesses membership in zip(range,range); extract index
    rcases List.mem_zip.mp hi with ⟨hInRange, hInRange', hEq⟩
    -- elements equal since both from range at the same position
    have : i = i := rfl
    exact ⟨i, rfl⟩
  rcases this with ⟨i, rfl⟩
  simpa using eqVarConstraint_refl_satisfied a i

lemma satisfiedNative_iff_r1cs_identity (sys : System) (a : R1CS.Var → ℚ)
    (hId : sys.copyPermutation = List.range sys.copyPermutation.length) :
    sys.satisfiedNative a ↔ (System.toR1CS sys).satisfied a :=
  satisfiedNative_iff_r1cs (sys := sys) (a := a) hId

end Plonk
end ZK
end Crypto
end HeytingLean
