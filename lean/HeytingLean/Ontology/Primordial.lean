import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic
import HeytingLean.LoF.Nucleus

/-!
# Primordial ontology

This module encodes the metamathematical narrative:
- `ReentryKernel` captures the self-referential act of distinction as an idempotent map.
- `primordialOscillation` realises the Euler monad `e^{iθ}`.
- Complementary lemmas show antiphasic cancellation, aligning with the idea of recursive zeros.
- `dialecticPair` packages the oscillation and its counter-process.
- `zero_sum` states the recursive-zero condition.
- `supported` links these abstract pieces back to an actual `Reentry` nucleus.
- `supported_oscillation` instantiates the oscillation proof for any nucleus.

These formal objects mirror the refined ontological sequence: Distinction-as-Re-entry → Dialectic → Euler boundary → Recursive zero.
-/

namespace HeytingLean
namespace Ontology

open Complex
open scoped Real

noncomputable section

/-- An abstract self-referential distinction. -/
structure ReentryKernel (α : Type u) where
  distinction : α → α
  self_reference : ∀ a, distinction (distinction a) = distinction a

/-- Every nucleus yields a self-referential kernel. -/
def Reentry.kernel {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) : ReentryKernel α :=
  { distinction := R
    self_reference := by
      intro a
      exact R.idempotent a }

/-- The fundamental oscillation `e^{iθ}`. -/
def primordialOscillation (θ : ℝ) : ℂ :=
  Complex.exp (Complex.I * θ)

@[simp] lemma oscillation_antiphase (θ : ℝ) :
    primordialOscillation (θ + _root_.Real.pi) = - primordialOscillation θ := by
  have hmul :
      Complex.I * (θ + _root_.Real.pi) = Complex.I * θ + Complex.I * _root_.Real.pi := by
    simp [mul_add]
  have hcalc :
      primordialOscillation (θ + _root_.Real.pi)
          = Complex.exp (Complex.I * θ)
              * Complex.exp (Complex.I * _root_.Real.pi) := by
    simp [primordialOscillation, hmul, Complex.exp_add]
  have hpi : Complex.exp (Complex.I * _root_.Real.pi) = (-1 : ℂ) := by
    simpa [mul_comm] using Complex.exp_pi_mul_I
  simpa [primordialOscillation, hpi, mul_comm, mul_left_comm, mul_assoc,
    mul_neg_one] using hcalc

lemma oscillation_pair_cancel (θ : ℝ) :
    primordialOscillation θ + primordialOscillation (θ + _root_.Real.pi) = 0 := by
  simp [oscillation_antiphase]

/-- Process and counter-process bundled as a single datum. -/
def dialecticPair (θ : ℝ) : ℂ × ℂ :=
  ⟨primordialOscillation θ, primordialOscillation (θ + _root_.Real.pi)⟩

lemma zero_sum (θ : ℝ) :
    (dialecticPair θ).1 + (dialecticPair θ).2 = 0 :=
  oscillation_pair_cancel θ

/-- A `Reentry` nucleus supports the Euler oscillation. -/
structure Supported (α : Type u) [LoF.PrimaryAlgebra α] where
  kernel : ReentryKernel α
  enhances : ℝ → ℂ
  counter : ℝ → ℂ
  cancel : ∀ θ, enhances θ + counter θ = 0

/-- Every nucleus gives a supported oscillation via `e^{iθ}`. -/
def supported_oscillation {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) : Supported α :=
  { kernel := Reentry.kernel (R := R)
    enhances := primordialOscillation
    counter := fun θ => primordialOscillation (θ + _root_.Real.pi)
    cancel := oscillation_pair_cancel }

noncomputable def thetaCycle : ℝ → ℂ × ℂ :=
  fun θ => dialecticPair θ

lemma thetaCycle_zero_sum (θ : ℝ) :
    (thetaCycle θ).1 + (thetaCycle θ).2 = 0 :=
  zero_sum θ

lemma thetaCycle_supported_cancel {α : Type u} [LoF.PrimaryAlgebra α]
    (R : LoF.Reentry α) (θ : ℝ) :
    (supported_oscillation (R := R)).enhances θ
        + (supported_oscillation (R := R)).counter θ = 0 :=
  (supported_oscillation (R := R)).cancel θ

end

end Ontology
end HeytingLean
