import HeytingLean.Crypto.ZK.R1CS
import HeytingLean.Crypto.ZK.Support
import HeytingLean.Crypto.ZK.R1CSBool
import HeytingLean.Crypto.BoolLens

namespace HeytingLean
namespace Crypto
namespace ZK
namespace R1CSSoundness

open R1CSBool
open BoolLens

/-- The compiled R1CS system is satisfied by the canonical assignment. -/
theorem compile_satisfied {n : ℕ} (φ : Form n) (ρ : Env n) :
    System.satisfied (compile φ ρ).assignment (compile φ ρ).system := by
  classical
  -- Unfold compile to the underlying builder result
  have hSatTail :=
    compileTraceToR1CSFromEmpty_satisfied (ρ := ρ) (prog := Form.compile φ)
  -- Satisfaction is invariant under reversing the constraint list
  have hRev :
      System.satisfied (compile φ ρ).assignment
        { constraints :=
            ((compileSteps (ρ := ρ) (Form.compile φ)
                (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1).constraints.reverse } ↔
      System.satisfied (compile φ ρ).assignment
        { constraints :=
            ((compileSteps (ρ := ρ) (Form.compile φ)
                (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1).constraints } := by
    simpa using System.satisfied_reverse
      (assign := (compile φ ρ).assignment)
      (cs :=
        ((compileSteps (ρ := ρ) (Form.compile φ)
              (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1).constraints)
  -- Transport satisfaction to the `compile` system via definitional equalities
  have hSat :
      System.satisfied (compile φ ρ).assignment
        { constraints :=
            ((compileSteps (ρ := ρ) (Form.compile φ)
                (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1).constraints.reverse } := by
    -- rewrite assignment
    have hAssign :
        (compile φ ρ).assignment =
          (compileSteps (ρ := ρ) (Form.compile φ)
            (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1.assign := by
      simp [R1CSBool.compile]
    -- rewrite system
    have hSys :
        (compile φ ρ).system.constraints =
          ((compileSteps (ρ := ρ) (Form.compile φ)
            (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1).constraints.reverse := by
      simp [R1CSBool.compile]
    -- use satisfaction on the unreversed builder.system and flip via `hRev.symm`
    have := System.satisfied_reverse
      (assign :=
        (compileSteps (ρ := ρ) (Form.compile φ)
          (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1.assign)
      (cs :=
        ((compileSteps (ρ := ρ) (Form.compile φ)
          (BoolLens.traceFrom ρ (Form.compile φ) []) [] {}).1).constraints)
    have this' := (this).2
    have hSat' := this' hSatTail
    -- transport assignment equality into the satisfied predicate
    -- equivalently: rewrite both sides using hAssign and hSys
    -- here we conclude by aligning goals with `simp` equalities above
    -- we return a term of the desired shape and use `simpa` with hAssign/hSys
    simpa [hAssign, hSys]
      using hSat'
  -- close by rewriting to `(compile φ ρ).system`
  simpa [R1CSBool.compile]
    using hSat

/-- Completeness: the compiled system is satisfiable (witnessed by the canonical assignment). -/
theorem compile_satisfiable {n : ℕ} (φ : Form n) (ρ : Env n) :
    ∃ a, System.satisfied a (compile φ ρ).system :=
  ⟨(compile φ ρ).assignment, compile_satisfied (φ := φ) (ρ := ρ)⟩

/-- The compiled output variable encodes the boolean evaluation as a rational. -/
theorem compile_output_eval {n : ℕ} (φ : Form n) (ρ : Env n) :
    boolToRat (BoolLens.eval φ ρ) =
      (compile φ ρ).assignment (compile φ ρ).output := by
  classical
  -- abbreviations for the internal builder result
  let prog := Form.compile φ
  let trace := BoolLens.traceFrom ρ prog []
  let result := compileSteps (ρ := ρ) prog trace [] {}
  let builder := result.1
  let stackVars := result.2
  have hMatches := compile_matches (φ := φ) (ρ := ρ)
  -- exec pushes exactly the boolean eval on the stack
  have hExec : BoolLens.exec ρ prog [] = [BoolLens.eval φ ρ] := by
    simp [prog, BoolLens.exec_compile_aux]
  -- specialize Matches to the 1-element stack and extract the head equality
  have hHead : boolToRat (BoolLens.eval φ ρ) = builder.assign (stackVars.headD 0) := by
    -- rewrite matches to the singleton stack
    have h' : Matches builder [BoolLens.eval φ ρ] stackVars := by
      simpa [prog, trace, result, builder, stackVars, hExec]
        using hMatches
    -- analyze the shape of the right list by cases; Forall₂ over a singleton
    cases stackVars using List.rec with
    | nil =>
        -- impossible: Forall₂ cannot relate a nonempty list with []
        cases h'
    | @cons v vs =>
        -- extract the head equality from the Forall₂.cons case
        have hCons : Matches builder (BoolLens.eval φ ρ :: []) (v :: vs) := h'
        have hHead' : boolToRat (BoolLens.eval φ ρ) = builder.assign v :=
          matches_cons_head (builder := builder)
            (b := BoolLens.eval φ ρ) (stack := []) (v := v) (vars := vs) hCons
        -- align with headD
        simpa using hHead'
  -- rewrite back to the `compile` structure
  simpa [R1CSBool.compile, prog, trace, result, builder, stackVars]
    using hHead

end R1CSSoundness
end ZK
end Crypto
end HeytingLean
