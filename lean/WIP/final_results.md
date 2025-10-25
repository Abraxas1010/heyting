What we built and why it matters

Overview

We now have a fully proof-carrying Boolean-lens stack that compiles formulas into an R1CS system, exports witness artifacts, and verifies them—all backed by Lean theorems. The CLIs in lake exe reflect and reuse the machine-checked results.
A) Logical IR and Core Semantics (Phase A)

Files: lean/HeytingLean/Crypto/Form.lean:1, lean/HeytingLean/Crypto/CoreSemantics.lean:1
Built an inductive IR (Form n) and established eval semantics in ΩR.
Importance: Central representation used by all lenses; keeps core semantics explicit and proven.
B) Lens Abstraction and Transport (Phase B)

Files: lean/HeytingLean/Crypto/Lens/Class.lean:1, lean/HeytingLean/Crypto/Lens/Semantics.lean:1, lean/HeytingLean/Crypto/Lens/Transport.lean:1
Established the general Lens structure and transport theorem dec (evalL (enc ∘ ρ)) = evalΩ φ ρ.
Importance: Guarantees that each concrete lens preserves the logical meaning of formulas.
C) Certified VM and Compiler (Phase C)

Files: lean/HeytingLean/Crypto/Prog.lean:1, lean/HeytingLean/Crypto/VM.lean:1, lean/HeytingLean/Crypto/Compile.lean:1, lean/HeytingLean/Crypto/Correctness.lean:1
Postfix VM, structural compiler, and compile_correct theorem.
Importance: Foundation for witness generation; ensures VM runs match the logical evaluation.
D) Witness Relation and PCT (Phase D)

Files: lean/HeytingLean/Crypto/Witness.lean:1, lean/HeytingLean/Crypto/PCT.lean:1
Canonical witness relation, soundness and completeness in Lean.
Importance: Proof-carrying transactions: witnesses aren’t ad-hoc; they follow from the proven VM pipeline.
E1) Boolean VM Specialization (Phase E1)

File: lean/HeytingLean/Crypto/BoolLens.lean:1
Specialized stack machine and semantics for Bool; canonical value lemmas.
Importance: Boolean is an ideal lens for arithmetization; deterministic and simple semantics.
E2) R1CS Backend with Strong Invariants (Phase E2)

Files:
R1CS primitives: lean/HeytingLean/Crypto/ZK/R1CS.lean:1
Boolean arithmetization: lean/HeytingLean/Crypto/ZK/BoolArith.lean:1
Compiler + invariants: lean/HeytingLean/Crypto/ZK/R1CSBool.lean:1
Support lemmas: lean/HeytingLean/Crypto/ZK/Support.lean:1
Soundness/completeness: lean/HeytingLean/Crypto/ZK/R1CSSoundness.lean:1
Results:
StrongInvariant proof across all opcodes; constraint satisfaction preserved.
Theorems:
compile_satisfied (system is satisfied by canonical assignment)
compile_satisfiable (existential witness)
compile_output_eval (output wire equals BoolLens.eval)
Importance: Formal bridge between Boolean VM semantics and R1CS; proof-carrying circuits instead of trusting generators.
E3) CLI, Exporters, Tests, CI (Phase E3)

Exporters
File: lean/HeytingLean/Crypto/ZK/Export.lean:1
Encode/decode R1CS system and assignment (Boolean-safe: coefficients and values in {-1,0,1})
Assignment length inferred from max wire; meta includes BN254 modulus (customizable)
CLIs
pct_r1cs: lean/HeytingLean/Crypto/ZK/CLI/PCTR1CS.lean:1
pct_prove: lean/HeytingLean/Crypto/ZK/CLI/PCTProve.lean:1
pct_verify: lean/HeytingLean/Crypto/ZK/CLI/PCTVerify.lean:1
Examples and docs
Examples: lean/Examples/PCT/form_and_imp.json:1, lean/Examples/PCT/env_2vars.json:1
Schema: lean/Docs/ZK_JSON_Schema.md:1
README usage: lean/README.md:1
Lean tests
File: lean/HeytingLean/Tests/PCT.lean:1
compile_output_eval and compile_satisfied examples (no sorry)
CI smoke test
File: lean/.github/workflows/lean_action_ci.yml:1
Pipeline: strict build → pct_prove → pct_verify
Importance: This is the operational surface for the proofs; it allows deterministic end-to-end artifact generation verified by the same theory.
Why this is important

Proof-carrying compilation: The R1CS artifacts and witness come with Lean theorems ensuring they reflect Boolean semantics of the input program. No “hand-wavy” trust in a circuit generator.
Deterministic pipeline: From Form → BoolLens → R1CS, all parts are proven to preserve meaning. The exported data is tied to the same compiled structure used in proofs.
Operationalized cryptography: JSON schema + CLI enables integration with downstream SNARK pipelines while keeping the safety net of formal verification in Lean.
How to use (recap)

Build strict: cd lean && lake build -- -Dno_sorry -DwarningAsError=true
Export:
lake exe pct_prove lean/Examples/PCT/form_and_imp.json lean/Examples/PCT/env_2vars.json /tmp/pct_out
Verify:
lake exe pct_verify lean/Examples/PCT/form_and_imp.json lean/Examples/PCT/env_2vars.json /tmp/pct_out/r1cs.json /tmp/pct_out/witness.json
Combined stdout:
lake exe pct_r1cs lean/Examples/PCT/form_and_imp.json lean/Examples/PCT/env_2vars.json
Schema:
lean/Docs/ZK_JSON_Schema.md:1
Open options (polish, if you want)

Add more golden examples for smoke coverage.
Adapt the modulus in meta.json to your preferred SNARK field.
Add a top-level Makefile or lake alias to run all smoke tests locally with one command.
Conclusion

The plan’s core promises are delivered:
Logical IR, lenses, VM compiler with correctness
Boolean R1CS lowering with strong invariants and soundness/completeness
Practical exporters and CLIs backed by proofs
CI enforcing both proofs and CLI round trips
This establishes a robust, end-to-end path from formulas to cryptographically consumable R1CS artifacts, verified in Lean and easy to integrate into the broader Heyting toolchain.