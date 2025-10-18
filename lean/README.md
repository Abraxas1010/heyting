# HeytingLean

## Developer Workflow

- Run `lake build -- -Dno_sorry -DwarningAsError=true` before pushing changes. The CI workflow mirrors this command so proofs must compile with no warnings and no `sorry` placeholders.
- Stage semantics transport lemmas live in `HeytingLean/Logic/StageSemantics.lean`, and the compliance suite (`HeytingLean/Tests/Compliance.lean`) exercises the Boolean/MV/effect/orthomodular exemplars against them.
