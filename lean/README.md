# HeytingLean

## Developer Workflow

- Run `lake build -- -Dno_sorry -DwarningAsError=true` before pushing changes. The CI workflow mirrors this command so proofs must compile with no warnings and no `sorry` placeholders.
- Stage semantics transport lemmas live in `HeytingLean/Logic/StageSemantics.lean`, and the compliance suite (`HeytingLean/Tests/Compliance.lean`) exercises the Boolean/MV/effect/orthomodular exemplars against them.

## PCT CLI (Boolean R1CS)

- Export R1CS + witness

```
lake exe pct_prove lean/Examples/PCT/form_and_imp.json \
                   lean/Examples/PCT/env_2vars.json \
                   /tmp/pct_out
```

- Verify artefacts end-to-end

```
lake exe pct_verify lean/Examples/PCT/form_and_imp.json \
                    lean/Examples/PCT/env_2vars.json \
                    /tmp/pct_out/r1cs.json \
                    /tmp/pct_out/witness.json
```

- Emit combined JSON to stdout

```
lake exe pct_r1cs lean/Examples/PCT/form_and_imp.json \
                  lean/Examples/PCT/env_2vars.json
```

Schema is documented in `Docs/ZK_JSON_Schema.md`.
