PCT CLI examples

- Prove and export R1CS + witness

```
lake exe pct_prove lean/Examples/PCT/form_and_imp.json \
                   lean/Examples/PCT/env_2vars.json \
                   /tmp/pct_out
```

- Verify exported artefacts

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
