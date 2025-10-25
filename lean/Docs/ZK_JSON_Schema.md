ZK JSON schema (Boolean lens)

- LinComb
  - Object with fields:
    - const: string, one of "-1","0","1"
    - terms: array of [var, coeff] where var is integer (Nat), coeff string in {"-1","0","1"}

- Constraint
  - Object with fields A,B,C each a LinComb

- System
  - Object with field constraints: array of Constraint

- Assignment
  - Array of strings for each variable index, values in {"0","1"}

- Meta
  - Object with fields:
    - outputVar: integer wire id
    - eval: string "true"|"false"
    - field (optional): string, e.g. "prime"
    - modulus (optional): string decimal modulus value

Notes
- Boolean circuits embed over any sufficiently large prime field by interpreting the strings as elements mod p.
- The exported witness length equals max referenced var + 1; unused trailing entries are assumed 0 if present.

Backend-Specific JSON Encodings (Preview)
----------------------------------------

PLONK (plonk.json)
- gates: array of { A: LinComb, B: LinComb, C: LinComb }
- copyPermutation: array of integers (indices)
- CLI: `pct_export plonk … --plonk-gates=name1,name2` (names are placeholders; gates serialize full A/B/C)

AIR (air.json)
- trace: { width: number, length: number }
- r1cs: inlined R1CS `System` (see Boolean schema)
- CLI: `pct_export air … --air-width=n --air-length=m`

Bulletproofs (bullet.json)
- commitments: array of { label: string }
- r1cs: inlined R1CS `System`
- CLI: `pct_export bullet … --bullet-labels=a,b,c`
