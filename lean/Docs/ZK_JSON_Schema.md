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
