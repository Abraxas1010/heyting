Implementation Steps (Quick Start)

1) Preconditions
   - Ensure strict build is the only path: `cd lean && lake build -- -Dno_sorry -DwarningAsError=true`.
   - Keep all new code under `HeytingLean/Crypto` and follow the lens (enc/dec + transport) discipline.

2) Define the unified interface
   - Add `HeytingLean/Crypto/ZK/IR.lean` with:
     - `structure Backend (F)` exposing `Sys`, `Assign`, `compile`, `satisfies`, `public`.
     - `structure Laws (B : Backend F)` with `sound` and `complete` in terms of BoolLens canonical run.

3) Reuse the Bool VM as the single source of truth
   - Compile every backend from the Bool VM program (`Form.compile` → BoolLens program), not from a duplicate evaluator.
   - Rely on existing `compile_correct` to connect back to `Ω_R` via transport.

4) Implement a backend (repeat per ZK scheme)
   - Create an IR/module, e.g. `ZK/PlonkIR.lean`, `ZK/AirIR.lean`, `Lens/Plonk.lean`, etc.
   - Implement `Backend.compile : BoolProg → Sys` for that scheme.
   - Define `satisfies` and `public` for the system.
   - Prove `Laws` (sound/complete) by replaying the Bool canonical run inside the backend’s rules.

5) Keep lenses honest (transport)
   - For any “lens-like” carrier (e.g., Bulletproofs), provide `enc/dec` and a closure discipline (I) or equivalent.
   - Prove RT-1/RT-2 so `dec (evalL (enc ∘ ρ)) = evalΩ φ ρ` holds.

6) Exporters and codecs
   - Add/extend JSON codecs in `HeytingLean/Crypto/ZK/Export.lean` for each backend system and assignment.
   - Keep Boolean-safe encoding for rationals (strings "-1","0","1").

7) CLI wiring
   - Extend `lakefile.lean` with new executables as needed.
   - Follow the existing pattern (`pct_prove`, `pct_verify`, `pct_r1cs`) and select backend via a flag or separate entrypoint.

8) Tests
   - Add Lean-side theorems exercising `compile_satisfied`, `compile_output_eval` for sample forms/envs.
   - Add CLI smoke tests: prove/export, verify, permutation-invariance (reversed constraints), mutation-negative test.

9) CI enforcement
   - Update `.github/workflows/lean_action_ci.yml` to run strict build and the CLI smoke tests.
   - Treat any warning or CLI failure as a hard failure.

10) Documentation
   - Add/update schema notes under `lean/Docs/`.
   - Keep this file as the architectural guide; reference concrete file paths and the Backend/Laws interface.

Great—here’s a tight rewrite of your **ZK multi-compiler** note so it drops straight into your Lean stack and matches the nucleus/lens discipline you already enforce.

# ZK Multi-Compiler (Lean-native), refit for **HeytingLean**

## Additional Capabilities (Once Implemented)

- Multi‑Backend ZK Choice
  - Compile the same BoolLens VM into R1CS, PLONK‑style (custom gates), or AIR/STARK traces.
  - Pick per‑deployment trade‑offs: trusted setup (PLONK) vs no‑setup (STARK), prover speed vs verifier speed, memory vs constraints.

- Hybrid Back‑End Selection
  - Route sub‑problems to the “best” backend (range checks → Bulletproofs, bulk arithmetic → PLONK, long uniform traces → AIR) while preserving a single semantic proof chain.

- Proof‑Carrying Portability
  - Unified Backend/Laws interface lets new systems (Halo2, Groth16, Nova, Spartan) slot in without re‑proving policy semantics.
  - Transport soundness remains anchored in Ω_R → BoolLens → VM; correctness is backend‑agnostic.

- Strong Security Guarantees
  - End‑to‑end soundness/completeness for every backend derived from the same canonical Bool witness.
  - Negative tests (mutated witness) and permutation invariance baked into CI for continual assurance.

- Performance & Cost Modeling
  - Deterministic mapping from IR opcodes to backend constraints/gates enables up‑front proving/verification time and memory estimates.
  - “What‑if” backend comparisons for a given policy (swap R1CS ↔ PLONK ↔ AIR and compare sizes/times).

- Better Circuit Expressivity
  - Custom gates (PLONK) reduce constraint counts for common patterns (AND/OR/IMPLIES, equality checks, linear combos).
  - AIR encodes long, stateful computations succinctly (transition/boundary constraints).

- Range/Commitment Proofs
  - Bulletproofs lens enables efficient range and value‑commitment sub‑proofs; compose with the main policy proof at export time.

- Interop & Ecosystem Fit
  - JSON codecs for each backend (system, assignment, meta) integrate with external SNARK/STARK provers and verifiers.
  - Meta fields (prime/modulus, backend tag, version) make artifacts self‑describing and auditable.

- Deterministic Reproducibility
  - Canonical traces/witnesses across backends yield reproducible builds; CI verifies both proofs and CLI round‑trips.

- Faster Iteration, Lower Integration Risk
  - New backends only implement compile/satisfies/public + Laws; no redefinition of policy semantics.
  - Clear file layout and strict build contract simplify code review and long‑term maintenance.

- Advanced Composability (Forward‑Looking)
  - Foundations for recursive/aggregated proofs (PLONK‑style polynomial backends), enabling rollups and batched attestations.
  - Multi‑policy bundling: prove several policies once, verify in constant time on‑chain/off‑chain.

- Privacy & Governance
  - Fine‑grained backend selection lets you tune privacy/performance to jurisdictional or application needs without changing the verified core logic.

- Operational Confidence
  - CI enforces strict Lean proofs, exporter correctness, order‑invariance, and negative mutation tests, providing “always‑on” assurance.

- Developer Experience
  - Unified CLIs (pct_prove, pct_verify, pct_r1cs) remain “the theorem” for every backend, with schema docs and examples to accelerate adoption.

## 0) What stays invariant

* Policies live in the constructive core (Ω_R); every carrier (“lens”) defines (\lor_R,⇒_R,¬_R) by **closing** with its nucleus (I) and obeys RT-1/RT-2 round-trip contracts. This is the anchor we must not break.  
* All new ZK backends are **lenses** in the same sense: they provide `enc/dec` + a closure (I) (or an equivalent discipline) and prove transport soundness. 

---

## 1) Where the new pieces live (file map)

```
lean/HeytingLean/Crypto/
  Lens/
    Class.lean                     -- existing
    BoolLens.lean                  -- existing (exec + canonical trace)
    Plonk.lean                     -- NEW (polynomial-constraint lens)
    Air.lean                       -- NEW (trace/AIR lens for STARKs)
    Bulletproof.lean               -- NEW (range-commitment lens)
  ZK/
    IR.lean                        -- NEW (unified proof IR interfaces)
    R1CS.lean                      -- existing
    R1CSBool.lean                  -- existing
    PlonkIR.lean                   -- NEW (gates, copy checks)
    AirIR.lean                     -- NEW (trace table + constraints)
    BulletIR.lean                  -- NEW (IPA commitments interface)
    Export.lean                    -- NEW (JSON codecs for each)
  Exec/
    pct_prove.lean                 -- uses proved predicates
    pct_verify.lean                -- ditto
    pct_export.lean                -- selects backend + writes JSON
```

This keeps the “core lens→transport” story intact and lets each ZK scheme plug in behind the same façade. 

---

## 2) One unified interface (lightweight)

```lean
-- HeytingLean/Crypto/ZK/IR.lean
namespace HeytingLean.Crypto.ZK

/-- Minimal structure every backend shares. -/
structure Backend (F : Type) :=
  (Sys     : Type)                      -- constraint system type
  (Assign  : Type)                      -- witness/assignment type
  (compile : BoolProg → Sys)            -- compile from Bool lens VM/program
  (satisfies : Sys → Assign → Prop)
  (public   : Sys → Assign → List F)    -- public outputs for comparison

/-- Law each backend must prove to live in the ecosystem. -/
structure Laws {F} (B : Backend F) :=
  (sound :
    ∀ (φ : Form) (ρ : Env),
      let p  := compile_bool φ;         -- existing, from BoolLens/VM
      let s  := B.compile p;
      let as := canonical_assign p ρ;   -- from BoolLens canonical trace
      B.satisfies s as
      ∧ decode (B.public s as) = BoolLens.eval φ ρ)
  (complete :
    ∀ φ ρ, ∃ as, let s := B.compile (compile_bool φ);
                 B.satisfies s as
                 ∧ decode (B.public s as) = BoolLens.eval φ ρ)

end HeytingLean.Crypto.ZK
```

*Why “Bool first”?* You already run everything through the Bool lens for ZK lowering; the unified `Backend` just formalizes “compile from Bool VM to each system.” 

---

## 3) Backends as lenses (how each fits)

### 3.1 PLONK/Marlin (Polynomial lens)

* **Carrier:** polynomial constraint systems with custom gates + copy/permutation checks.
* **Compile:** select gates (`Add/Mul/Custom`), emit wiring & permutations, expose public I/O. 
* **Obligations:** (i) transport soundness to (Ω_R) via Bool semantics; (ii) permutation/copy constraints consistent with Bool wire order.

```lean
-- HeytingLean/Crypto/Lens/Plonk.lean
structure PLONKGate := (poly : Poly) (wires : Wires)
def compilePlonk : BoolProg → List PLONKGate := ...
theorem transport_sound_plonk :
  ∀ φ ρ, dec (eval_plonk (compilePlonk (compile_bool φ)) (enc ∘ ρ)) = evalΩ φ ρ
```

This is the cleaned-up version of your PLONK sketch (custom gates, better arithmetic than raw R1CS). 

### 3.2 STARKs (AIR lens)

* **Carrier:** execution traces; constraints are polynomials over rows/columns, with boundary & transition conditions.
* **Compile:** lower Bool VM steps to rows of a trace; impose step/consistency constraints. 
* **Benefits:** no trusted setup, scalable; perfect for long uniform computations. 

### 3.3 Bulletproofs (range-proof lens)

* **Carrier:** inner-product arguments with Pedersen commitments (good for ranges, balances, quotas).
* **Compile:** keep logic in Bool; emit separate subcircuits for range checks using IPA proofs; combine at the transcript.  
* **Transport theorem:** `verify_bulletproof … ↔ evalΩ φ ρ` (proved via the Bool semantics + commitment correctness). 

> The “lens composition / hybrid” stays, but we implement it as **backend selection** (range-checks → Bullet, rest → Bool/PLONK) at export time, so the *core* compile correctness remains a single theorem. 

---

## 4) Keep the lens discipline (nucleus + RT contracts)

Every new lens still follows the nucleus recipe: define closure (I) (or the equivalent discipline) and close non-meet ops with it so adjunction/residuation hold; then prove RT-1/RT-2. This keeps semantics identical to the core (Ω_R).  

---

## 5) “Ready-to-prove” skeletons

### 5.1 PLONK backend laws

```lean
-- HeytingLean/Crypto/ZK/PlonkIR.lean
def PlonkBackend : Backend F := { ... }
def PlonkLaws    : Laws PlonkBackend := by
  -- sound: replay Bool canonical run inside gates + copy checks
  -- complete: build witness from canonical Bool assignment
  exact ...
```

### 5.2 AIR backend laws

```lean
-- HeytingLean/Crypto/ZK/AirIR.lean
def AirBackend : Backend F := { ... }
def AirLaws    : Laws AirBackend := by
  -- sound: VM-step ↦ row; transition constraints encode step semantics
  -- complete: fill trace from Bool canonical run
  exact ...
```

### 5.3 Bulletproofs lens laws

```lean
-- HeytingLean/Crypto/Lens/Bulletproof.lean
theorem bulletproof_transport_sound :
  ∀ φ ρ, verify_bulletproof (compile φ) (enc ∘ ρ) ↔ evalΩ φ ρ := by
  ...
```

(Hard focus on range constraints + commitment correctness.) 

---

## 6) CLI glue (one path, many exports)

* `pct_prove` and `pct_verify` keep using the **proved** VM/transport (`compile_correct`) so they *are* the theorem.
* `pct_export --backend=r1cs|plonk|air|bullet` calls the corresponding `Backend.compile` and dumps JSON; an external checker/SNARK verifies `satisfies` and public outputs equal the same (evalΩ). (Your plan already enforces builds under `-Dno_sorry -DwarningAsError=true`.) 

---

## 7) Do’s & don’ts (tight coupling to your stack)

* **Do** keep *one* correctness chain: `Ω_R ≡ BoolLens ≡ VM`—then **backends** compile *from* Bool VM. This preserves your single `compile_correct` and avoids duplicating evaluators. 
* **Do** treat each backend as a **lens**: `enc/dec`, a closure discipline, and a transport proof. 
* **Don’t** re-specify policy semantics per backend; reuse (Ω_R) + transport. 
* **Do** keep hybrid selection at **export** time (range→Bullet; arithmetic→PLONK; bulk→AIR). 

---

## 8) Why this rewrite is “Lean-correct”

It (i) preserves your lens/nucleus contracts, (ii) pulls new ZK systems in as backends with **the same** proof shape (sound+complete from the Bool canonical run), and (iii) keeps the directory and CI discipline you’ve already codified.  

---

### Tie-backs to your original write-up

* “Lens abstraction layer → pluggable ZK backends” → now formalized as `Backend` + `Laws`, with concrete PLONK/AIR/Bulletproof slots. 
* Concrete sketches for Bulletproofs transport and PLONK gates are retained but reshaped to the Lens/Backend pattern.  

If you want, I can generate the minimal `.lean` stubs for `PlonkIR`, `AirIR`, `BulletIR`, and the `Backend/Laws` module exactly as above so they compile cleanly under your `lake` contract and slot into `pct_export`.
