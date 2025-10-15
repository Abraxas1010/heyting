

# A) Stage-aware helpers beyond `DialParam.base`

We treat each stage as a monoidal/partial-monoidal structure *transported across a bridge* via `shadow : α → Ω` and `lift : Ω → α` satisfying RT contracts:

* **RT-1 (retract):** `shadow (lift u) = u`
* **RT-2 (comparison):** `lift (shadow x) ≤ x` (upgrade to `=` on “exact” bridges)

The helpers on a bridge `B : Bridge α Ω` are then **defined by transport** from the stage operations on the core carrier `Ω` for the selected `DialParam`:

## MV stage (Łukasiewicz / MV-algebra)

* Core ops on `Ω` (when `DialParam ≥ MV`):

  * `mvAdd : Ω → Ω → Ω` (think `min(1, x + y)` in [0,1])
  * `mvNeg : Ω → Ω` (think `1 − x`), `zero`, `one`
* **Helper definitions (transport):**

  * `stageMvAdd x y := lift (mvAdd (shadow x) (shadow y))`
  * (optionally) `stageMvNeg x := lift (mvNeg (shadow x))`
* **Required laws on α (inherited by transport):**

  * **(MV1) assoc/comm/neutral:** `stageMvAdd` associative & commutative; `lift zero` is left/right identity.
  * **(MV2) neg interaction:** `stageMvAdd x (stageMvNeg x) = lift one`.
  * **(MV3) residuation (if provided on Ω):** `shadow (stageMvAdd x y) = mvAdd (shadow x) (shadow y)` (see B-laws).
  * **(MV4) monotone in each argument.**

## Effect stage (partial addition / effect algebra)

* Core ops on `Ω`:

  * `effectAdd? : Ω → Ω → Option Ω` (defined iff “orthosummable”)
  * `effectCompat : Ω → Ω → Prop` (definedness predicate)
  * `orthosupp : Ω → Ω` with `u ⊕ u^⊥ = 1` where defined
* **Helper definitions (transport):**

  * `stageEffectCompatible x y := effectCompat (shadow x) (shadow y)`
  * `stageEffectAdd? x y := (effectAdd? (shadow x) (shadow y)).map lift`
  * `stageOrthosupp x := lift (orthosupp (shadow x))`
* **Required laws on α:**

  * **(E1) compat symmetry & down-closure:** `stageEffectCompatible x y ↔ stageEffectCompatible y x`, and if `x'≤x, y'≤y` and `x#y` then `x'#y'`.
  * **(E2) definedness equivalence:** `isSome (stageEffectAdd? x y) ↔ stageEffectCompatible x y`.
  * **(E3) partial comm/assoc on defined triples** (transport ensures these).
  * **(E4) orthosupp/units:** `isSome (stageEffectAdd? x (stageOrthosupp x))` and result equals `lift one`; `stageEffectAdd? x (lift zero) = some x`.

> Concrete lenses (recommended instantiations)
>
> * **Tensor lens:** pointwise effect: `x ⊕ y` defined iff `∀i, xi + yi ≤ 1`; sum is `(xi+yi)` coordinatewise; `orthosupp x` is pointwise `1−xi`.
> * **Clifford/Hilbert lens (effects `[0,I]`):** `A ⊕ B` defined iff `A+B ≤ I`; `orthosupp A = I−A`.
> * **Graph lens:** nodewise effect with the same guards.

## Orthomodular stage (OML)

* Core ops on `Ω`:

  * `meet, join, compl`, `bot, top`, with orthomodular law
* **Helper definitions (transport):**

  * `stageOrthocomplement x := lift (compl (shadow x))`
  * (optionally) `stageMeet/Join` transported likewise if you expose them on bridges
* **Required laws on α:**

  * **(O1) involution:** `stageOrthocomplement (stageOrthocomplement x) = (lift ∘ shadow) x` (becomes `= x` on exact bridges).
  * **(O2) antitone:** `x ≤ y → stageOrthocomplement y ≤ stageOrthocomplement x`.
  * **(O3) orthomodular identity (on transported meet/join):**
    If `shadow x ≤ shadow y` then

    ```
    shadow y = shadow x ⊔ (compl (shadow x) ⊓ shadow y)
    ```

    (and hence the same equality holds after `lift` if the bridge is exact).

---

# B) How `logicalShadow` must interact with stage ops

We standardize on **“commute up to round-trip”** (a *monoidal nucleus* compatibility). This is strong enough to automate proofs but weak enough to hold across all bridges:

Let `S := logicalShadow : α → Ω` and `L := lift : Ω → α`.

## MV stage

* **Exact commutation on the nose:**

  ```
  (MV-Shadow)    S (stageMvAdd x y) = mvAdd (S x) (S y)
  (MV-Neg)       S (stageMvNeg x)   = mvNeg (S x)
  (MV-Units)     S (lift zero) = zero,  S (lift one) = one
  ```
* If you support residuation on Ω, also require

  ```
  S (stageMvAdd x y) = mvNeg (S x) ⇒ (S x) ⇒ (S y)   -- standard Łukasiewicz law
  ```

  which you can package as rewrite lemmas for automation.

## Effect stage

* **Preserve & reflect definedness (preferred):**

  ```
  (E-Def)    isSome (stageEffectAdd? x y)  ↔  isSome (effectAdd? (S x) (S y))
            ↔ stageEffectCompatible x y    ↔  effectCompat (S x) (S y)
  ```

  (If some bridge can’t reflect, drop “↔” to “→” and mark that bridge as *lax*.)
* **Commutation on value where defined:**

  ```
  (E-Shadow)  stageEffectAdd? x y = some z  →  S z = v
               where effectAdd? (S x) (S y) = some v
  (E-Orth)    S (stageOrthosupp x) = orthosupp (S x)
  ```

## Orthomodular stage

* **Complement commutes exactly; meet/joins commute up to RT:**

  ```
  (O-Compl)   S (stageOrthocomplement x) = compl (S x)
  (O-Meet)    S (stageMeet x y)  = meet (S x) (S y)           -- if you expose meet on bridges
  (O-Join≤)   S (stageJoin x y) ≤ join (S x) (S y)            -- interior is ≤ id
  (O-OMLaw)   If S x ≤ S y then   S y = (S x) ⊔ (compl (S x) ⊓ S y)
  ```

  On exact bridges (RT-2 as equality), strengthen `(O-Join≤)` to equality.

> **Why this shape?**
> `logicalShadow` is your interior map to the core. For interiors, finite meets and complements of fixed points are preserved strictly, while joins are preserved **laxly** unless the bridge is exact. That is exactly what you need to make OML proofs go through while keeping Tensor/Graph/Clifford implementations realistic.

---

# C) Lean skeletons (drop-in) + laws/tests

Below is a minimal, concrete API you can copy into `Bridges/Stage.lean` (or split per stage). It encodes the transport pattern and the shadow-commutation laws as typeclass assumptions you can instantiate per bridge.

```lean
/-
Core stage classes on Ω (the core/fixed-point carrier under the chosen DialParam)
-/
class MvCore (Ω : Type u) where
  mvAdd : Ω → Ω → Ω
  mvNeg : Ω → Ω
  zero  : Ω
  one   : Ω
  -- laws (assoc/comm/neutral/involution/residuation) as fields or in a separate namespace

class EffectCore (Ω : Type u) where
  effectAdd?   : Ω → Ω → Option Ω   -- partial ⊕
  compat       : Ω → Ω → Prop
  orthosupp    : Ω → Ω
  zero one     : Ω
  compat_iff_defined :
    ∀ u v, compat u v ↔ (effectAdd? u v).isSome
  -- assoc/comm on defined, units, etc.

class OmlCore (Ω : Type u) where
  meet join : Ω → Ω → Ω
  compl     : Ω → Ω
  bot top   : Ω
  -- orthomodular law, de Morgan, etc.

/-
Bridge: shadow/lift with round-trip contracts
-/
structure Bridge (α Ω : Type u) [LE α] [LE Ω] :=
  (shadow : α → Ω)
  (lift   : Ω → α)
  (rt₁    : ∀ u, shadow (lift u) = u)
  (rt₂    : ∀ x, lift (shadow x) ≤ x)

namespace Bridge

variable {α Ω : Type u} [LE α] [LE Ω]
variable (B : Bridge α Ω)

-- MV helpers transported to α
def stageMvAdd [MvCore Ω] (x y : α) : α :=
  B.lift (MvCore.mvAdd (B.shadow x) (B.shadow y))

def stageMvNeg  [MvCore Ω] (x : α) : α :=
  B.lift (MvCore.mvNeg (B.shadow x))

-- Effect helpers transported to α
def stageEffectCompatible [EffectCore Ω] (x y : α) : Prop :=
  EffectCore.compat (B.shadow x) (B.shadow y)

def stageEffectAdd? [EffectCore Ω] (x y : α) : Option α :=
  (EffectCore.effectAdd? (B.shadow x) (B.shadow y)).map B.lift

def stageOrthosupp [EffectCore Ω] (x : α) : α :=
  B.lift (EffectCore.orthosupp (B.shadow x))

-- Orthomodular complement on α
def stageOrthocomplement [OmlCore Ω] (x : α) : α :=
  B.lift (OmlCore.compl (B.shadow x))

end Bridge
```

### Shadow-commutation laws (as `@[simp]` lemmas)

```lean
namespace Bridge

variable {α Ω} [LE α] [LE Ω] (B : Bridge α Ω)

@[simp] theorem shadow_stageMvAdd
  [MvCore Ω] (x y : α) :
  B.shadow (B.stageMvAdd x y) =
    MvCore.mvAdd (B.shadow x) (B.shadow y) := by
  unfold stageMvAdd; simpa [B.rt₁]

@[simp] theorem shadow_stageMvNeg
  [MvCore Ω] (x : α) :
  B.shadow (B.stageMvNeg x) = MvCore.mvNeg (B.shadow x) := by
  unfold stageMvNeg; simpa [B.rt₁]

@[simp] theorem defined_iff_compat
  [EffectCore Ω] (x y : α) :
  (B.stageEffectAdd? x y).isSome ↔ B.stageEffectCompatible x y := by
  unfold stageEffectAdd? stageEffectCompatible
  simpa [Option.isSome_map, EffectCore.compat_iff_defined]

@[simp] theorem shadow_stageEffectAdd?
  [EffectCore Ω] (x y : α) :
  (match B.stageEffectAdd? x y with
   | some z => some (B.shadow z)
   | none   => none) =
  EffectCore.effectAdd? (B.shadow x) (B.shadow y) := by
  unfold stageEffectAdd?
  cases h : EffectCore.effectAdd? (B.shadow x) (B.shadow y) <;>
  simp [h, B.rt₁]

@[simp] theorem shadow_stageOrthosupp
  [EffectCore Ω] (x : α) :
  B.shadow (B.stageOrthosupp x) = EffectCore.orthosupp (B.shadow x) := by
  unfold stageOrthosupp; simpa [B.rt₁]

@[simp] theorem shadow_stageOrthocomplement
  [OmlCore Ω] (x : α) :
  B.shadow (B.stageOrthocomplement x) = OmlCore.compl (B.shadow x) := by
  unfold stageOrthocomplement; simpa [B.rt₁]

end Bridge
```

### Laws you can assert once and reuse (tests/automation)

* **MV associativity/commutativity/neutrality on α** are immediate corollaries of the core laws plus `rt₁`. Register as:

  ```lean
  attribute [simp] Bridge.shadow_stageMvAdd Bridge.shadow_stageMvNeg
  ```

  and prove `@[simp]` neutral/assoc/comm for `stageMvAdd` by `shadow`-reflection.

* **Effect definedness equivalence**, commutativity/associativity “on defined triples”, and orthosupp unit all reduce to core lemmas via `shadow_stageEffectAdd?`.

* **Orthomodular identity** on `shadow` (and therefore on `α` after `lift`) is inherited from `Ω`; you can add:

  ```lean
  theorem omlaw_on_shadow [OmlCore Ω] (x y : α) (h : B.shadow x ≤ B.shadow y) :
    B.shadow y = OmlCore.join (B.shadow x) (OmlCore.meet (OmlCore.compl (B.shadow x)) (B.shadow y)) := ...
  ```

---

# D) Practical guidance + defaults per lens

* **Default stance:** implement the helpers by *transport* (as above). This gives you **strict commuting** with `logicalShadow` for MV/effect and complement, and *lax join* for OML unless the bridge is exact.
* **Tensor lens:** make MV/effect ops pointwise; OML usually not exposed here (unless you add a subspace view).
* **Graph lens:** same as Tensor, nodewise; expose OML only if you specifically model subspaces/closures.
* **Clifford/Hilbert lens:** use operator effects `[0,I]` with `A ⊕ B` defined iff `A+B ≤ I`; `orthosupp A = I−A`; OML on closed subspaces/projectors.

---

# E) What this buys you (proof strategy)

1. **Single source of truth:** all stage laws are proved once on `Ω` (core). Bridges get them “for free” via transport + the `shadow_…` lemmas.
2. **Round-trip invariants baked in:** `shadow ∘ stageOp = coreOp ∘ (shadow × shadow)` and `stageOp (lift u) (lift v) = lift (coreOp u v)`.
3. **Automation:** with the `@[simp]` lemmas above, `aesop`/`simp` discharge nearly all stage-compatibility and RT goals.

---

