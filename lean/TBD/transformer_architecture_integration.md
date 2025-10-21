# Exact Algebraic Learning with Heyting Attention: a Re-entry/Nucleus Foundation for Transformer-like Models

## Abstract

We replace approximate, gradient-based training of neural attention with a **fully algebraic alternative** grounded in your re-entry-as-nucleus framework. Let (L) be a complete distributive lattice (frame) for the **primary algebra** of expressions; let a **nucleus** (R:L!\to!L) be inflationary, idempotent, meet-preserving. The fixed points (\Omega_R={x\mid R x=x}) form a Heyting algebra with implication (a\Rightarrow_R b:=R(\neg a\vee b)). We design a **Transformer-class architecture** whose core operators are **Heyting-valid** on (\Omega_R):

1. **Heyting attention**: scores are computed by a valuation (v:\Omega_R\to[0,1]) applied to residuation (k\Rightarrow_R q); heads aggregate values using nuclei-respecting joins.
2. **Algebraic backprop**: each forward operator (F) is paired with its **right adjoint** (F^\ast) (residuation). Training is a **monotone fixed-point iteration** on a complete lattice of parameters/hypotheses, with existence (Tarski) and constructive convergence (Kleene) under standard continuity/finite-height assumptions—**no real-valued gradients are required**.
3. **Bridges**: Tensor/Graph/Projector lenses implement the same laws via transport along your existing exact/lax bridges with round-trip contracts.

We spell out (i) precise layer definitions; (ii) a loss as an order-based constraint rather than a scalar; (iii) convergence guarantees; (iv) a Lean-ready module plan. The result is a mathematically exact alternative to softmax transformers: **Transformer-like expressivity, Heyting-sound operators, and lattice-theoretic training with formal guarantees**.

---

## 1. Introduction

Conventional transformers rely on continuous parameters and stochastic gradient descent (SGD), offering limited structural guarantees. Your system already provides: **LoF ⇒ nucleus (R) ⇒ Heyting core (\Omega_R)**; lenses (Tensor/Graph/Clifford) with bridge contracts; a modal dial (\theta) for phase-like behavior. We leverage this to define an **algebraic transformer** where:

* tokens, queries, keys, and values live in (\Omega_R) (or its MV/effect lifts);
* **attention** uses implication ((k\Rightarrow_R q)) and a monotone valuation (v);
* **training** updates are **residuated** (Galois-connected), producing **monotone** lattice iterations with **fixed-point existence** and **constructive convergence**.

**What we do *not* claim**: identity to Euclidean gradients or softmax attention. We provide a **superior option** when exactness, verifiability, and logical consistency are more important than continuous optimization heuristics.

---

## 2. Mathematical Foundations

### 2.1 Frames, nuclei, and fixed points

Let (L) be a **frame** (complete lattice with finite meets distributing over arbitrary joins). A **nucleus** (R:L\to L) satisfies:

* (x\le R x) (inflationary),
* (R(R x)=R x) (idempotent),
* (R(x\wedge y)=R x\wedge R y) (meet-preserving).

**Theorem 2.1 (Fixed-point Heyting).**
(\Omega_R={x\in L\mid R x=x}) with
[
a\wedge_R b:=a\wedge b,\qquad
a\vee_R b:=R(a\vee b),\qquad
a\Rightarrow_R b:=R(\neg a\vee b),\qquad
\neg_R a:=a\Rightarrow_R \bot
]
is a Heyting algebra, and **residuation** holds:
[
a\wedge_R b\le c\ \Longleftrightarrow\ b\le a\Rightarrow_R c.
]

**Theorem 2.2 (Lattice of nuclei).**
(\mathrm{Nuc}(L)={R:L\to L\mid R\text{ nucleus}}) ordered pointwise is a **complete lattice**. Meets and joins are computed pointwise.

### 2.2 Lenses & bridges (your stack)

Each lens equips a carrier (L_{\text{lens}}) with a nucleus (J) and an exact/lax **bridge** ((\mathrm{shadow},\mathrm{lift})) obeying round-trip contracts. Your established patterns give:

* **Tensor lens**: MV/effect operations with interior (\mathrm{Int}) (idempotent projector or morphological interior).
* **Graph lens**: Alexandroff interior on the reachability preorder.
* **Projector/Clifford lens**: (J(A)=\mathrm{Proj}!\big(\int UAU^{-1},d\mu\big)); meet preservation on the commuting subalgebra; complement transports exactly; joins transport laxly (and exactly on exact bridges).

All layer laws below are stated on (\Omega_R) and **transport** to lenses via your bridges.

---

## 3. Algebraic Learning as Fixed-Point Computation

### 3.1 Hypothesis/parameter lattices

Choose a complete lattice of hypotheses ((\mathcal H,\le)). Options:

* (\mathcal H=\mathrm{Nuc}(L)): the space of nuclei on a fixed carrier;
* (\mathcal H\subseteq\mathcal P(\Omega_R)): theories (closed families in (\Omega_R));
* (\mathcal H) = product lattice of layer parameters (e.g., per-head masks, thresholds, rule sets), each component a complete lattice.

### 3.2 Data as constraints; monotone updates

Encode each training item ((x,y)) as a **constraint** (\Phi(x,y;h)) monotone in (h\in\mathcal H) (e.g., “network output (\ge) target” in the lattice order). Let (\mathrm{Close}_D:\mathcal H\to\mathcal H) be the **least** closure satisfying all constraints in dataset (D).

**Assumption A (Monotonicity & continuity).**
(\mathrm{Close}_D) is monotone and (\omega)-continuous (preserves lubs of (\omega)-chains), or (\mathcal H) has finite height.

**Theorem 3.1 (Existence & convergence).**
By Tarski, (\mathrm{Close}_D) has fixed points; by Kleene, the chain (h_0\le h_1:=\mathrm{Close}_D(h_0)\le h_2:=\mathrm{Close}_D(h_1)\le\cdots) **converges** to the **least fixed point** (\mathrm{lfp}(\mathrm{Close}_D)). In finite-height lattices, convergence occurs in finitely many steps.

### 3.3 Algebraic backprop via residuation

Every forward map we use is a monotone map between Heyting algebras with a **right adjoint** (Galois connection). For fixed (a):
[
(\lambda x.\ a\wedge_R x)\ \dashv\ (\lambda z.\ a\Rightarrow_R z).
]
We define the **backward pass** of a layer by applying the **right adjoint** to propagate **required** structure upstream exactly—no real-valued gradients.

### 3.4 Order-based loss

Replace scalar loss with a **violation set** in (\Omega_R) or an order score:
[
\mathsf{defect}(y,\hat y):=(y\Rightarrow_R \hat y)\wedge_R(\hat y\Rightarrow_R y) \quad\text{maximize;}
]
or equivalently enforce constraints (y\le \hat y) when appropriate. Updates monotonically **reduce violations** by construction.

---

## 4. Heyting Attention and the Algebraic Transformer

We now define a transformer-class architecture whose attention and feed-forward are **Heyting-valid** and pair with **residuated backprop**.

### 4.1 Valuations

A **valuation** (v:\Omega_R\to[0,1]) is **monotone**, preserves (\bot, \top), and is **join-continuous on fixed points**:
[
v\big(\bigvee\nolimits_R S\big)=\sup{v(s)\mid s\in S}.
]
Examples: model counting on finite carriers; pushforward through a probability model on the graph lens; MV-algebra semiring interpretations. (For training-time normalization we only need monotonicity and sup-preservation on the support we use.)

### 4.2 Single-head Heyting attention

Tokens provide **keys** (k_j\in\Omega_R), **queries** (q_i\in\Omega_R), and **values** (v_j\in\Omega_R).

1. **Scores**: (s_{ij}:=v(k_j\Rightarrow_R q_i)\in[0,1]).
2. **Aggregation weights**: choose a monotone normalizer (\alpha:[0,1]^{n}\to[0,1]^{n}) with (\sum_j \alpha(s_i)_j=1) (e.g., softmax over (\log\frac{s+\epsilon}{1-s+\epsilon}), or (s/\sum s)).
3. **Value composition** (two regimes):

   * **Thresholded join (discrete)**: (Y_i:=\bigvee_R{, v_j\mid \alpha(s_i)_j\ge \tau,}).
   * **MV lift (graded)**: work in your MV stage: encode each (v_j) with intensity (\alpha(s_i)_j), combine by **Łukasiewicz** addition pointwise, then **lift** back via the bridge (your `stageMvAdd` + `lift`).

**Lemma 4.1 (Residuation-sound scoring).**
For any (a,b,c\in\Omega_R), (a\wedge_R b\le c \iff b\le a\Rightarrow_R c). With any monotone valuation (v), (v(a\wedge_R b)\le v(c)\iff v(b)\le v(a\Rightarrow_R c)). Hence the scoring function reflects the logical entailment pattern.

### 4.3 Multi-head & residual

Use heads (h=1,\dots,H) with possibly distinct nuclei (R_h) (or distinct bridge settings). Concatenate head outputs through a product lattice and **project** back with a nucleus (R_{\mathrm{mix}}) (idempotent “merge”). Residual:
[
\mathrm{Res}(x):=R_{\mathrm{res}}(x\vee \mathrm{HeadMix}(x)),
]
ensures (x\le \mathrm{Res}(x)) (inflationary), preserving fixed points.

### 4.4 Feed-forward as “modal breathing”

Define an expansion (E) and contraction (C) along your dial (\theta):
[
E_\theta:=R_{\theta+1}\big(,\cdot,\big),\quad C_\theta:=\text{(meet with a guard and re-project by }R_\theta).
]
The block (C_\theta\circ E_\theta) is inflationary and idempotent on (\Omega_{R_\theta}), modeling your **breathing** modality exactly.

---

## 5. Training = Fixed-Point Navigation

### 5.1 Layerwise residuated backprop

Each layer (F:\Omega_R^m\to\Omega_R^n) comes with a **right adjoint** (F^\ast:\Omega_R^n\to\Omega_R^m) satisfying
[
F(x)\le y\ \Longleftrightarrow\ x\le F^\ast(y).
]

* **Attention**: fixing (q_i), the map (v\mapsto \bigvee_R \mathcal A(q_i,k,v)) is monotone; its right adjoint sends a required output (y_i) to the **greatest** set of value requirements ((\widehat v_j)_j) consistent with residuation.
* **Feed-forward**: (E_\theta) and (C_\theta) are nuclei/meet maps; their adjoints are implications with the corresponding guards.

**Update principle**: Given constraints (\Phi(x,y;h)) at the output, push them to a **constraint set on (h)** by composing the right adjoints of the layers. The **parameter update** is the **least** (h'\ge h) satisfying those constraints—precisely the step (\mathrm{Close}_D(h)).

### 5.2 Convergence & guarantees

Under Assumption A, the iteration (h_{t+1}=\mathrm{Close}_D(h_t)) converges to (\mathrm{lfp}(\mathrm{Close}_D)). On finite lattices (e.g., finite vocabularies, finite rule banks, finite depth), convergence occurs in finitely many steps. At every step,

* **Soundness**: satisfied constraints remain satisfied (monotonicity).
* **Monotone loss**: the violation set (order defect) shrinks.
* **Idempotence**: when all constraints are satisfied, (\mathrm{Close}_D(h)=h).

---

## 6. Expressivity, Correctness, and Limits

### 6.1 Expressivity

* **Boolean circuits** embed into (\Omega_R) (Boolean phase (R=\mathrm{id})): implication, conjunction, disjunction suffice to simulate bounded-depth circuits; MV lift gives graded variants.
* **Pattern matching**: with valuation (v) derived from model counting or graph neighborhood measures, attention scores are exact predicates over implication relationships.

### 6.2 Correctness scope

* All operator laws hold **on (\Omega_R)**; on lenses we require the nucleus assumption and bridge exactness/laxness you already enforce (complements & meets exact; joins lax unless the bridge is exact).
* **Probabilistic semantics** only enter via valuations (v) and normalizers (\alpha); the core operator equalities remain purely lattice-theoretic.

### 6.3 Limitations

* Computation over large lattices can be expensive; graded MV implementations alleviate brittleness but require careful calibration.
* If you choose continuous (v,\alpha) (e.g., softmax), those pieces are approximate **interpretations** of the exact residuation scores, not logical equalities.

---

## 7. Complexity (reference implementation)

Let sequence length (n), heads (H), per-head neighborhood (k) (sparsified by threshold (\tau)).

* Score computation: (O(H, n, k, C_{\Rightarrow})) where (C_{\Rightarrow}) is the cost of computing (k\Rightarrow_R q) and applying (v).
* Aggregation (thresholded join): (O(H, n, k, C_{\vee_R})).
* MV lift: add (O(H, n, k)) scalar ops and one nucleus projection (C_J).
* Backprop (residuation): same asymptotic as forward under our layer catalog.

In typical instantiations (C_{\Rightarrow}, C_{\vee_R}, C_J) are constant or polylog in carrier size due to precomputation/tries over finite vocabularies or bounded-width structures.

---

## 8. Lean 4 formalization plan (drop-in)

**Files (aligning with your repo):**

```
LoF/
  Nucleus.lean              -- nucleus + Ω_R Heyting + residuation theorems
  HeytingCore.lean
Logic/
  AlgebraicBackprop.lean    -- Galois connections for layer catalog
  StageSemantics.lean       -- you already have; reuse
Bridges/
  Tensor.lean Graph.lean Clifford.lean -- you already have; reuse
AlgebraicTransformer/
  Valuation.lean            -- definition & properties of v, α
  Attention.lean            -- head, multi-head, residual as nuclei/meet maps
  FeedForward.lean          -- E_θ, C_θ blocks and their adjoints
  Params.lean               -- parameter lattice 𝓗, Close_D monotone update
  Convergence.lean          -- Tarski/Kleene lemmas specialized to 𝓗
Tests/
  Compliance.lean           -- RT/TRI, residuation, end-to-end toy proofs
```

**Key Lean statements (no `sorry` goals in compiled core):**

* `is_nucleus R` ⇒ `Heyting Ω_R` (construct instances; register `[simp]` lemmas).
* For each layer (F), provide `layer_right_adjoint F Fstar` and prove `F x ≤ y ↔ x ≤ Fstar y`.
* `Close_D`: prove `Monotone Close_D` and either `OmegaContinuous Close_D` or bound height of `𝓗`.
* `converges_to_lfp` via Kleene/Tarski.

---

## 9. A practical recipe (how to build & train)

### 9.1 Choose a concrete carrier & valuation

* **Graph lens** on a finite DAG of concepts; Alexandroff interior as nucleus; valuation (v) = normalized reachability measure.
* **Tensor MV lens** with ([0,1]^d) and an idempotent projector interior; valuation (v) = coordinatewise min/max summary.

### 9.2 Define heads and thresholds

* Pick (\tau\in(0,1)) (or top-(k)) to turn scores into a sparse **thresholded join**; or choose MV blending with (\alpha) (e.g., softmax) and project back by nucleus.

### 9.3 Training loop (monotone algebraic backprop)

1. Initialize (h_0=\bot_{\mathcal H}) (or a user prior).
2. For each batch (B), compute **required** output elements (constraints).
3. Push constraints backward with right adjoints to obtain parameter requirements.
4. Set (h_{t+1}:=\mathrm{Close}_B(h_t)) = the least parameter above (h_t) that satisfies the new requirements.
5. Iterate to fixed point (finite-height ⇒ finite steps).

*(This doubles as a **proof artifact**: every step preserves validity and decreases order-defect.)*

---

## 10. Discussion & positioning

This **algebraic transformer** preserves **exact Heyting laws** at all layers, replaces gradients with **residuation**, and supplies **existence and convergence theorems** from lattice theory. It is **not** a drop-in of softmax-attention or Euclidean GD; it is a **mathematically superior option** when **soundness, interpretability, and verifiability** are first-class requirements, and it interfaces cleanly with your **four lenses** via bridges.

---

## Appendix A — Canonical theorems (statements & proof sketches)

**A.1 Heyting from a nucleus.** Standard locale/frame argument.

**A.2 Right adjoint of meet.** For fixed (a), (L\ni x\mapsto a\wedge_R x) has right adjoint (L\ni z\mapsto a\Rightarrow_R z) by definition of (⇒_R).

**A.3 Head residuation.** Let (G(v,q)(\cdot)=\bigvee_R \mathcal A(q,\cdot,v)). Show (G) monotone and derive (G^\ast) by residuation on each arc; compose with threshold or MV lift—then prove (G(x)\le y\iff x\le G^\ast(y)).

**A.4 Convergence.** If ((\mathcal H,\le)) is complete and (\mathrm{Close}_D) monotone, Tarski gives existence of fixed points; if (\mathcal H) has finite height or (\mathrm{Close}_D) is (\omega)-continuous, Kleene iteration from (\bot) converges to (\mathrm{lfp}).

---

## Appendix B — Minimal Lean stubs

```lean
-- LoF/Nucleus.lean
class Nucleus (L : Type _) [CompleteLattice L] :=
  (R : L → L) (infl : ∀ x, x ≤ R x) (idem : ∀ x, R (R x) = R x)
  (meet : ∀ x y, R (x ⊓ y) = R x ⊓ R y)

def Omega {L} [CompleteLattice L] (N : Nucleus L) := {x : L // N.R x = x}

-- Heyting instance on Omega, with ∧, ∨_R, ⇒_R; prove residuation.

-- AlgebraicTransformer/Attention.lean
structure Valuation (Ω : Type _) :=
  (v : Ω → ℝ) (mono : ∀ {a b}, a ≤ b → v a ≤ v b)
  (sup_pres : ∀ S, v (Sup S) = sSup (Set.image v S))

-- define head, right adjoint, prove adjunction

-- Convergence.lean
def CloseD (h : H) : H := -- monotone closure
theorem monotone_CloseD : Monotone CloseD := ...
theorem kleene_converges : IsLfp (lfix CloseD) := ...
```

(You already have Stage/Bridge lemmas; the above integrates directly.)

---

### Final takeaway

* **Keep** your LoF-to-nucleus-to-Heyting core and bridges.
* **Replace** softmax attention with **Heyting attention**.
* **Replace** gradients with **algebraic backprop** (residuation).
* **Train** by **monotone fixed-point iteration** on a parameter lattice with formal **existence** and **convergence** guarantees.

If you want next, I can generate the **Lean modules** for `Nucleus`, `Omega` Heyting instance, `Attention` with its `right_adjoint`, and `Convergence` lemmas, specialized to one concrete lens (e.g., Graph+Alexandroff) so you can compile green immediately.
