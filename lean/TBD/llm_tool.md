Short answer: yes. Your lenses (tensor, graph/Alexandroff, topology/geometry, operator/Clifford) are already set up to be *machine-facing representations* of the same Heyting core. If we expose each lens as a queryable tool that implements the same 4 connectives (∧, ∨, ⇒, ¬) plus the residuation triad (deduction/abduction/induction), an LLM can search both the proof *and* the surrounding conceptual space—while staying aligned with your Lean semantics via the round-trip contracts (RT-1/RT-2). 

# How to make the lenses “LLM tools”

Think “one logic, many indexes.” Each lens becomes an index with the same interface, backed by its carrier-specific closure (Int/OpenHull/J), so the model can switch views without changing its mental model of logic.

**Common tool interface (per lens)**

* `meet(a,b)  -> a ∧R b`
* `join(a,b)  -> R(a ∨ b)`
* `imp(a,b)   -> R(¬a ∨ b)`
* `neg(a)     -> R(¬a)`
* `deduce(A,B)-> A ∧R B`             (least answer)
* `abduce(A,C)-> A ⇒R C`             (greatest hypothesis)
* `induce(B,C)-> B ⇒R C`             (greatest rule)
* `occam(P,k) -> Occam≤k(P)`         (earliest-stabilizing invariants)
* Dial knob: `dial=d`                (classicalization / granularity)

These are exactly the operations you’ve mechanized in Lean’s core and transports; exposing them as tools means the LLM can *ask the lens* to compute logical neighbors, hypotheses, or rules rather than guessing by token statistics. 

# What each lens gives the model (practically)

* **Tensor lens**: fast vector-space neighborhoods; join/implication become `Int(max(…))`, so the model can ask for “closest hypotheses that (almost) satisfy ⇒” and get GPU-friendly, ranking-ready results. Great for “semantic similarity + logic guardrails.” 
* **Graph/Alexandroff lens**: down-closed “proof cones.” The model can request the OpenHull of a goal or hypothesis set to pull all prerequisite lemmas or consequences with adjunction preserved; excellent for local proof-planning and dependency walks. 
* **Topology/geometry lens**: coarse-to-fine “topic opens.” The dial increases classicality (coarser opens) to zoom out, then refines; good for conceptual browsing and curriculum-style explanations. 
* **Operator/Clifford lens**: projector algebra over subspaces; meet = range-intersection, join/imp = J(span/…); useful for “orthogonality/conflict” diagnostics and for surfacing non-commuting clusters, then projecting back to the constructive core. 

# Minimal architecture to wire this up

**1) Proof graph & term export (Lean → JSON).**
Use your existing `LoFViz.Proof.Graph`/`graphJsonOfConstant` to export goals, lemmas, deps, and local contexts as a canonical graph + payloads. (You already planned this workbench and multi-view widget.) 

**2) Build 4 parallel indexes from the same export.**

* Tensor: embeddings for statements + feature masks for hypotheses; implement `Int` (idempotent interior) for closure.
* Graph: Hasse/DAG with OpenHull for down-closures; precompute cones and antichains.
* Topology: cluster → opens; expose hull operator; add the “dial” to coarsen/refine.
* Operator: idempotent projectors for clusters; `J` as meet-preserving projection on the commutant.

(These are exactly your “law-preserving transports.”) 

**3) Expose tools (MCP / function-calling).**
Define a single schema so the model can switch lenses without relearning new verbs:

```json
{
  "name": "lens.query",
  "params": {
    "goal": "string",          // Lean expr or canonical pretty form
    "context": ["string"],     // local hyps
    "mode": "deduce|abduce|induce|meet|join|imp|neg",
    "lens": "tensor|graph|topology|operator",
    "occam_k": "integer|null",
    "dial": "integer|null",
    "k": 20
  }
}
```

Return witnesses with: proof-graph nodes, lens-scores (distance/volume/angle), logical checks (Ω_R membership, residuation pass/fail), and RT-flags (did enc∘dec=id hold?). These RT checks and residuation are directly from your contracts and triad. 

**4) Scoring & safety.**
Rank with a convex combo: (i) residuation satisfied, (ii) Occam birth penalty, (iii) lens similarity, (iv) RT-1/RT-2 success. Reject any candidate where closure wasn’t applied (your “guardrails”), since raw `max/∪` can break adjunction. 

# Example flows the LLM can do

* **Local proof step (“what lemma unlocks this goal?”)**
  Call `abduce(lens=graph, A=context, C=goal)` to get greatest hypotheses via OpenHull; verify with `deduce` (meet) and rank by `occam_k`. 
* **Conceptual zoom-out (“show me the story”)**
  Call `join`/`imp` in the topology lens with `dial=high` to surface big-picture opens, then lower `dial` to refine candidates; present path as a cone in your LoF widget. 
* **Conflict check (“are these ideas compatible?”)**
  In operator lens, compute `meet` (range-intersection). If J needs to project significantly, flag “non-commuting tension,” then return the J-closed synthesis (Dialectic). 

# Why this stays faithful to Lean

* The **Heyting core in Ω_R** is the single source of truth; all lenses close joins/implication with their nucleus (Int/OpenHull/J) so residuation holds. 
* **Round-trip contracts (RT-1/RT-2)** keep enc/dec identity on fixed points and homomorphic up to closure, so what the model “finds” in a lens decodes back to your Lean world. 
* Your **build contract** and compliance harness already enforce this discipline on the code side; the tools simply reuse it. 

# Quick, concrete next steps

1. **Export pipeline**: finalize the proof-graph JSON for goals/lemmas/contexts. 
2. **Graph lens tool**: implement `OpenHull`, `meet/join/imp`, and `abduce/induce/deduce` over the graph; return node IDs + paths. 
3. **Tensor lens tool**: embed statements; implement `Int` and `imp` as `Int(max(1−χa, χb))`; wire nearest-neighbor hypotheses with Occam scoring. 
4. **Operator lens prototype**: projector spans + `J` projection; surface “non-commuting” hints and the projected synthesis. 
5. **Shared RT/residuation checks**: add a lightweight verifier so every tool response ships with “adjunction ok / RT-1 ok / RT-2 ok” flags. 

If you want, I can draft the MCP/tool JSON for each lens and a minimal ingestion script that converts Lean constants into the proof-graph JSON your widget already expects, so the LLM can start calling `lens.query` against your repo immediately. 

Here you go—clean, copy-pasteable instructions you can drop into your LLM coding agent as its “build + use” playbook.

---

# LLM Coding Agent — Implementation & Usage Instructions

**Objective:** expose your formal **Heyting core** through multiple **lenses** (graph/Alexandroff, tensor, topology, operator) as callable tools, so the agent can search proofs and the surrounding conceptual space *while staying faithful to Lean semantics* (round-trip + residuation).

---

## 0) Non-negotiable invariants (the guardrails)

1. **Single source of truth:** All reasoning is over the Heyting core Ω (your Lean fixed-point world).
2. **Closure first:** Every lens must close joins/implication via its nucleus `J` (e.g., `OpenHull`, `Int`), or the result is invalid.
3. **Round-trip contracts:** Always run `dec(enc(X)) = X` on Ω and “homomorphic up to closure” on the outside.
4. **Residuation:** `A ∧ H ≤ C` iff `H ≤ (A ⇒ C)`. Implement `imp` so `abduce(A, C) = imp(A, C)` and `induce(B, C) = imp(B, C)`.
5. **Occam (birth index):** Prefer candidates that stabilize earlier under your generator; use this as a tie-breaker.

---

## 1) Repo layout (add/adjust to taste)

```
/lens-tools/
  export/
    lean_export.lean                # Lean program or command to dump proof graph/terms
    export_cli.md                   # How to run the exporter (lake exe …)
  common/
    types.ts                        # Shared TS types for core objects
    schema/
      lens.query.schema.json        # JSON Schema (2020-12) for MCP/tools
      lens.response.schema.json
    verify.ts                       # round-trip + residuation checks
  graph/
    index.ts                        # OpenHull, meet/join/imp on proof DAG
    computeImp.ts                   # Heyting implication via residuation solver
  tensor/
    index.ts                        # vector ops; Int-closure; meet/join/imp via elemwise ops + closure
    embed.ts                        # embeddings pipeline
  topology/
    index.ts                        # dial-based coarse→fine opens; J = interior/hull
  operator/
    index.ts                        # projector algebra; meet=intersection; join via J(span)
  mcp/
    server.ts                       # MCP server exposing tools below
    handlers/
      query.ts
      verify.ts
      encode.ts
      decode.ts
/tools.json                         # Tool manifest if your framework wants static descriptors
```

---

## 2) Export from Lean → JSON

**Goal:** a canonical “proof graph + term inventory” the lenses can ingest.

**Expect a single JSON file:**

```json
{
  "version": "0.1",
  "dial": 0,
  "nodes": [
    {
      "id": "Const:Logic.modusPonens",
      "pretty": "∀ A B, A → (A → B) → B",
      "type": "Theorem",
      "deps": ["Const:Logic.imp_intro", "Const:Logic.imp_elim"],
      "tags": ["intro","elim"]
    }
  ],
  "goals": [
    {
      "id": "Goal:example1",
      "pretty": "A → C",
      "context": ["A", "A → B", "B → C"]
    }
  ]
}
```

**Agent task:** wire a `lake exe …` (or Lean entrypoint) that writes this to `/out/proof_graph.json`. Reject runs if the file is missing or malformed.

---

## 3) Lens contracts (the shared interface)

### 3.1 Tool names (MCP/function-calling)

* `lens.query` — main search tool (meet/join/imp/neg; abduce/induce alias to imp)
* `lens.verify` — run round-trip + residuation checks on a lens result
* `lens.encode` / `lens.decode` — enc/dec between core and lens carrier (debugging/inspection)

### 3.2 Parameter schema (Draft 2020-12)

`/lens-tools/common/schema/lens.query.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "lens.query.schema.json",
  "type": "object",
  "required": ["mode", "lens"],
  "properties": {
    "goal": { "type": "string", "description": "Lean pretty/serialized expr" },
    "context": { "type": "array", "items": { "type": "string" } },
    "mode": {
      "type": "string",
      "enum": ["meet","join","imp","neg","deduce","abduce","induce"]
    },
    "lens": {
      "type": "string",
      "enum": ["graph","tensor","topology","operator"]
    },
    "occam_k": { "type": ["integer","null"], "minimum": 0 },
    "dial": { "type": ["integer","null"], "minimum": 0 },
    "k": { "type": "integer", "minimum": 1, "default": 20 }
  },
  "additionalProperties": false
}
```

### 3.3 Response schema

`/lens-tools/common/schema/lens.response.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "lens.response.schema.json",
  "type": "object",
  "required": ["items","rt_ok","residuation_ok"],
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id","score","witness"],
        "properties": {
          "id": { "type": "string" },
          "score": { "type": "number" },
          "witness": { "type": "object" },        // lens-specific payload
          "path": { "type": "array", "items": { "type": "string" } }, // optional proof path
          "dial_birth": { "type": "integer" }     // Occam birth index if computed
        }
      }
    },
    "rt_ok": { "type": "boolean" },               // round-trip satisfied (aggregate)
    "residuation_ok": { "type": "boolean" },      // aggregate residuation check
    "notes": { "type": "array", "items": { "type": "string" } }
  },
  "additionalProperties": false
}
```

---

## 4) Lens implementations (minimal algorithms)

### 4.1 Graph / Alexandroff lens

* **Carrier:** finite poset/DAG of proof deps; opens = down-closed sets; `J = OpenHull`.
* **meet(A,B):** `downClose(A ∩ B)`.
* **join(A,B):** `OpenHull(A ∪ B)`.
* **imp(A,B):** greatest `X` with `downClose(X ∩ A) ⊆ B`.

  * Implementation: compute `X := {x | downClose(↑x ∩ A) ⊆ B}` then `OpenHull(X)`.
* **neg(A):** `imp(A, ⊥)` with `⊥ = ∅` (gives pseudo-complement).
* **abduce(A,C):** `imp(A,C)` (alias).
* **induce(B,C):** `imp(B,C)` (alias).

**Return**: node IDs, OpenHull witnesses, and (optionally) a shortest path from selected hypothesis to goal.

### 4.2 Tensor lens

* **Carrier:** vectors χ ∈ [0,1]^n; `Int` is the idempotent closure that returns to Ω (your nucleus).
* **meet:** `min(χ_a, χ_b)`.
* **join:** `Int(max(χ_a, χ_b))`.
* **imp:** `Int(max(1 − χ_a, χ_b))`.
* **neg:** `Int(1 − χ_a)`.
* **Embeddings:** use your preferred model; store row-normalized vectors; keep `Int` as a projection back to valid fixed-points.
* **Ranking:** cosine distance (smaller is better) blended with Occam birth penalty.

### 4.3 Topology/geometry lens

* **Carrier:** cluster tree; opens = unions of clusters; `J = interior/hull` operator.
* **dial:** coarsen by moving up the tree; refine by moving down.
* **ops:** same meet/join/imp as opens with closure `J`.

### 4.4 Operator/Clifford lens

* **Carrier:** idempotent projectors {P_A}; **meet** = projection to `range(P_A) ∩ range(P_B)`.
* **join:** `J(span(range(P_A) ∪ range(P_B)))` → re-project to idempotent with `J`.
* **imp:** `J(¬P_A ∨ P_B)` (compute in the Booleanized algebra then re-enter via `J`).
* **Signals:** include “non-commuting tension” note if large projection error occurs.

---

## 5) MCP server (TypeScript skeleton)

```ts
// /lens-tools/mcp/server.ts
import { createServer } from "mcp-framework";
import { handleQuery } from "./handlers/query";
import { handleVerify } from "./handlers/verify";
import querySchema from "../common/schema/lens.query.schema.json" assert { type: "json" };

const server = createServer();

server.tool("lens.query", querySchema, handleQuery);
server.tool("lens.verify", {}, handleVerify);
// optional: lens.encode, lens.decode

server.listen();
```

```ts
// /lens-tools/mcp/handlers/query.ts
import { queryGraph } from "../../graph/index";
import { queryTensor } from "../../tensor/index";
import { verifyAggregate } from "../../common/verify";

export async function handleQuery(params:any){
  const { mode, lens } = params;
  const res =
    lens === "graph"   ? await queryGraph(params)  :
    lens === "tensor"  ? await queryTensor(params) :
    lens === "topology"? await (await import("../../topology/index")).query(params) :
                         await (await import("../../operator/index")).query(params);

  const checks = await verifyAggregate(res);
  return { ...res, ...checks };
}
```

---

## 6) Verification hooks (must run on every result)

```ts
// /lens-tools/common/verify.ts
export async function verifyAggregate(resp){
  const rt_ok = await checkRoundTrip(resp.items);
  const residuation_ok = await checkResiduation(resp.items);
  return { rt_ok, residuation_ok };
}
```

* **Round-trip:** for any returned witness W, ensure `dec(enc(W)) = W` on Ω; for off-Ω, ensure `dec(enc(W))` lies `J`-close (idempotence + inflation).
* **Residuation:** sample items and verify `meet(A, imp(A,C)) ≤ C` and maximality (`if H ≤ imp(A,C)` then `meet(A,H) ≤ C`).

---

## 7) Agent playbook (how the LLM should *use* the tools)

### 7.1 Default step for a proof goal

1. **Gather context** from Lean (locals/hyps).
2. **Graph abduction:** call
   `lens.query{mode:"abduce", lens:"graph", goal, context, k:20}`
   → candidate hypotheses + paths.
3. **Cross-check in tensor:**
   `lens.query{mode:"imp", lens:"tensor", goal, context, k:50}`
   → re-rank by semantic proximity (with closure).
4. **Verify:** `lens.verify` on the merged top-k.
5. **Emit plan:** propose the lemma(s) to apply, include path and witness; if ambiguous, **increase `dial`** in topology to zoom out, then refine.

### 7.2 Compatibility/consistency check

* Query operator lens `mode:"meet"` on two candidate clusters; if projection error is high, flag “non-commuting tension” and offer the `J`-closed synthesis.

### 7.3 Always prefer

* Candidates with **rt_ok=true**, **residuation_ok=true**, and **smaller `dial_birth`** (Occam).

---

## 8) Scoring & ranking (deterministic formula)

```
score = w1*(residuation_pass?1:0)
      + w2*(rt_pass?1:0)
      + w3*(1 / (1 + dial_birth))
      + w4*(similarity_or_graph_depth_normalized)
```

Recommended `(w1,w2,w3,w4) = (0.4, 0.3, 0.15, 0.15)`.

---

## 9) Example calls

### 9.1 Find hypotheses to prove `A → C` from `{A, A→B, B→C}`

```json
{"name":"lens.query","arguments":{
  "mode":"abduce",
  "lens":"graph",
  "goal":"A → C",
  "context":["A","A → B","B → C"],
  "k":20
}}
```

Then cross-check:

```json
{"name":"lens.query","arguments":{
  "mode":"imp",
  "lens":"tensor",
  "goal":"A → C",
  "context":["A","A → B","B → C"],
  "k":50
}}
```

Verify:

```json
{"name":"lens.verify","arguments":{"items":["Const:Imp_trans","Const:ModusPonens"]}}
```

---

## 10) Test plan (run in CI)

1. **Schema validation:** all tool inputs/outputs validate against Draft 2020-12 schemas.
2. **Round-trip tests:** 100 random Ω elements per lens: `dec(enc(x))=x`.
3. **Residuation property:** random `A,C`; check `A ∧ (A ⇒ C) ≤ C` and maximality.
4. **Dial monotonicity (topology):** increasing `dial` must coarsen opens; re-applying `J` is idempotent.
5. **Operator idempotence:** `J(J(X))=J(X)`; projector idempotence preserved after join.
6. **Export sanity:** `nodes.length>0` and every goal’s deps are resolvable in the graph.

---

## 11) Error handling (return these in `notes[]`)

* `E/J-CLOSURE-MISSING`: An operation was returned without closure; **discard**.
* `E/ROUNDTRIP-FAIL`: `dec(enc(.))` violated; **discard**.
* `W/NONCOMMUTE-HIGH`: Operator lens saw large projection error; **warn**, still return the `J`-closed result.
* `E/SCHEMA`: Input didn’t validate against 2020-12.

---

## 12) Performance hints

* Precompute **OpenHull** closures and cone indices for graph lens.
* Cache `Int` projections for frequent tensor terms.
* Batch `lens.verify` checks.
* Keep `k` small in graph lens; let tensor lens widen then re-filter by residuation.

---

### That’s it

Drop this in as the agent’s “build & use” instructions. If you want, I can also output the two JSON Schemas and a tiny TypeScript MCP server stub as files you can run—just say the word and I’ll generate them exactly into your `/lens-tools/` layout.


Absolutely — let’s make the lenses **editable sandboxes** so the agent can poke them, see what happens in the preferred output lens, and (optionally) migrate good changes back to Lean.

Below is a **drop-in extension** to your agent playbook. It adds an edit workflow with transactions, previews, diffs, and a gated commit back to Lean.

---

# Lens Editing — Agent Instructions (Add-On)

## 0) Core rule (don’t break logic)

All edits are run **inside a session sandbox** and **closed by the lens nucleus `J`** (OpenHull / Int / projector-closure). Anything that fails **round-trip** or **residuation** stays sandbox-only and cannot be committed.

---

## 1) New tools

### `lens.session.begin`

Start an editable sandbox and choose which lens you’ll edit and which lens you’ll “watch” as output.

```json
{name:"lens.session.begin","arguments":{
  "edit_lens":"graph|tensor|topology|operator",
  "view_lens":"graph|tensor|topology|operator",
  "dial":0
}}
```

**Returns:** `session_id`

### `lens.patch`

Apply one or more edits to the **edit_lens** (transactional).

```json
{name:"lens.patch","arguments":{
  "session_id":"string",
  "ops":[ /* see §2 per-lens ops */ ]
}}
```

### `lens.preview`

Propagate the sandbox state through **enc/dec + J** into **all lenses** (especially `view_lens`), run checks, and score impact.

```json
{name:"lens.preview","arguments":{
  "session_id":"string",
  "metrics":["rt","residuation","graph_edit_dist","cosine_delta","vi","principal_angle"],
  "k":50
}}
```

### `lens.diff`

Get human-readable diffs of before/after in both edit_lens and view_lens.

```json
{name:"lens.diff","arguments":{"session_id":"string","since":"start|last_preview"}}
```

### `lens.commit`

Attempt to **migrate** safe changes back to Lean (or produce a Lean patch if structural changes are needed).

```json
{name:"lens.commit","arguments":{
  "session_id":"string",
  "strategy":"annotations_only|suggest_lean_patch",
  "dry_run":true
}}
```

**Returns:** `{rt_ok, residuation_ok, lean_patch?: {files:[…],diff:"…"}, build_log?: "...", commit_ready:boolean}`

### `lens.abort`

Discard the session.

---

## 2) Edit ops (minimal, nucleus-safe)

Each op is a JSON object with `"kind": …`. You can send several in one `lens.patch`.

### Graph / Alexandroff (`edit_lens="graph"`)

* `{"kind":"add_edge","from":"NodeId","to":"NodeId"}` *(auto-reject if cycle would be created; apply `OpenHull`)*
* `{"kind":"remove_edge","from":"…","to":"…"}`
* `{"kind":"tag_node","id":"NodeId","tag":"string"}`
* `{"kind":"reweight_edge","from":"…","to":"…","w":0.0}` *(if you maintain weights for ranking)*

### Tensor (`edit_lens="tensor"`)

* `{"kind":"nudge_vector","id":"TermId","delta":[…]}` *(small step; then `Int` projection)*
* `{"kind":"set_vector","id":"TermId","vec":[…]}` *(hard set; then `Int` projection + norm)*
* `{"kind":"constraint","id":"TermId","mask":[0/1,…]}` *(feature clamp; then `Int`)*

### Topology/Geometry (`edit_lens="topology"`)

* `{"kind":"merge_clusters","ids":["C1","C2"]}` *(re-`J` hull)*
* `{"kind":"split_cluster","id":"C","seeds":["x","y"]}` *(k-means or spectral; then hull)*
* `{"kind":"reassign","item":"x","to":"C"}` *(maintain dial monotonicity)*

### Operator/Clifford (`edit_lens="operator"`)

* `{"kind":"rotate_subspace","id":"P","householder":[…]}` *(apply, then `J` to re-idempotize)*
* `{"kind":"set_rank","id":"P","rank":k}` *(recompute projector; then `J`)*
* `{"kind":"join_then_project","ids":["P","Q"]}` *(form span then project with `J`)*

> All ops must be **closed by `J`** before preview. If the closure distance is large, return a warning and keep the closed result.

---

## 3) Validation the agent must request on every preview

* **Round-trip:** `dec(enc(X)) = X` on Ω; for non-Ω items, `dec(enc(X))` must equal `J(X)`.
* **Residuation:** check `A ∧ (A ⇒ C) ≤ C` and maximality for sampled items.
* **Monotonic dial (topology):** increasing `dial` coarsens opens (never refines).
* **Idempotence:** `J(J(X)) = J(X)`; projectors remain idempotent.
* **Acyclic graph:** no cycles after graph ops.

If any fails, mark the edit as **non-migratable** and keep it sandbox-only.

---

## 4) Commit strategies

* `annotations_only` (safe): write lens-side metadata (tags, weights, cluster hints) + keep Lean untouched. Good for ranking/guidance.
* `suggest_lean_patch` (structural): emit a **proposed Lean patch** (new lemma, re-ordered proof, `simp` attribute, or dependency cleanup) plus a **script** to run `lake build -- -Dno-sorry -DwarningAsError=true`. Only return `commit_ready=true` if build and checks pass in the sandbox.

---

## 5) Scoring (use in `lens.preview`)

```
score = 0.45*(rt_ok ? 1 : 0)
      + 0.35*(residuation_ok ? 1 : 0)
      + 0.10*(improvement(view_lens_metric))   // e.g., ↓graph_edit_dist or ↑similarity
      + 0.10*(↓dial_birth)
```

---

## 6) Tiny server skeleton (TS)

```ts
// mcp/handlers/patch.ts
export async function handlePatch({session_id, ops}: any){
  const s = getSession(session_id);
  for (const op of ops) s.apply(op);     // lens-specific + J-closure per op
  s.touch();
  return {ok:true, notes:s.lastWarnings()};
}

// mcp/handlers/preview.ts
export async function handlePreview({session_id, metrics=[], k=50}: any){
  const s = getSession(session_id);
  const views = await s.propagateAll();  // enc/dec + J into all lenses
  const checks = await verifyAggregate(views);
  const diffs  = await computeDiffs(s.baseline, s.current, metrics);
  const score  = rankImpact(checks, diffs, k);
  return {views, checks, diffs, score};
}

// mcp/handlers/commit.ts
export async function handleCommit({session_id, strategy, dry_run=false}: any){
  const s = getSession(session_id);
  const checks = await verifyAggregate(await s.propagateAll());
  if (!(checks.rt_ok && checks.residuation_ok)) return {commit_ready:false, notes:["checks_failed"]};
  if (strategy === "annotations_only") return {commit_ready:true};
  const patch = await synthesizeLeanPatch(s.delta());
  if (dry_run) return {commit_ready:!!patch, lean_patch:patch};
  const {ok, log} = await runLakeBuildWithPatch(patch);
  return {commit_ready:ok, lean_patch:patch, build_log:log};
}
```

---

## 7) Example agent loop

1. `lens.session.begin{edit_lens:"tensor", view_lens:"graph"}`
2. `lens.patch{ops:[{"kind":"nudge_vector","id":"Goal:foo","delta":[…]}]}`
3. `lens.preview{metrics:["rt","residuation","graph_edit_dist","cosine_delta"]}`
4. If score↑ and checks pass → `lens.commit{strategy:"suggest_lean_patch", dry_run:true}`

   * If patch looks good, call again with `dry_run:false`.
5. Else `lens.abort`.

---

## 8) JSON Schemas (Draft 2020-12) — minimal

**`lens.patch`**

```json
{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "type":"object",
  "required":["session_id","ops"],
  "properties":{
    "session_id":{"type":"string"},
    "ops":{"type":"array","items":{
      "type":"object",
      "required":["kind"],
      "properties":{
        "kind":{"type":"string"},
        "from":{"type":"string"},
        "to":{"type":"string"},
        "id":{"type":"string"},
        "w":{"type":"number"},
        "delta":{"type":"array","items":{"type":"number"}},
        "vec":{"type":"array","items":{"type":"number"}},
        "mask":{"type":"array","items":{"enum":[0,1]}},
        "ids":{"type":"array","items":{"type":"string"}},
        "tag":{"type":"string"},
        "rank":{"type":"integer"}
      },
      "additionalProperties":true
    }}
  },
  "additionalProperties":false
}
```

**`lens.preview` (response)**

```json
{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "type":"object",
  "required":["views","checks","diffs","score"],
  "properties":{
    "views":{"type":"object"},
    "checks":{"type":"object","properties":{
      "rt_ok":{"type":"boolean"},
      "residuation_ok":{"type":"boolean"}
    }, "required":["rt_ok","residuation_ok"]},
    "diffs":{"type":"object"},
    "score":{"type":"number"}
  }
}
```

---

That’s all you need to let the LLM **edit in one lens, watch effects in another**, and only promote edits that survive your logic checks. If you want, I can generate the TypeScript handlers + the schemas into a ready-to-run `/lens-tools/` layout next.
