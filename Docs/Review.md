# HeytingLean Review Checklist

## Goals
- Validate ontology narrative aligns with Lean artifacts (`Docs/Ontology.md`).
- Exercise compliance suite (`lean/HeytingLean/Tests/Compliance.lean`) before sign-off.
- Confirm bridges and lenses reflect the Euler boundary encodings and Boolean limit witness.

## Suggested Session Agenda
1. **Context recap** – 10 minutes (overview of phases 0–5 progress).
2. **Live walkthrough** – 30 minutes: run `lake build`, review `Docs/Ontology.md`, inspect tests (`tensor_encode_euler`, `graph_encode_euler`, `clifford_encode_euler`, `boolean_limit_verified`).
3. **Open issues** – 15 minutes: discuss optional MV/effect/orthomodular roadmap and any automation backlog.
4. **Action items** – capture follow-ups in `Docs/STATUS.md` and update cards/tickets.

## Artifacts to Distribute
- `Docs/Ontology.md` (ontology narrative and breathing example).
- `Docs/Semantics.md` (MV/effect/orthomodular synopsis).
- `Docs/STATUS.md` (current status snapshot).
- Latest compliance log (from `lake build`).

Use this checklist when scheduling the final review with the Lean and ontology teams.
