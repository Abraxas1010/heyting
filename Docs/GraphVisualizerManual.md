# Graph Visualizer Manual

This guide explains how to regenerate, serve, and interpret the interactive dependency graph that accompanies the Laws of Form blueprint.

---

## 1. Overview

The graph visualizer is a D3-based web page generated from the blueprint sources and the current Lean build. It links every documented definition or theorem to the Lean declaration that implements it, highlights missing links, and exposes dependencies via hover/popup interactions.

Artifacts are written to `lean/Docs/LoFBlueprint/dep_graph_document/` each time the packaging script runs.

---

## 2. Prerequisites

Before regenerating the graph:

1. Ensure the Lean toolchain is available (the repo is pinned to Lean 4.24 via `lean-toolchain`).
2. Run `lake build` once so `.olean` files for `HeytingLean` are present:  
   ```bash
   cd lean
   lake build
   ```
3. Install Python 3 (used for the packaging scripts and ad‑hoc HTTP server).

---

## 3. Regenerating the Graph

Run the packaging script from the repository root to refresh all assets:

```bash
cd blueprint
./tools/package_blueprint.sh
```

The script performs the following steps:

1. Extracts the Lean targets mentioned in `src/content.tex` into `build/lean_targets.json`.
2. Executes `lake exe generateDepGraph` to export declaration metadata from the compiled `HeytingLean` modules (`build/lean_deps.json`).
3. Rebuilds `graph-data.js` and copies the web assets into `lean/Docs/LoFBlueprint/dep_graph_document/`.

If the script fails, check the console output. Warnings about missing declarations indicate a mismatched `\lean{}` tag; correct the tag or remove it.

---

## 4. Serving the Visualizer

To preview the graph locally:

```bash
cd blueprint
python3 -m http.server 8000
```

Then open <http://localhost:8000/src/dep_graph_document/index.html> in a browser.  
The Lean documentation under `lean/Docs/LoFBlueprint/dep_graph_document/index.html` contains the same files and can be hosted by any static web server.

---

## 5. Interacting with the Graph

- **Pan & Zoom:** Use the scroll wheel or trackpad to zoom; drag the canvas to pan. The initial fit runs once after the layout stabilizes and does not override manual zooming.
- **Node Dragging:** Click and drag a node to reposition a portion of the graph.
- **Hover:** Hovering over a node highlights its immediate neighbourhood and softens unrelated edges.
- **Click:** Clicking a node opens a popup with a synopsis, Lean status, documentation links, and inbound/outbound dependencies. The same information is mirrored in the details pane on the right.
- **Legend:** The legend at the top summarises node shapes and status colours (see §6).

---

## 6. Node legend

Shapes encode the type; colours encode Lean status:

| Shape  | Meaning                              |
|--------|--------------------------------------|
| Box    | Blueprint definition or data carrier |
| Ellipse| Lemma, proposition, or theorem       |

| Colour / Border              | Meaning                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| Blue border, pale blue fill  | Statement is documented and all prerequisites are ready (`statement-ready`). |
| Grey border + fill           | Narrative-only node; no Lean declaration is attached.                   |
| Orange border + fill         | Blueprint references a Lean symbol that was not found (`out-of-sync`).  |
| Green fill                   | Declaration is implemented in the repo and located during export (`formalized`). |
| Dark green fill              | Declaration is provided by Mathlib (`mathlib`).                         |

The popup shows whether a Lean declaration was found and, when present, the exported docstring.

---

## 7. Troubleshooting

| Symptom | Actions |
|---------|---------|
| `lake exe generateDepGraph` fails with "object file ... does not exist" | Run `lake build` inside `lean/` to refresh all `.olean` artefacts. |
| Many nodes marked “out-of-sync” | The corresponding `\lean{module}{declaration}` tags no longer match real Lean names. Update the blueprint or add shims. |
| Graph loads but is empty | Ensure `graph-data.js` was generated (check `lean/Docs/LoFBlueprint/dep_graph_document/js/graph-data.js`). Rerun the packaging script if necessary. |
| Popup links 404 | Verify the documentation files in `lean/Docs/LoFBlueprint/` were copied by the packaging script; rerun if stale. |

---

## 8. Maintenance Checklist

1. Keep `blueprint/src/content.tex` in sync with the Lean codebase; adjust `\lean{}` tags when modules move.
2. Run `./tools/package_blueprint.sh` after any blueprint or Lean module change to regenerate the assets.
3. Review the legend status: investigate any unexpected orange nodes (they indicate blueprint/Lean drift).
4. Commit only the relevant artefacts (`Docs/LoFBlueprint/...` and the manual); the transient files in `blueprint/build/` are ignored via `.gitignore`.

---

With this workflow the dependency graph stays aligned with the Lean project, doubles as a drill-down documentation index, and clearly flags any blueprint sections that need attention.
