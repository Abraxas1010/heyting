#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[Blueprint] Regenerating dependency graph..."
python3 "$ROOT/tools/emit_lean_targets.py"
pushd "$ROOT/../lean" >/dev/null
if command -v lake >/dev/null 2>&1; then
  lake build HeytingLean >/dev/null
  lake exe generateDepGraph
else
  echo "[Blueprint] lake executable not found; skipping Lean metadata export."
fi
popd >/dev/null

python3 "$ROOT/tools/generate_dep_graph.py"

if command -v plastex >/dev/null 2>&1; then
  echo "[Blueprint] Rendering HTML with plasTeX (plastex.cfg)..."
  plastex --config="$ROOT/plastex.cfg" "$ROOT/src/web.tex"
else
  echo "[Blueprint] plasTeX not found on PATH; skipping HTML regeneration."
fi

DEST="$ROOT/../lean/Docs/LoFBlueprint"

if [ -d "$DEST" ]; then
  echo "[Blueprint] Syncing HTML assets into $DEST"
  rsync -av --delete "$ROOT/src/web/" "$DEST/"
  rsync -av --delete "$ROOT/src/dep_graph_document/" "$DEST/dep_graph_document/"
else
  echo "[Blueprint] Destination $DEST does not exist; skipping sync."
fi

echo "[Blueprint] Done."
