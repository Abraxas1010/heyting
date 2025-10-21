#!/usr/bin/env python3
"""Generate dependency graph data for the HeytingLean blueprint."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
CONTENT_TEX = ROOT / "src" / "content.tex"
WEB_DIR = ROOT / "src" / "web"
OUTPUT_DIR = ROOT / "src" / "dep_graph_document"
GRAPH_DATA_JS = OUTPUT_DIR / "js" / "graph-data.js"
LEAN_EXPORT = ROOT / "build" / "lean_deps.json"


BEGIN_PATTERN = re.compile(
    r"\\begin\{(?P<kind>definition|lemma|theorem|proposition|corollary)\}"
    r"(?:\[(?P<title>[^\]]+)\])?"
)
END_PATTERN = re.compile(
    r"\\end\{(?P<kind>definition|lemma|theorem|proposition|corollary)\}"
)
LABEL_PATTERN = re.compile(r"\\label\{(?P<label>[^}]+)\}")
LEAN_PATTERN = re.compile(r"\\lean\{(?P<module>[^}]+)\}\{(?P<decl>[^}]+)\}")
USES_PATTERN = re.compile(r"\\uses\{(?P<uses>[^}]*)\}")
ID_PATTERN = re.compile(r'id="([^"]+)"')


@dataclass
class BlueprintNode:
    """Metadata about a blueprint declaration."""

    id: str
    kind: str
    title: str
    lean_module: Optional[str] = None
    lean_decl: Optional[str] = None
    uses: List[str] = field(default_factory=list)
    url: Optional[str] = None
    content_lines: List[str] = field(default_factory=list)

    @property
    def lean_fqn(self) -> Optional[str]:
        if self.lean_decl and self.lean_decl.startswith("HeytingLean."):
            return self.lean_decl
        if self.lean_module and self.lean_decl:
            return f"{self.lean_module}.{self.lean_decl}"
        if self.lean_module:
            return self.lean_module
        if self.lean_decl:
            return self.lean_decl
        return None


def parse_nodes() -> Dict[str, BlueprintNode]:
    """Parse nodes from the main blueprint content."""
    nodes: Dict[str, BlueprintNode] = {}
    current: Optional[BlueprintNode] = None

    with CONTENT_TEX.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            begin_match = BEGIN_PATTERN.search(line)
            if begin_match:
                kind = begin_match.group("kind")
                title = (begin_match.group("title") or kind.title()).strip()
                current = BlueprintNode(id="", kind=kind, title=title)
                continue

            if current is None:
                continue

            label_match = LABEL_PATTERN.search(line)
            if label_match:
                current.id = label_match.group("label").strip()
                continue

            lean_match = LEAN_PATTERN.search(line)
            if lean_match and current.lean_decl is None:
                raw_module = lean_match.group("module").strip()
                raw_decl = lean_match.group("decl").strip()
                current.lean_module = raw_module.replace("\\_", "_")
                current.lean_decl = raw_decl.replace("\\_", "_")
                continue

            uses_match = USES_PATTERN.search(line)
            if uses_match:
                deps = [
                    dep.strip()
                    for dep in uses_match.group("uses").split(",")
                    if dep.strip()
                ]
                current.uses.extend(deps)
                continue

            end_match = END_PATTERN.search(line)
            if end_match:
                if not current.id:
                    current = None
                    continue
                nodes[current.id] = current
                current = None
                continue

            current.content_lines.append(line)

    # Handle unterminated environment at EOF.
    if current and current.id:
        nodes[current.id] = current

    root_id = "def:primitives"
    if root_id in nodes:
        for node in nodes.values():
            if node.id == root_id:
                continue
            if not node.uses:
                node.uses.append(root_id)

    return nodes


def map_labels_to_urls(labels: Iterable[str]) -> Dict[str, str]:
    """Map LaTeX labels to generated HTML file anchors."""
    label_urls: Dict[str, str] = {}
    pending = set(labels)

    for html_path in sorted(WEB_DIR.glob("*.html")):
        contents = html_path.read_text(encoding="utf-8", errors="ignore")
        for match in ID_PATTERN.finditer(contents):
            label = match.group(1)
            if label in pending and label not in label_urls:
                label_urls[label] = f"../web/{html_path.name}#{label}"

    return label_urls


def ensure_directories() -> None:
    (OUTPUT_DIR / "js").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "styles").mkdir(parents=True, exist_ok=True)


def load_lean_metadata() -> Dict[str, dict]:
    if not LEAN_EXPORT.exists():
        return {}
    try:
        data = json.loads(LEAN_EXPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed Lean metadata JSON: {exc}") from exc
    result: Dict[str, dict] = {}
    for entry in data.get("constants", []):
        name = entry.get("lean")
        if isinstance(name, str):
            result[name] = entry
    return result


def strip_tex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\lean\{[^}]*\}\{[^}]*\}", "", text)
    text = re.sub(r"\\uses\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\\", "", text)
    text = text.replace(r"\item", " - ")
    text = text.replace("itemize", "")
    text = text.replace("enumerate", "")
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def classify_status(node: BlueprintNode, lean_entry: Optional[dict]) -> str:
    if lean_entry:
        if not lean_entry.get("found", False):
            return "out-of-sync"
        lean_name = lean_entry.get("lean", "")
        if isinstance(lean_name, str) and lean_name.startswith("Mathlib"):
            return "mathlib"
        return "formalized"
    if node.lean_fqn:
        return "out-of-sync"
    if node.uses:
        return "statement-ready"
    return "needs-work"


def write_graph_data(nodes: Dict[str, BlueprintNode], lean_meta: Dict[str, dict]) -> None:
    """Serialise nodes and edges to a JavaScript module."""
    ensure_directories()

    edges = set()
    missing: Dict[str, BlueprintNode] = {}
    lean_to_labels: Dict[str, List[str]] = {}
    for node in nodes.values():
        if node.lean_fqn:
            lean_to_labels.setdefault(node.lean_fqn, []).append(node.id)

    for node in nodes.values():
        for dep in node.uses:
            if dep:
                edges.add((dep, node.id))
            if dep and dep not in nodes and dep not in missing:
                missing[dep] = BlueprintNode(
                    id=dep,
                    kind="external",
                    title=dep,
                )

    full_nodes = {**nodes, **missing}
    url_map = map_labels_to_urls(full_nodes.keys())

    node_payload = []
    for node_id, node in sorted(full_nodes.items(), key=lambda item: item[0]):
        summary = strip_tex(" ".join(node.content_lines)) if node.content_lines else ""
        summary = summary[:420]
        shape = "box" if node.kind == "definition" else "ellipse"
        lean_entry = lean_meta.get(node.lean_fqn or "")
        status = classify_status(node, lean_entry)
        lean_found = bool(lean_entry and lean_entry.get("found"))
        lean_doc = lean_entry.get("doc") if lean_entry else None
        lean_deps = []
        if lean_entry and isinstance(lean_entry.get("deps"), list):
            lean_deps = [
                dep for dep in lean_entry["deps"] if isinstance(dep, str)
            ]
        payload = {
            "id": node_id,
            "name": node.title,
            "kind": node.kind,
            "lean": node.lean_fqn,
            "url": url_map.get(node_id),
            "isPlaceholder": node.kind == "external",
            "summary": summary or None,
            "shape": shape,
            "status": status,
            "leanFound": lean_found,
            "leanDoc": lean_doc,
            "leanDeps": lean_deps,
        }
        node_payload.append(payload)

    for node_id, node in full_nodes.items():
        deps = set(dep for dep in node.uses if dep)
        lean_entry = lean_meta.get(node.lean_fqn or "")
        if lean_entry and lean_entry.get("found"):
            for dep_name in lean_entry.get("deps", []):
                if not isinstance(dep_name, str):
                    continue
                for label in lean_to_labels.get(dep_name, []):
                    deps.add(label)
        if node_id != "def:primitives" and not deps:
            deps.add("def:primitives")
        for dep in deps:
            if dep in full_nodes:
                edges.add((dep, node_id))

    edge_payload = [
        {"source": source, "target": target}
        for source, target in sorted(edges)
        if source in full_nodes and target in full_nodes
    ]

    graph_data = {"nodes": node_payload, "edges": edge_payload}

    GRAPH_DATA_JS.write_text(
        "window.depGraphData = "
        + json.dumps(graph_data, indent=2, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )


def main() -> None:
    nodes = parse_nodes()
    lean_meta = load_lean_metadata()
    write_graph_data(nodes, lean_meta)


if __name__ == "__main__":
    main()
