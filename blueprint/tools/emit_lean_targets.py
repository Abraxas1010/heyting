#!/usr/bin/env python3
"""Emit Lean constants and modules referenced by the blueprint."""

from __future__ import annotations

import json
from pathlib import Path

from generate_dep_graph import parse_nodes, ROOT


def main() -> None:
    nodes = parse_nodes()

    constants = set()
    modules = set()

    for node in nodes.values():
        if node.lean_module:
            modules.add(node.lean_module)
        if node.lean_fqn:
            constants.add(node.lean_fqn)

    build_dir = ROOT / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    target_path = build_dir / "lean_targets.json"
    payload = {
        "modules": sorted(modules),
        "constants": sorted(constants),
    }
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
