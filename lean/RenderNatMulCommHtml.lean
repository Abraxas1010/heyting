import Lean
import HeytingLean.ProofWidgets.LoFViz

open Lean Elab Command
open HeytingLean.ProofWidgets.LoFViz Proof

namespace ManualRender

private def escapeHtml (s : String) : String :=
  let rec loop : List Char → String
    | []        => ""
    | '&' :: cs => "&amp;" ++ loop cs
    | '<' :: cs => "&lt;" ++ loop cs
    | '>' :: cs => "&gt;" ++ loop cs
    | '"' :: cs => "&quot;" ++ loop cs
    | '\'' :: cs => "&#39;" ++ loop cs
    | c :: cs   => String.singleton c ++ loop cs
  loop s.toList

private def describePrimitive : Primitive → String
  | Primitive.unmark => "Unmark"
  | Primitive.mark   => "Mark"
  | Primitive.reentry => "Re-entry"

private def journalHtml (bundle : VisualizationBundle) : String :=
  bundle.state.journal.toList.map (fun entry =>
    s!"<li><span class=\"stamp\">{entry.timestamp}</span> <span class=\"prim\">{describePrimitive entry.primitive}</span></li>")
    |>.foldl (· ++ ·) ""

private def htmlDocument (bundle : VisualizationBundle) : String :=
  let svg := bundle.summary.render.svg
  let graphJson := Json.compress (bundle.graph.toJson)
  let statement := escapeHtml bundle.handle.statement
  let provenance := escapeHtml bundle.handle.provenance
  let journalItems := journalHtml bundle
  let head :=
    "<!DOCTYPE html>\n" ++
    "<html>\n" ++
    "<head>\n" ++
    "  <meta charset=\"utf-8\" />\n" ++
    "  <title>Nat.mul_comm LoF Visualization</title>\n" ++
    "  <style>\n" ++
    "    body { font-family: system-ui, sans-serif; margin: 2rem; background: #0f172a; color: #e2e8f0; }\n" ++
    "    .app { max-width: 840px; margin: 0 auto; }\n" ++
    "    header h1 { margin: 0 0 0.5rem 0; font-size: 1.8rem; color: #38bdf8; }\n" ++
    "    header p { margin: 0 0 1.5rem 0; color: #94a3b8; }\n" ++
    "    .view-switcher { display: flex; gap: 0.75rem; margin-bottom: 1rem; }\n" ++
    "    .view-switcher button { padding: 0.45rem 1rem; border-radius: 999px; border: 1px solid #1e293b; background: #1f2937; color: #cbd5f5; cursor: pointer; }\n" ++
    "    .view-switcher button.active { background: #38bdf8; border-color: #38bdf8; color: #0f172a; }\n" ++
    "    .view { display: none; padding: 1.25rem; border: 1px solid #1e293b; border-radius: 12px; background: #111827; }\n" ++
    "    .view.active { display: block; }\n" ++
    "    .summary p { margin: 0.35rem 0; color: #cbd5f5; }\n" ++
    "    .summary ul { list-style: none; padding: 0; margin: 0.5rem 0 0 0; }\n" ++
    "    .summary li { padding: 0.35rem 0; border-bottom: 1px solid #1e293b; font-size: 0.9rem; color: #cbd5f5; }\n" ++
    "    .journal { margin-top: 2rem; }\n" ++
    "    .journal ol { background: #111827; border: 1px solid #1e293b; padding: 1rem 1.5rem; border-radius: 12px; }\n" ++
    "    .journal .stamp { display: inline-block; min-width: 2.5rem; color: #38bdf8; }\n" ++
    "    .journal .prim { color: #cbd5f5; }\n" ++
    "    .import-panel { margin-top: 2.5rem; border: 1px solid #1e293b; padding: 1.25rem; border-radius: 12px; background: #111827; }\n" ++
    "    .import-panel textarea { width: 100%; min-height: 120px; background: #0b1220; color: #e2e8f0; border: 1px solid #1e293b; border-radius: 8px; padding: 0.75rem; font-family: ui-monospace, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }\n" ++
    "    .import-panel .buttons { display: flex; gap: 0.75rem; margin-top: 0.75rem; flex-wrap: wrap; }\n" ++
    "    .import-panel button { padding: 0.5rem 1rem; border: 1px solid #1e293b; background: #1f2937; color: #cbd5f5; border-radius: 8px; cursor: pointer; }\n" ++
    "    .feedback { margin-top: 0.75rem; font-size: 0.9rem; }\n" ++
    "    .feedback.success { color: #34d399; }\n" ++
    "    .feedback.error { color: #f87171; }\n" ++
    "    .feedback.info { color: #60a5fa; }\n" ++
    "  </style>\n" ++
    "</head>\n"
  let bodyIntro :=
    s!"<body>\n  <div class=\"app\">\n    <header>\n      <h1>{statement}</h1>\n      <p>{provenance}</p>\n    </header>\n    <div class=\"view-switcher\">\n      <button data-view=\"visual\" class=\"active\">Visualization</button>\n      <button data-view=\"summary\">Graph summary</button>\n    </div>\n    <div id=\"view-visual\" class=\"view active\">{svg}</div>\n    <div id=\"view-summary\" class=\"view\">\n      <div class=\"summary\" id=\"summary-panel\"></div>\n    </div>\n"
  let journalSection :=
    s!"    <section class=\"journal\">\n      <h2>LoF Journal</h2>\n      <ol>{journalItems}</ol>\n    </section>\n"
  let importSection :=
    "    <section class=\"import-panel\">\n" ++
    "      <h2>Load Custom Proof Graph</h2>\n" ++
    "      <p>Paste a proof-graph JSON payload to explore. Click \"Reset\" to restore the original graph.</p>\n" ++
    "      <textarea id=\"json-input\" placeholder=\"Paste proof graph JSON here...\"></textarea>\n" ++
    "      <div class=\"buttons\">\n" ++
    "        <button id=\"apply-json\" type=\"button\">Apply Graph</button>\n" ++
    "        <button id=\"reset-graph\" type=\"button\">Reset</button>\n" ++
    "      </div>\n" ++
    "      <p id=\"import-feedback\" class=\"feedback info\">Loaded embedded graph.</p>\n" ++
    "    </section>\n" ++
    "  </div>\n"
  let script :=
    "  <script>\n" ++
    s!"  const GRAPH_DATA = {graphJson};\n" ++
    "  const GRAPH_ORIGINAL = JSON.parse(JSON.stringify(GRAPH_DATA));\n\n" ++
    "  function cloneGraph(data) {\n    return data ? JSON.parse(JSON.stringify(data)) : null;\n  }\n\n" ++
    "  function truncate(text, max = 60) {\n    if (typeof text !== 'string') return '';\n    return text.length > max ? text.slice(0, max - 1) + '…' : text;\n  }\n\n" ++
    "  function collectStats(graph) {\n    if (!graph || !Array.isArray(graph.nodes)) {\n      return { totalNodes: 0, totalEdges: 0, primitives: { Mark: 0, Unmark: 0, 'Re-entry': 0 } };\n    }\n    const primitives = { Mark: 0, Unmark: 0, 'Re-entry': 0 };\n    (graph.nodes || []).forEach((node) => {\n      const key = node.primitive;\n      if (key && primitives[key] !== undefined) {\n        primitives[key] += 1;\n      }\n    });\n    return {\n      totalNodes: graph.nodes.length,\n      totalEdges: (graph.edges || []).length,\n      primitives\n    };\n  }\n\n" ++
    "  let currentGraph = cloneGraph(GRAPH_DATA);\n\n" ++
    "  function renderSummary() {\n    const panel = document.getElementById('summary-panel');\n    if (!panel) return;\n    panel.innerHTML = '';\n    if (!currentGraph || !Array.isArray(currentGraph.nodes)) {\n      panel.textContent = 'No graph data available.';\n      return;\n    }\n    const stats = collectStats(currentGraph);\n    const info = document.createElement('p');\n    info.textContent = `Nodes: ${stats.totalNodes} • Edges: ${stats.totalEdges}`;\n    panel.appendChild(info);\n    const primitiveRow = document.createElement('p');\n    primitiveRow.textContent = `Primitive steps – Mark: ${stats.primitives.Mark}, Unmark: ${stats.primitives.Unmark}, Re-entry: ${stats.primitives['Re-entry']}`;\n    panel.appendChild(primitiveRow);\n    const nodeHeader = document.createElement('p');\n    nodeHeader.textContent = 'Node preview:';\n    panel.appendChild(nodeHeader);\n    const nodeList = document.createElement('ul');\n    (currentGraph.nodes || []).slice(0, 12).forEach((node) => {\n      const li = document.createElement('li');\n      li.textContent = `${node.id ?? '?'} • ${node.kind ?? '?'} — ${truncate(node.label || node.kind || '')}`;\n      nodeList.appendChild(li);\n    });\n    panel.appendChild(nodeList);\n    if ((currentGraph.nodes || []).length > 12) {\n      const note = document.createElement('p');\n      note.textContent = 'Showing first 12 nodes.';\n      panel.appendChild(note);\n    }\n    const edgeHeader = document.createElement('p');\n    edgeHeader.textContent = 'Edge preview:';\n    panel.appendChild(edgeHeader);\n    const edgeList = document.createElement('ul');\n    (currentGraph.edges || []).slice(0, 12).forEach((edge) => {\n      const li = document.createElement('li');\n      li.textContent = `${edge.src ?? '?'} → ${edge.dst ?? '?'} (${edge.kind ?? 'edge'}${edge.label ? ' — ' + truncate(edge.label, 40) : ''})`;\n      edgeList.appendChild(li);\n    });\n    panel.appendChild(edgeList);\n    if ((currentGraph.edges || []).length > 12) {\n      const note = document.createElement('p');\n      note.textContent = 'Showing first 12 edges.';\n      panel.appendChild(note);\n    }\n  }\n\n" ++
    "  function setFeedback(message, kind) {\n    const banner = document.getElementById('import-feedback');\n    if (!banner) return;\n    banner.textContent = message;\n    banner.className = `feedback ${kind}`;\n  }\n\n" ++
    "  function setGraph(graph, message, kind) {\n    currentGraph = cloneGraph(graph);\n    renderSummary();\n    setFeedback(message, kind);\n  }\n\n" ++
    "  document.querySelectorAll('[data-view]').forEach((btn) => {\n    btn.addEventListener('click', () => {\n      document.querySelectorAll('[data-view]').forEach((other) => other.classList.remove('active'));\n      document.querySelectorAll('.view').forEach((pane) => pane.classList.remove('active'));\n      btn.classList.add('active');\n      const target = document.getElementById(`view-${btn.dataset.view}`);\n      if (target) target.classList.add('active');\n      if (btn.dataset.view === 'summary') renderSummary();\n    });\n  });\n\n" ++
    "  const applyButton = document.getElementById('apply-json');\n  if (applyButton) {\n    applyButton.addEventListener('click', () => {\n      const input = document.getElementById('json-input');\n      const raw = input ? input.value.trim() : '';\n      if (!raw) {\n        setFeedback('Paste JSON before applying.', 'info');\n        return;\n      }\n      try {\n        const parsed = JSON.parse(raw);\n        setGraph(parsed, 'Loaded graph from pasted JSON.', 'success');\n      } catch (err) {\n        setFeedback('Unable to parse JSON: ' + (err.message || err), 'error');\n      }\n    });\n  }\n\n" ++
    "  const resetButton = document.getElementById('reset-graph');\n  if (resetButton) {\n    resetButton.addEventListener('click', () => setGraph(GRAPH_ORIGINAL, 'Restored original graph.', 'info'));\n  }\n\n" ++
    "  setGraph(GRAPH_ORIGINAL, 'Loaded embedded graph.', 'info');\n  renderSummary();\n  document.getElementById('view-visual').classList.add('active');\n  document.querySelector('[data-view=\\\"visual\\\"]').classList.add('active');\n  </script>\n"
  head ++ bodyIntro ++ journalSection ++ importSection ++ script ++ "</body>\n</html>\n"

def outputPath : System.FilePath :=
  "Nat_mul_comm_lof.html"

#eval show CommandElabM Unit from do
  let bundle ← bundleOfConstant ``Nat.mul_comm
  let doc := htmlDocument bundle
  liftIO <| IO.FS.writeFile outputPath.toString doc
  logInfo m!"Wrote {outputPath}"

end ManualRender
