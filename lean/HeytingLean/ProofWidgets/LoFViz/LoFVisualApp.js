import React, {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { RpcContext } from "@leanprover/infoview";
import { SAMPLE_GRAPHS } from "./LoFVisualSamples";

const GRAPH_SCRIPT_ID = "lof-proof-graph";

const MODE_OPTIONS = [
  { value: "boundary", label: "Boundary" },
  { value: "euler", label: "Euler" },
  { value: "hypergraph", label: "Hypergraph" },
  { value: "fiber", label: "Fiber Bundle" },
  { value: "string", label: "String Diagram" },
  { value: "split", label: "Split View" },
];

const LENS_OPTIONS = [
  { value: "logic", label: "Logic" },
  { value: "tensor", label: "Tensor" },
  { value: "graph", label: "Graph" },
  { value: "clifford", label: "Clifford" },
];

const DIAL_OPTIONS = [
  { value: "s0_ontic", label: "S0: Ontic" },
  { value: "s1_symbolic", label: "S1: Symbolic" },
  { value: "s2_circle", label: "S2: Circle" },
  { value: "s3_sphere", label: "S3: Sphere" },
];

const PRIMITIVE_OPTIONS = [
  { value: "unmark", label: "Unmark" },
  { value: "mark", label: "Mark" },
  { value: "reentry", label: "Re-entry" },
];

function labelFor(options, value) {
  const option = options.find((entry) => entry.value === value);
  return option ? option.label : value ?? "—";
}

function truncate(text, max = 80) {
  if (typeof text !== "string") {
    return "";
  }
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function readProofGraphPayload() {
  if (typeof document === "undefined") {
    return null;
  }
  const script = document.getElementById(GRAPH_SCRIPT_ID);
  if (!script) {
    return null;
  }
  const raw = script.textContent ?? script.innerText ?? "";
  if (!raw.trim()) {
    return null;
  }
  return JSON.parse(raw);
}

function cloneGraph(graph) {
  return graph ? JSON.parse(JSON.stringify(graph)) : null;
}

function collectGraphStats(graph) {
  if (!graph) {
    return { totalNodes: 0, totalEdges: 0, primitiveCounts: { Mark: 0, Unmark: 0, "Re-entry": 0 } };
  }
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];
  const primitiveCounts = nodes.reduce(
    (acc, node) => {
      const key = node.primitive ?? null;
      if (key && acc[key] !== undefined) {
        acc[key] += 1;
      }
      return acc;
    },
    { Mark: 0, Unmark: 0, "Re-entry": 0 }
  );
  return {
    totalNodes: nodes.length,
    totalEdges: edges.length,
    primitiveCounts
  };
}

function HypergraphView({ graph, layout }) {
  if (!graph || !Array.isArray(graph.nodes) || graph.nodes.length === 0) {
    return React.createElement("p", { className: "graph-empty" }, "No proof graph available.");
  }
  const layerCount = layout.layers.length || 3;
  const layerWidth = 220;
  const nodeSpacing = 36;
  const width = Math.max(400, 120 + layerCount * layerWidth);
  const maxLayerSize = Math.max(...layout.layers.map((layer) => layer.length), 6);
  const height = Math.max(260, 120 + maxLayerSize * nodeSpacing);
  const xForLayer = (layerIdx) => 120 + layerIdx * layerWidth;
  const positionById = new Map();
  layout.layers.forEach((layer, layerIdx) => {
    layer.slice(0, 80).forEach((node, position) => {
      const y = 90 + position * nodeSpacing;
      positionById.set(node.id, { ...node, x: xForLayer(layerIdx), y });
    });
  });
  const displayedNodes = Array.from(positionById.values());
  const edges = (graph.edges ?? []).filter((edge) => positionById.has(edge.src) && positionById.has(edge.dst)).slice(0, 120);

  return React.createElement(
    "svg",
    { viewBox: `0 0 ${width} ${height}`, className: "hypergraph-view" },
    React.createElement("rect", { x: 10, y: 10, width: width - 20, height: height - 20, rx: 12, className: "hypergraph-bg" }),
    layout.layers.map((_, idx) =>
      React.createElement(
        "text",
        { key: `layer-${idx}`, x: xForLayer(idx), y: 40, className: "hypergraph-column-label" },
        idx === 0 ? "STATE" : idx === 1 ? "PRIMITIVE" : "TERM"
      )
    ),
    edges.map((edge, idx) => {
      const src = positionById.get(edge.src);
      const dst = positionById.get(edge.dst);
      const stroke = edge.kind === "dependency" ? "#f97316" : "#38bdf8";
      return React.createElement("line", {
        key: `${edge.src}-${edge.dst}-${idx}`,
        x1: src.x,
        y1: src.y,
        x2: dst.x,
        y2: dst.y,
        stroke,
        "stroke-width": 2,
        "stroke-linecap": "round",
        opacity: 0.8
      });
    }),
    displayedNodes.map((node) =>
      React.createElement(
        "g",
        { key: node.id },
        React.createElement("circle", {
          cx: node.x,
          cy: node.y,
          r: node.kind === "primitive" ? 12 : 10,
          className: `hypergraph-node hypergraph-${node.kind ?? "term"}`
        }),
        React.createElement(
          "text",
          { x: node.x + 16, y: node.y + 4, className: "hypergraph-node-label" },
          `${node.id}: ${truncate(node.label ?? node.kind ?? "", 48)}`
        )
      )
    )
  );
}

function computeDagLayout(graph) {
  if (!graph || !Array.isArray(graph.nodes)) {
    return { layers: [], ranks: new Map() };
  }
  const nodesByKind = new Map();
  for (const node of graph.nodes) {
    const key = node.kind ?? "term";
    if (!nodesByKind.has(key)) {
      nodesByKind.set(key, []);
    }
    nodesByKind.get(key).push(node);
  }
  const layers = [
    nodesByKind.get("state") ?? [],
    nodesByKind.get("primitive") ?? [],
    nodesByKind.get("term") ?? []
  ];
  const ranks = new Map();
  layers.forEach((layer, layerIdx) => {
    layer.forEach((node, position) => {
      ranks.set(node.id, { layer: layerIdx, position });
    });
  });
  return { layers, ranks };
}

function computeDagLayout(graph) {
  if (!graph || !Array.isArray(graph.nodes)) {
    return { layers: [], ranks: new Map() };
  }
  const nodesByKind = new Map();
  for (const node of graph.nodes) {
    const key = node.kind ?? "term";
    if (!nodesByKind.has(key)) {
      nodesByKind.set(key, []);
    }
    nodesByKind.get(key).push(node);
  }
  const layers = [
    nodesByKind.get("state") ?? [],
    nodesByKind.get("primitive") ?? [],
    nodesByKind.get("term") ?? []
  ];
  const ranks = new Map();
  layers.forEach((layer, idx) => {
    layer.forEach((node, position) => {
      ranks.set(node.id, { layer: idx, position });
    });
  });
  return { layers, ranks };
}

function EulerTensorView({ graph, stats }) {
  const summary = stats ?? collectGraphStats(graph);
  const entries = [
    { label: "Mark", value: summary.primitiveCounts.Mark, color: "#38bdf8" },
    { label: "Unmark", value: summary.primitiveCounts.Unmark, color: "#f97316" },
    { label: "Re-entry", value: summary.primitiveCounts["Re-entry"], color: "#a855f7" }
  ];
  const total = entries.reduce((acc, item) => acc + item.value, 0) || 1;

  return React.createElement(
    "div",
    { className: "euler-tensor-view" },
    React.createElement(
      "header",
      null,
      `Primitive balance (${total} steps)`
    ),
    React.createElement(
      "div",
      { className: "euler-bars" },
      entries.map((item) =>
        React.createElement(
          "div",
          { key: item.label, className: "euler-bar" },
          React.createElement("span", { className: "legend-dot", style: { backgroundColor: item.color } }),
          React.createElement("span", { className: "label" }, `${item.label}`),
          React.createElement(
            "div",
            { className: "bar-fill" },
            React.createElement("span", {
              className: "bar-progress",
              style: { width: `${(item.value / total) * 100}%`, backgroundColor: item.color }
            })
          ),
          React.createElement("span", { className: "value" }, item.value)
        )
      )
    ),
    React.createElement(
      "p",
      { className: "euler-summary" },
      "The breathing cycle stays constructive when Mark and Unmark remain balanced. Re-entry spikes highlight modal loops."
    )
  );
}

function CausalDependencyView({ graph }) {
  if (!graph || !Array.isArray(graph.edges)) {
    return React.createElement("p", { className: "graph-empty" }, "No dependency edges recorded.");
  }
  const dependencies = graph.edges.filter((edge) => edge.kind === "dependency");
  const grouped = dependencies.reduce((acc, edge) => {
    const key = `${edge.src}`;
    if (!acc.has(key)) {
      acc.set(key, []);
    }
    acc.get(key).push(edge);
    return acc;
  }, new Map());

  const topEntries = Array.from(grouped.entries())
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 12);

  return React.createElement(
    "div",
    { className: "causal-view" },
    topEntries.length === 0
      ? React.createElement("p", null, "No dependency edges detected.")
      : topEntries.map(([src, edges]) =>
          React.createElement(
            "div",
            { key: src, className: "causal-entry" },
            React.createElement("header", null, `Node ${src}`),
            React.createElement(
              "ol",
              null,
              edges.map((edge, idx) =>
                React.createElement(
                  "li",
                  { key: `${edge.dst}-${idx}` },
                  `→ ${edge.dst} (${edge.label ?? edge.kind})`
                )
              )
            )
          )
        )
  );
}

function SampleProofGallery({ onSelect }) {
  return React.createElement(
    "div",
    { className: "sample-gallery" },
    React.createElement("header", null, "Sample proofs"),
    React.createElement(
      "div",
      { className: "sample-grid" },
      SAMPLE_GRAPHS.map((sample) =>
        React.createElement(
          "button",
          {
            key: sample.id,
            type: "button",
            onClick: () => onSelect(sample)
          },
          React.createElement("strong", null, sample.title),
          React.createElement("span", { className: "sample-constant" }, sample.constant),
          React.createElement("span", { className: "sample-description" }, sample.description)
        )
      )
    )
  );
}

function UploadOrPastePanel({ onImport, onError, onFetchConstant }) {
  const [text, setText] = useState("");
  const [constant, setConstant] = useState("");

  const handleFile = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));
        onImport(cloneGraph(data), { title: file.name, constant: "uploaded" });
      } catch (err) {
        onError(err instanceof Error ? err.message : "Unable to parse JSON file.");
      }
    };
    reader.onerror = () => onError("Unable to read selected file.");
    reader.readAsText(file);
  };

  const handlePaste = () => {
    if (!text.trim()) {
      onError("Paste JSON into the text area before importing.");
      return;
    }
    try {
      const data = JSON.parse(text);
      onImport(cloneGraph(data), { title: "Pasted proof graph", constant: "pasted" });
      setText("");
    } catch (err) {
      onError(err instanceof Error ? err.message : "Unable to parse pasted JSON.");
    }
  };

  const handleFetch = () => {
    if (!onFetchConstant) return;
    const trimmed = constant.trim();
    if (!trimmed) {
      onError("Enter a constant name to fetch.");
      return;
    }
    onFetchConstant(trimmed);
  };

  return React.createElement(
    "div",
    { className: "upload-panel" },
    React.createElement("header", null, "Bring your own proof graph"),
    React.createElement(
      "div",
      { className: "fetch-row" },
      React.createElement("input", {
        type: "text",
        value: constant,
        onChange: (evt) => setConstant(evt.target.value),
        placeholder: "Constant name (e.g. Nat.mul_comm)",
      }),
      React.createElement(
        "button",
        { type: "button", onClick: handleFetch },
        "Fetch from Lean"
      )
    ),
    React.createElement(
      "div",
      { className: "upload-row" },
      React.createElement("label", null, "Upload JSON", React.createElement("input", { type: "file", accept: ".json,application/json", onChange: handleFile }))
    ),
    React.createElement(
      "div",
      { className: "paste-row" },
      React.createElement("textarea", {
        value: text,
        onChange: (evt) => setText(evt.target.value),
        placeholder: 'Paste a proof graph as JSON (fields: "nodes", "edges", ...)',
        rows: 4
      }),
      React.createElement(
        "button",
        { type: "button", onClick: handlePaste },
        "Import pasted JSON"
      )
    )
  );
}

function ProofBadges({ proof }) {
  if (!proof) {
    return null;
  }
  const indicators = [
    { label: "Adjunction", ok: !!proof.adjunction },
    { label: "RT-1", ok: !!proof["rt₁"] },
    { label: "RT-2", ok: !!proof["rt₂"] },
    { label: "Classical", ok: !!proof.classicalized },
  ];
  return React.createElement(
    "aside",
    { className: "lof-proof-badges" },
    React.createElement("header", null, "Proof status"),
    React.createElement(
      "ul",
      null,
      indicators.map((indicator) =>
        React.createElement(
          "li",
          {
            key: indicator.label,
            className: indicator.ok ? "ok" : "pending",
          },
          React.createElement("span", { className: "label" }, indicator.label),
          React.createElement("span", { className: "dot" })
        )
      )
    ),
    proof.messages && proof.messages.length > 0
      ? React.createElement(
          "section",
          { className: "messages" },
          proof.messages.map((msg, idx) =>
            React.createElement("p", { key: idx }, msg)
          )
        )
      : null
  );
}

export default function LoFVisualApp(initialProps = {}) {
  const sceneId = initialProps.sceneId ?? "demo";
  const clientVersion = initialProps.clientVersion ?? "0.1.0";
  const rpc = useContext(RpcContext);
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeView, setActiveView] = useState("visual");
  const [graph, setGraph] = useState(null);
  const [graphError, setGraphError] = useState(null);
  const [graphSource, setGraphSource] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [feedbackKind, setFeedbackKind] = useState("info");
  const [graphLoading, setGraphLoading] = useState(false);
  const [layout, setLayout] = useState({ layers: [], ranks: new Map() });

  const send = useCallback(
    async (event) => {
      if (!rpc) return;
      setIsLoading(true);
      const payload = { ...event, sceneId, clientVersion };
      try {
        const result = await rpc.call(
          "HeytingLean.ProofWidgets.LoFViz.apply",
          payload
        );
        setResponse(result);
      } catch (err) {
        console.error("LoFVisualApp RPC error", err);
      } finally {
        setIsLoading(false);
      }
    },
    [clientVersion, rpc, sceneId]
  );

  useEffect(() => {
    // Load the initial scene on mount.
    send({ kind: "mode", mode: "boundary" });
  }, [send]);

  useEffect(() => {
    try {
      const payload = readProofGraphPayload();
      if (payload) {
        setGraph(payload);
        setGraphSource({ title: "Embedded proof graph", constant: "current" });
      }
      setGraphError(null);
    } catch (err) {
      console.warn("LoFVisualApp graph parse error", err);
      setGraph(null);
      setGraphError(
        err instanceof Error ? err.message : "Unknown error while parsing graph payload."
      );
    }
  }, []);

  const currentHud = response?.render?.hud;
  const currentMode = currentHud?.mode ?? "boundary";
  const currentLens = currentHud?.lens ?? "logic";
  const currentStage = currentHud?.dialStage ?? "s0_ontic";

  const svgMarkup = useMemo(
    () => ({ __html: response?.render?.svg ?? "" }),
    [response]
  );

  const graphStats = useMemo(() => collectGraphStats(graph), [graph]);

  useEffect(() => {
    setLayout(computeDagLayout(graph));
  }, [graph]);

  const triggerPrimitive = useCallback(
    (primitive) => send({ kind: "primitive", primitive }),
    [send]
  );
  const triggerMode = useCallback(
    (mode) => send({ kind: "mode", mode }),
    [send]
  );
  const triggerLens = useCallback(
    (lens) => send({ kind: "lens", lens }),
    [send]
  );
  const triggerDial = useCallback(
    (dialStage) => send({ kind: "dial", dialStage }),
    [send]
  );

  const loadGraphFromServer = useCallback(
    async (constant, meta, fuel) => {
      if (!rpc) {
        setFeedback("Lean RPC session unavailable.");
        setFeedbackKind("error");
        return false;
      }
      setGraphLoading(true);
      try {
        const payload = await rpc.call(
          "HeytingLean.ProofWidgets.LoFViz.graphOfConstant",
          { constant, fuel }
        );
        setGraph(cloneGraph(payload.graph));
        setGraphSource(meta ?? { title: constant, constant });
        setGraphError(null);
        setActiveView("graph");
        setFeedback(`Loaded proof graph for ${constant}.`);
        setFeedbackKind("success");
        return true;
      } catch (err) {
        console.error("LoFVisualApp graph RPC error", err);
        const message =
          err instanceof Error
            ? err.message
            : "Failed to load proof graph from Lean.";
        setFeedback(message);
        setFeedbackKind("error");
        return false;
      } finally {
        setGraphLoading(false);
      }
    },
    [rpc]
  );

  const header = React.createElement(
    "header",
    null,
    React.createElement("h2", null, "LoF Visualization"),
    isLoading
      ? React.createElement(
          "span",
          { className: "status loading" },
          "Updating…"
        )
      : null
  );

  const infoStrip = React.createElement(
    "section",
    { className: "info-strip" },
    React.createElement(
      "span",
      { className: "badge" },
      `Dial: ${labelFor(DIAL_OPTIONS, currentStage)}`
    ),
    React.createElement(
      "span",
      { className: "badge" },
      `Lens: ${labelFor(LENS_OPTIONS, currentLens)}`
    ),
    React.createElement(
      "span",
      { className: "badge" },
      `Mode: ${labelFor(MODE_OPTIONS, currentMode)}`
    )
  );

  const primitiveButtons = React.createElement(
    "div",
    { className: "primitive-group" },
    PRIMITIVE_OPTIONS.map((option) =>
      React.createElement(
        "button",
        {
          key: option.value,
          onClick: () => triggerPrimitive(option.value),
        },
        option.label
      )
    )
  );

  const controls = React.createElement(
    "section",
    { className: "controls" },
    primitiveButtons,
    React.createElement(
      "div",
      { className: "select-group" },
      React.createElement(
        "label",
        null,
        "Mode",
        React.createElement(
          "select",
          {
            value: currentMode,
            onChange: (evt) => triggerMode(evt.target.value),
          },
          MODE_OPTIONS.map((option) =>
            React.createElement(
              "option",
              { key: option.value, value: option.value },
              option.label
            )
          )
        )
      ),
      React.createElement(
        "label",
        null,
        "Lens",
        React.createElement(
          "select",
          {
            value: currentLens,
            onChange: (evt) => triggerLens(evt.target.value),
          },
          LENS_OPTIONS.map((option) =>
            React.createElement(
              "option",
              { key: option.value, value: option.value },
              option.label
            )
          )
        )
      ),
      React.createElement(
        "label",
        null,
        "Dial",
        React.createElement(
          "select",
          {
            value: currentStage,
            onChange: (evt) => triggerDial(evt.target.value),
          },
          DIAL_OPTIONS.map((option) =>
            React.createElement(
              "option",
              { key: option.value, value: option.value },
              option.label
            )
          )
        )
      )
    )
  );

  const renderSurface = React.createElement("section", {
    className: "render-surface",
    dangerouslySetInnerHTML: svgMarkup,
    "aria-live": "polite",
  });

  const graphChildren = [];
  if (graphError) {
    graphChildren.push(
      React.createElement(
        "p",
        { key: "graph-error", className: "graph-error" },
        `Unable to parse proof graph payload: ${graphError}`
      )
    );
  } else if (graph) {
    const totalNodes = Array.isArray(graph.nodes) ? graph.nodes.length : 0;
    const totalEdges = Array.isArray(graph.edges) ? graph.edges.length : 0;
    const previewNodes = (graph.nodes ?? []).slice(0, 12);
    const previewEdges = (graph.edges ?? []).slice(0, 12);
    const countsByKind = previewNodes.reduce((acc, node) => {
      const kind = node.kind ?? "unknown";
      acc[kind] = (acc[kind] ?? 0) + 1;
      return acc;
    }, {});

    graphChildren.push(
      React.createElement(
        "div",
        { key: "graph-stats", className: "graph-stats" },
        `Nodes: ${totalNodes} • Edges: ${totalEdges}${
          typeof graph.root === "number" ? ` • Root: ${graph.root}` : ""
        }`
      )
    );

    graphChildren.push(
      React.createElement(
        "p",
        { key: "graph-kind-stats", className: "graph-kind-stats" },
        Object.entries(countsByKind)
          .map(([kind, count]) => `${kind}: ${count}`)
          .join(" • ") || "Node preview unavailable."
      )
    );

    graphChildren.push(
      React.createElement(
        "section",
        { key: "graph-nodes", className: "graph-section" },
        React.createElement("h4", null, "Node Preview"),
        React.createElement(
          "ol",
          { className: "graph-list" },
          ...previewNodes.map((node, idx) =>
            React.createElement(
              "li",
              { key: node.id ?? `node-${idx}` },
              `${node.id ?? "?"} • ${node.kind ?? "?"}${
                node.label ? ` — ${truncate(node.label, 60)}` : ""
              }`
            )
          )
        ),
        totalNodes > previewNodes.length
          ? React.createElement(
              "p",
              { className: "graph-footnote" },
              `Showing first ${previewNodes.length} nodes.`
            )
          : null
      )
    );

    graphChildren.push(
      React.createElement(
        "section",
        { key: "graph-edges", className: "graph-section" },
        React.createElement("h4", null, "Edge Preview"),
        React.createElement(
          "ol",
          { className: "graph-list" },
          ...previewEdges.map((edge, idx) =>
            React.createElement(
              "li",
              { key: `${edge.src}-${edge.dst}-${idx}` },
              `${edge.src ?? "?"} → ${edge.dst ?? "?"} (${edge.kind ?? "?"}${
                edge.label ? ` — ${truncate(edge.label, 50)}` : ""
              })`
            )
          )
        ),
        totalEdges > previewEdges.length
          ? React.createElement(
              "p",
              { className: "graph-footnote" },
              `Showing first ${previewEdges.length} edges.`
            )
          : null
      )
    );
  } else {
    graphChildren.push(
      React.createElement(
        "p",
        { key: "graph-empty", className: "graph-empty" },
        "Proof graph payload not found. Re-run the visualization command to embed the graph JSON."
      )
    );
  }

  const graphPanel = React.createElement(
    "section",
    { className: "graph-panel" },
    React.createElement(
      "header",
      null,
      graphSource ? `Graph: ${graphSource.title}` : "Proof graph preview"
    ),
    React.createElement(
      "p",
      { className: "graph-stats" },
      `Nodes: ${graphStats.totalNodes} • Edges: ${graphStats.totalEdges}`
    ),
    ...graphChildren
  );

  const notesPanel =
    currentHud && Array.isArray(currentHud.notes) && currentHud.notes.length > 0
      ? React.createElement(
          "section",
          { className: "hud-panel" },
          React.createElement("header", null, "Notes"),
          React.createElement(
            "ol",
            null,
            currentHud.notes.map((note, idx) =>
              React.createElement("li", { key: idx }, note)
            )
          )
        )
      : null;

  const viewSwitcher = React.createElement(
    "section",
    { className: "view-switcher" },
    [
      { key: "visual", label: "Visualization" },
      { key: "graph", label: "Proof Graph" },
      { key: "hyper", label: "Hypergraph" },
      { key: "euler", label: "Euler/Tensor" },
      { key: "causal", label: "Causal" },
    ].map((entry) =>
      React.createElement(
        "button",
        {
          key: entry.key,
          type: "button",
          className: activeView === entry.key ? "active" : "",
          onClick: () => setActiveView(entry.key),
        },
        entry.label
      )
    )
  );

  const mainContent =
    activeView === "visual"
      ? renderSurface
      : activeView === "graph"
      ? graphPanel
      : activeView === "hyper"
      ? React.createElement(HypergraphView, { graph, layout })
      : activeView === "euler"
      ? React.createElement(EulerTensorView, { graph, stats: graphStats })
      : React.createElement(CausalDependencyView, { graph });

  const handleSampleSelect = async (sample) => {
    setFeedback(null);
    setGraphError(null);
    if (sample.constant) {
      const ok = await loadGraphFromServer(
        sample.constant,
        { title: sample.title, constant: sample.constant },
        sample.fuel
      );
      if (ok) {
        return;
      }
    }
    if (sample.graph) {
      setGraph(cloneGraph(sample.graph));
      setGraphSource({ title: sample.title, constant: sample.constant ?? "sample" });
      setGraphError(null);
      setActiveView("graph");
      setFeedback(`Loaded sample graph: ${sample.title}`);
      setFeedbackKind("success");
    } else {
      setFeedback("Sample graph unavailable.");
      setFeedbackKind("warning");
    }
  };

  const handleImport = (importedGraph, meta) => {
    if (!importedGraph) {
      setFeedback("The imported graph was empty.");
      setFeedbackKind("warning");
      return;
    }
    setGraph(importedGraph);
    setGraphSource(meta);
    setGraphError(null);
    setActiveView("graph");
    setFeedback(`Imported proof graph (${meta?.title ?? "custom"})`);
    setFeedbackKind("success");
  };

  const handleGraphError = (message) => {
    setGraphError(message);
    setFeedback(message);
    setFeedbackKind("error");
  };

  const handleFetchConstant = async (constant) => {
    const trimmed = constant.trim();
    if (!trimmed) {
      setFeedback("Enter a constant name to fetch.");
      setFeedbackKind("warning");
      return;
    }
    setGraphError(null);
    await loadGraphFromServer(trimmed, { title: trimmed, constant: trimmed });
  };

  return React.createElement(
    "div",
    { className: "lof-app" },
    header,
    infoStrip,
    controls,
    React.createElement(
      "section",
      { className: "proof-loader" },
      React.createElement(SampleProofGallery, { onSelect: handleSampleSelect }),
      React.createElement(UploadOrPastePanel, {
        onImport: handleImport,
        onError: handleGraphError,
        onFetchConstant: handleFetchConstant,
      })
    ),
    graphLoading
      ? React.createElement("div", { className: "graph-loading" }, "Loading proof graph…")
      : null,
    feedback
      ? React.createElement("div", { className: `feedback feedback-${feedbackKind}` }, feedback)
      : null,
    graphError
      ? React.createElement("div", { className: "graph-error-banner" }, graphError)
      : null,
    viewSwitcher,
    mainContent,
    React.createElement(ProofBadges, { proof: response?.proof }),
    notesPanel
  );
}
