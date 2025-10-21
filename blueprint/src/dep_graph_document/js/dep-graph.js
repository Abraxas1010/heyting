(function () {
  if (!window.depGraphData) {
    console.error("Dependency graph data not found.");
    return;
  }

  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  const theme = getComputedStyle(document.documentElement);
  const defaultFg = (theme.getPropertyValue("--fg") || "#1f2933").trim();

  const width = 1000;
  const height = 720;
  const data = window.depGraphData;

  const nodes = data.nodes.map((d) => ({
    ...d,
    fx: undefined,
    fy: undefined,
  }));
  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  const links = data.edges.map((edge) => ({
    source: edge.source,
    target: edge.target,
  }));

  const statusStyles = {
    "statement-ready": {
      fill: "#EFF6FF",
      stroke: "#3B82F6",
      text: "#1E3A8A",
    },
    "needs-work": {
      fill: "#E2E8F0",
      stroke: "#94A3B8",
      text: defaultFg,
    },
    "out-of-sync": {
      fill: "#FEF3C7",
      stroke: "#F97316",
      text: "#7C2D12",
    },
    formalized: {
      fill: "#DCFCE7",
      stroke: "#16A34A",
      text: "#065F46",
    },
    proofready: {
      fill: "#DBEAFE",
      stroke: "#1D4ED8",
      text: "#1E3A8A",
    },
    mathlib: {
      fill: "#065F46",
      stroke: "#064E3B",
      text: "#ECFDF5",
    },
    default: {
      fill: "#E2E8F0",
      stroke: "#64748B",
      text: defaultFg,
    },
  };

  const svg = d3
    .select("#graph")
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("aria-hidden", "true");

  const zoomLayer = svg.append("g");

  const defs = svg.append("defs");
  defs
    .append("marker")
    .attr("id", "arrowhead")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 22)
    .attr("refY", 0)
    .attr("markerWidth", 8)
    .attr("markerHeight", 8)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,-5L10,0L0,5")
    .attr("fill", "#94A3B8")
    .attr("opacity", 0.8);

  const linkSelection = zoomLayer
    .append("g")
    .attr("stroke", "#94A3B8")
    .attr("stroke-opacity", 0.35)
    .attr("stroke-width", 1.4)
    .attr("marker-end", "url(#arrowhead)")
    .selectAll("line")
    .data(links)
    .join("line");

  const nodeSelection = zoomLayer
    .append("g")
    .attr("stroke-width", 1.2)
    .selectAll("g")
    .data(nodes)
    .join("g")
    .style("cursor", "pointer")
    .call(
      d3
        .drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded)
    );

  nodeSelection.each(function (d) {
    const group = d3.select(this);

    const label = group
      .append("text")
      .attr("class", "node-label")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .style("font-size", "12px")
      .style("font-weight", 600)
      .style("pointer-events", "none")
      .text(d.name);

    const bbox = label.node().getBBox();
    const paddingX = 20;
    const paddingY = 10;
    const width = Math.max(60, bbox.width + paddingX * 2);
    const height = Math.max(32, bbox.height + paddingY * 2);
    d.nodeWidth = width;
    d.nodeHeight = height;

    const style = statusStyles[d.status] || statusStyles.default;
    label.attr("fill", style.text || defaultFg);

    if (d.shape === "box") {
      group
        .insert("rect", "text")
        .attr("class", "node-shape node-shape--box")
        .attr("x", -width / 2)
        .attr("y", -height / 2)
        .attr("width", width)
        .attr("height", height)
        .attr("rx", 6)
        .attr("ry", 6)
        .attr("fill", style.fill)
        .attr("stroke", style.stroke);
    } else {
      group
        .insert("ellipse", "text")
        .attr("class", "node-shape node-shape--ellipse")
        .attr("rx", width / 2)
        .attr("ry", height / 2)
        .attr("fill", style.fill)
        .attr("stroke", style.stroke);
    }
  });

  let userHasInteracted = false;

  const zoom = d3
    .zoom()
    .scaleExtent([0.25, 2.6])
    .on("zoom", (event) => {
      if (event.sourceEvent && event.sourceEvent.type !== "end") {
        userHasInteracted = true;
      }
      zoomLayer.attr("transform", event.transform);
    });

  svg.call(zoom).on("dblclick.zoom", null);

  const popup = document.createElement("div");
  popup.className = "node-popup hidden";
  document.body.appendChild(popup);
  popup.addEventListener("click", (event) => event.stopPropagation());

  nodeSelection
    .on("click", function (event, node) {
      event.stopPropagation();
      focusNode(node);
      showPopup(node, event);
    })
    .on("mouseenter", function (_event, node) {
      highlightNode(node);
    })
    .on("mouseleave", function () {
      highlightNode(activeNode);
    });

  svg.on("click", () => {
    activeNode = null;
    focusNode(null);
    highlightNode(null);
    hidePopup();
  });

  document.addEventListener(
    "click",
    (event) => {
      if (!popup.contains(event.target)) {
        hidePopup();
      }
    },
    { capture: true }
  );

  const simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance(130)
        .strength(0.6)
    )
    .force("charge", d3.forceManyBody().strength(-420))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force(
      "collision",
      d3.forceCollide().radius((d) => (d.isPlaceholder ? 22 : 28))
    )
    .on("tick", ticked)
    .on("end", fitView);

  if (prefersReducedMotion) {
    simulation.alpha(0).alphaMin(0);
  }

  function ticked() {
    linkSelection
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    nodeSelection.attr(
      "transform",
      (d) => `translate(${d.x ?? 0}, ${d.y ?? 0})`
    );
  }

  function fitView() {
    if (userHasInteracted) return;

    const validNodes = nodes.filter(
      (node) => Number.isFinite(node.x) && Number.isFinite(node.y)
    );
    if (!validNodes.length) return;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    validNodes.forEach((node) => {
      minX = Math.min(minX, node.x);
      maxX = Math.max(maxX, node.x);
      minY = Math.min(minY, node.y);
      maxY = Math.max(maxY, node.y);
    });

    const dx = maxX - minX;
    const dy = maxY - minY;
    const padding = 80;
    if (dx === 0 || dy === 0) {
      svg.call(
        zoom.transform,
        d3.zoomIdentity.translate(width / 2 - minX, height / 2 - minY).scale(1)
      );
      return;
    }

    const scale = Math.min(
      (width - padding) / dx,
      (height - padding) / dy,
      2.2
    );
    const translateX = width / 2 - scale * ((minX + maxX) / 2);
    const translateY = height / 2 - scale * ((minY + maxY) / 2);

    svg
      .transition()
      .duration(prefersReducedMotion ? 0 : 650)
      .call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
  }

  function dragStarted(event) {
    hidePopup();
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragEnded(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }

  const detailsPane = document.getElementById("details");
  let activeNode = null;

  function focusNode(node) {
    activeNode = node;
    if (!node) {
      detailsPane.innerHTML =
        "<h2>Details</h2><p>Select a node to inspect its Lean declaration and dependencies.</p>";
      return;
    }

    const { incoming, outgoing } = computeAdjacency(node);

    detailsPane.innerHTML = `
      <h2>${node.name}</h2>
      <p class="node-summary">${formatSummary(node.summary)}</p>
      <dl>
        <dt>Kind</dt>
        <dd>${node.kind}</dd>
        <dt>Lean</dt>
        <dd>${renderLeanInfo(node)}</dd>
        <dt>Documentation</dt>
        <dd>${
          node.url
            ? `<a class="node-link" href="${node.url}" target="_blank" rel="noopener">${node.url}</a>`
            : "—"
        }</dd>
        <dt>Depends on</dt>
        <dd>${renderList(incoming)}</dd>
        <dt>Used by</dt>
        <dd>${renderList(outgoing)}</dd>
      </dl>
      ${renderLeanDoc(node)}
    `;
  }

  function showPopup(node, event) {
    const pointer = d3.pointer(event, document.body);
    const { incoming, outgoing } = computeAdjacency(node);

    popup.innerHTML = `
      <button class="node-popup__close" type="button" aria-label="Close panel">×</button>
      <h3>${node.name}</h3>
      <p class="node-summary">${formatSummary(node.summary)}</p>
      <dl>
        <dt>Kind</dt>
        <dd>${node.kind}</dd>
        <dt>Lean</dt>
        <dd>${renderLeanInfo(node)}</dd>
      </dl>
      <section>
        <h4>Depends on</h4>
        ${renderList(incoming)}
      </section>
      <section>
        <h4>Used by</h4>
        ${renderList(outgoing)}
      </section>
      ${renderLeanDoc(node)}
    `;

    popup
      .querySelector(".node-popup__close")
      .addEventListener("click", hidePopup, { once: true });

    popup.style.left = `${pointer[0] + 18}px`;
    popup.style.top = `${pointer[1] + 16}px`;
    popup.classList.remove("hidden");
  }

  function hidePopup() {
    popup.classList.add("hidden");
  }

  function computeAdjacency(node) {
    const incoming = [];
    const outgoing = [];

    links.forEach((edge) => {
      const sourceId =
        typeof edge.source === "object" ? edge.source.id : edge.source;
      const targetId =
        typeof edge.target === "object" ? edge.target.id : edge.target;
      if (targetId === node.id) {
        incoming.push(sourceId);
      }
      if (sourceId === node.id) {
        outgoing.push(targetId);
      }
    });

    return { incoming, outgoing };
  }

  function renderList(ids) {
    if (!ids.length) {
      return "<em>None</em>";
    }
    const items = ids
      .map((id) => {
        const node = nodeById.get(id);
        if (!node) return `<li>${escapeHtml(id)}</li>`;
        const text = escapeHtml(node.name || node.id);
        if (node.url) {
          return `<li><a class="node-link" href="${node.url}" target="_blank" rel="noopener">${text}</a></li>`;
        }
        return `<li>${text}</li>`;
      })
      .join("");
    return `<ul>${items}</ul>`;
  }

  function renderLeanInfo(node) {
    if (!node.lean) {
      return "—";
    }
    const leanId = `<code>${escapeHtml(node.lean)}</code>`;
    if (!node.leanFound) {
      return `${leanId} <span class="lean-status lean-status--missing">(missing in build)</span>`;
    }
    return `${leanId} <span class="lean-status lean-status--ok">(verified)</span>`;
  }

  function renderLeanDoc(node) {
    if (!node.leanDoc || !node.leanFound) {
      return "";
    }
    return `<section class="lean-doc"><h4>Lean Doc</h4>${formatLeanDoc(node.leanDoc)}</section>`;
  }

  function formatLeanDoc(doc) {
    return escapeHtml(doc)
      .split("\n")
      .map((line) => `<span>${line}</span>`)
      .join("<br>");
  }

  function highlightNode(node) {
    const highlight = new Set();
    if (node) {
      highlight.add(node.id);
      links.forEach((edge) => {
        const sourceId =
          typeof edge.source === "object" ? edge.source.id : edge.source;
        const targetId =
          typeof edge.target === "object" ? edge.target.id : edge.target;
        if (sourceId === node.id || targetId === node.id) {
          highlight.add(sourceId);
          highlight.add(targetId);
        }
      });
    }

    nodeSelection.each(function (d) {
      const isActive = !node || highlight.has(d.id);
      d3.select(this)
        .select(".node-shape")
        .attr("opacity", isActive ? 1 : 0.18);
      d3.select(this)
        .select(".node-label")
        .attr("opacity", isActive ? 1 : 0.35);
    });

    linkSelection.attr("stroke-opacity", (d) => {
      if (!node) return 0.35;
      const sourceId =
        typeof d.source === "object" ? d.source.id : d.source;
      const targetId =
        typeof d.target === "object" ? d.target.id : d.target;
      return sourceId === node.id || targetId === node.id ? 0.7 : 0.05;
    });
  }

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatSummary(summary) {
    if (!summary) {
      return "<em>No blueprint narrative for this node yet.</em>";
    }
    let text = escapeHtml(summary);
    text = text.replace(/\\n+/g, "<br>");
    text = text.replace(/\s*-\s*/g, "<br>- ");
    text = text.replace(/^<br>/, "");
    text = text.replace(/<br><br>/g, "<br>");
    return text;
  }

  const firstRealNode = nodes.find((n) => !n.isPlaceholder);
  if (firstRealNode) {
    focusNode(firstRealNode);
    highlightNode(firstRealNode);
  }
})();
