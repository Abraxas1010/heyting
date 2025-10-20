import React, {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { RpcContext } from "@leanprover/infoview";

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

  const currentHud = response?.render?.hud;
  const currentMode = currentHud?.mode ?? "boundary";
  const currentLens = currentHud?.lens ?? "logic";
  const currentStage = currentHud?.dialStage ?? "s0_ontic";

  const svgMarkup = useMemo(
    () => ({ __html: response?.render?.svg ?? "" }),
    [response]
  );

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

  return React.createElement(
    "div",
    { className: "lof-app" },
    header,
    infoStrip,
    controls,
    renderSurface,
    React.createElement(ProofBadges, { proof: response?.proof }),
    notesPanel
  );
}
