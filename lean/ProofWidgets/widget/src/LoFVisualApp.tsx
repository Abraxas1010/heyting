import React, { useCallback, useContext, useMemo, useState } from "react";
import { RpcContext } from "@leanprover/infoview";
import { ProofBadges } from "./proofBadges";
import type {
  ApplyResponse,
  DialStage,
  Lens,
  LoFEvent,
  LoFPrimitive,
  VisualMode,
} from "./types";

const modes: VisualMode[] = [
  "Boundary",
  "Euler",
  "Hypergraph",
  "FiberBundle",
  "StringDiagram",
  "Split",
];

const lenses: Lens[] = ["Logic", "Tensor", "Graph", "Clifford"];

const dialStages: DialStage[] = ["S0_ontic", "S1_symbolic", "S2_circle", "S3_sphere"];

export function LoFVisualApp(props: { sceneId?: string; clientVersion?: string }) {
  const sceneId = props.sceneId ?? "default";
  const clientVersion = props.clientVersion ?? "0.1.0";
  const rpc = useContext(RpcContext);
  const [response, setResponse] = useState<ApplyResponse | undefined>();
  const [isLoading, setIsLoading] = useState(false);

  const send = useCallback(
    async (event: Omit<LoFEvent, "clientVersion" | "sceneId">) => {
      if (!rpc) return;
      setIsLoading(true);
      const payload: LoFEvent = { ...event, sceneId, clientVersion };
      try {
        const result = (await rpc.call("LoF.apply", payload)) as ApplyResponse;
        setResponse(result);
      } finally {
        setIsLoading(false);
      }
    },
    [clientVersion, rpc, sceneId],
  );

  const currentHud = response?.render.hud;

  const currentMode = currentHud?.mode ?? "Boundary";
  const currentLens = currentHud?.lens ?? "Logic";
  const currentStage = currentHud?.dialStage ?? "S0_ontic";

  const svgMarkup = useMemo(() => ({ __html: response?.render.svg ?? "" }), [response]);

  const triggerPrimitive = (primitive: LoFPrimitive) =>
    send({ kind: "Primitive", primitive });
  const triggerMode = (mode: VisualMode) => send({ kind: "Mode", mode });
  const triggerLens = (lens: Lens) => send({ kind: "Lens", lens });
  const triggerDial = (dialStage: DialStage) => send({ kind: "Dial", dialStage });

  return (
    <div className="lof-app">
      <header>
        <h2>LoF Visualization</h2>
        {isLoading ? <span className="status loading">Updating…</span> : null}
      </header>

      <section className="info-strip">
        <span className="badge">Dial: {currentStage}</span>
        <span className="badge">Lens: {currentLens}</span>
        <span className="badge">Mode: {currentMode}</span>
      </section>

      <section className="controls">
        <div className="primitive-group">
          <button onClick={() => triggerPrimitive("Unmark")}>Unmark</button>
          <button onClick={() => triggerPrimitive("Mark")}>Mark</button>
          <button onClick={() => triggerPrimitive("Reentry")}>Re-entry</button>
        </div>

        <div className="select-group">
          <label>
            Mode
            <select
              value={currentMode}
              onChange={(evt) => triggerMode(evt.target.value as VisualMode)}
            >
              {modes.map((mode) => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
          </label>

          <label>
            Lens
            <select
              value={currentLens}
              onChange={(evt) => triggerLens(evt.target.value as Lens)}
            >
              {lenses.map((lens) => (
                <option key={lens} value={lens}>
                  {lens}
                </option>
              ))}
            </select>
          </label>

          <label>
            Dial
            <select
              value={currentStage}
              onChange={(evt) => triggerDial(evt.target.value as DialStage)}
            >
              {dialStages.map((stage) => (
                <option key={stage} value={stage}>
                  {stage}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section
        className="render-surface"
        dangerouslySetInnerHTML={svgMarkup}
        aria-live="polite"
      />

      <ProofBadges proof={response?.proof} />

      {currentHud?.notes && currentHud.notes.length > 0 ? (
        <section className="hud-panel">
          <header>Notes</header>
          <ol>
            {currentHud.notes.map((note, idx) => (
              <li key={idx}>{note}</li>
            ))}
          </ol>
        </section>
      ) : null}
    </div>
  );
}

export default LoFVisualApp;
