export type VisualMode =
  | "Boundary"
  | "Euler"
  | "Hypergraph"
  | "FiberBundle"
  | "StringDiagram"
  | "Split";

export type Lens = "Logic" | "Tensor" | "Graph" | "Clifford";

export type DialStage = "S0_ontic" | "S1_symbolic" | "S2_circle" | "S3_sphere";

export type LoFPrimitive = "Unmark" | "Mark" | "Reentry";

export type EventKind = "Primitive" | "Dial" | "Lens" | "Mode";

export interface LoFEvent {
  kind: EventKind;
  primitive?: LoFPrimitive;
  dialStage?: DialStage;
  lens?: Lens;
  mode?: VisualMode;
  clientVersion: string;
  sceneId: string;
}

export interface HudPayload {
  dialStage: DialStage;
  lens: Lens;
  mode: VisualMode;
  notes: string[];
}

export interface RenderSummary {
  sceneId: string;
  stage: string;
  lens: string;
  svg: string;
  hud: HudPayload;
}

export interface CertificateBundle {
  adjunction: boolean;
  rt₁: boolean;
  rt₂: boolean;
  classicalized: boolean;
  messages: string[];
}

export interface ApplyResponse {
  render: RenderSummary;
  proof: CertificateBundle;
}
