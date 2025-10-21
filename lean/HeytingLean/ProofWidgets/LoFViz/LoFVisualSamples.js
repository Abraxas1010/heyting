export const SAMPLE_GRAPHS = [
  {
    id: "sample.mul_comm",
    title: "Sample: ℕ multiplication commutativity",
    constant: "Nat.mul_comm",
    description: "Toy LoF trace for Nat.mul_comm (trimmed for the demo).",
    fuel: 128,
    graph: {
      root: 0,
      nodes: [
        { id: 0, kind: "state", depth: 0, label: "state[0] {}", region: [], journalIx: 0, primitive: null, const: null },
        { id: 1, kind: "primitive", depth: 1, label: "0: Mark", region: ["α"], journalIx: 0, primitive: "Mark", const: null },
        { id: 2, kind: "state", depth: 2, label: "state[1] {α}", region: ["α"], journalIx: 1, primitive: null, const: null },
        { id: 3, kind: "primitive", depth: 3, label: "1: Mark", region: ["α", "β"], journalIx: 1, primitive: "Mark", const: null },
        { id: 4, kind: "state", depth: 4, label: "state[2] {α, β}", region: ["α", "β"], journalIx: 2, primitive: null, const: null },
        { id: 5, kind: "primitive", depth: 5, label: "2: Unmark", region: ["α"], journalIx: 2, primitive: "Unmark", const: null },
        { id: 6, kind: "state", depth: 6, label: "state[3] {α}", region: ["α"], journalIx: 3, primitive: null, const: null },
        { id: 7, kind: "primitive", depth: 7, label: "3: Re-entry", region: ["α"], journalIx: 3, primitive: "Re-entry", const: null },
        { id: 8, kind: "state", depth: 8, label: "state[4] {α, δ}", region: ["α", "δ"], journalIx: 4, primitive: null, const: null },
        { id: 9, kind: "term", depth: 1, label: "const Nat.mul_comm", region: [], journalIx: null, primitive: null, const: "Nat.mul_comm" },
        { id: 10, kind: "term", depth: 2, label: "λ n", region: [], journalIx: null, primitive: null, const: null },
        { id: 11, kind: "term", depth: 2, label: "λ m", region: [], journalIx: null, primitive: null, const: null },
        { id: 12, kind: "term", depth: 3, label: "HMul.hMul", region: [], journalIx: null, primitive: null, const: "HMul.hMul" },
        { id: 13, kind: "term", depth: 3, label: "Eq", region: [], journalIx: null, primitive: null, const: "Eq" }
      ],
      edges: [
        { src: 0, dst: 1, kind: "journal", weight: 1, label: "Mark" },
        { src: 1, dst: 2, kind: "journal", weight: 1, label: "state" },
        { src: 2, dst: 3, kind: "journal", weight: 1, label: "Mark" },
        { src: 3, dst: 4, kind: "journal", weight: 1, label: "state" },
        { src: 4, dst: 5, kind: "journal", weight: 1, label: "Unmark" },
        { src: 5, dst: 6, kind: "journal", weight: 1, label: "state" },
        { src: 6, dst: 7, kind: "journal", weight: 1, label: "Re-entry" },
        { src: 7, dst: 8, kind: "journal", weight: 1, label: "state" },
        { src: 9, dst: 10, kind: "dependency", weight: 1, label: "arg" },
        { src: 10, dst: 11, kind: "dependency", weight: 1, label: "body" },
        { src: 11, dst: 12, kind: "dependency", weight: 1, label: "lhs" },
        { src: 11, dst: 13, kind: "dependency", weight: 1, label: "eq" }
      ]
    }
  },
  {
    id: "sample.add_comm",
    title: "Sample: ℕ addition commutativity",
    constant: "Nat.add_comm",
    description: "Toy LoF trace for Nat.add_comm with re-entry loop.",
    fuel: 128,
    graph: {
      root: 0,
      nodes: [
        { id: 0, kind: "state", depth: 0, label: "state[0] {}", region: [], journalIx: 0, primitive: null, const: null },
        { id: 1, kind: "primitive", depth: 1, label: "0: Mark", region: ["α"], journalIx: 0, primitive: "Mark", const: null },
        { id: 2, kind: "state", depth: 2, label: "state[1] {α}", region: ["α"], journalIx: 1, primitive: null, const: null },
        { id: 3, kind: "primitive", depth: 3, label: "1: Re-entry", region: ["α", "β"], journalIx: 1, primitive: "Re-entry", const: null },
        { id: 4, kind: "state", depth: 4, label: "state[2] {α, β, γ}", region: ["α", "β", "γ"], journalIx: 2, primitive: null, const: null },
        { id: 5, kind: "primitive", depth: 5, label: "2: Unmark", region: ["α", "β"], journalIx: 2, primitive: "Unmark", const: null },
        { id: 6, kind: "state", depth: 6, label: "state[3] {β}", region: ["β"], journalIx: 3, primitive: null, const: null },
        { id: 7, kind: "primitive", depth: 7, label: "3: Mark", region: ["β", "δ"], journalIx: 3, primitive: "Mark", const: null },
        { id: 8, kind: "state", depth: 8, label: "state[4] {β, δ}", region: ["β", "δ"], journalIx: 4, primitive: null, const: null },
        { id: 9, kind: "term", depth: 1, label: "const Nat.add_comm", region: [], journalIx: null, primitive: null, const: "Nat.add_comm" },
        { id: 10, kind: "term", depth: 2, label: "λ n", region: [], journalIx: null, primitive: null, const: null },
        { id: 11, kind: "term", depth: 2, label: "λ m", region: [], journalIx: null, primitive: null, const: null },
        { id: 12, kind: "term", depth: 3, label: "HMul.hMul", region: [], journalIx: null, primitive: null, const: "HMul.hMul" },
        { id: 13, kind: "term", depth: 3, label: "Eq", region: [], journalIx: null, primitive: null, const: "Eq" },
        { id: 14, kind: "term", depth: 4, label: "Nat.succ", region: [], journalIx: null, primitive: null, const: "Nat.succ" }
      ],
      edges: [
        { src: 0, dst: 1, kind: "journal", weight: 1, label: "Mark" },
        { src: 1, dst: 2, kind: "journal", weight: 1, label: "state" },
        { src: 2, dst: 3, kind: "journal", weight: 1, label: "Re-entry" },
        { src: 3, dst: 4, kind: "journal", weight: 1, label: "state" },
        { src: 4, dst: 5, kind: "journal", weight: 1, label: "Unmark" },
        { src: 5, dst: 6, kind: "journal", weight: 1, label: "state" },
        { src: 6, dst: 7, kind: "journal", weight: 1, label: "Mark" },
        { src: 7, dst: 8, kind: "journal", weight: 1, label: "state" },
        { src: 9, dst: 10, kind: "dependency", weight: 1, label: "arg" },
        { src: 10, dst: 11, kind: "dependency", weight: 1, label: "body" },
        { src: 11, dst: 12, kind: "dependency", weight: 1, label: "lhs" },
        { src: 11, dst: 13, kind: "dependency", weight: 1, label: "eq" },
        { src: 12, dst: 14, kind: "dependency", weight: 1, label: "succ" }
      ]
    }
  }
];
