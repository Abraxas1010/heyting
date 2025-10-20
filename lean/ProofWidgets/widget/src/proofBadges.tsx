import React from "react";
import type { CertificateBundle } from "./types";

export function ProofBadges({ proof }: { proof: CertificateBundle | undefined }) {
  if (!proof) return null;
  const indicators: Array<{ label: string; ok: boolean }> = [
    { label: "Adjunction", ok: proof.adjunction },
    { label: "RT-1", ok: proof.rt₁ },
    { label: "RT-2", ok: proof.rt₂ },
    { label: "Classical", ok: proof.classicalized },
  ];
  return (
    <aside className="lof-proof-badges">
      <header>Proof status</header>
      <ul>
        {indicators.map((badge) => (
          <li key={badge.label} className={badge.ok ? "ok" : "pending"}>
            <span className="label">{badge.label}</span>
            <span className="dot" />
          </li>
        ))}
      </ul>
      {proof.messages.length > 0 ? (
        <section className="messages">
          {proof.messages.map((msg, idx) => (
            <p key={idx}>{msg}</p>
          ))}
        </section>
      ) : null}
    </aside>
  );
}
