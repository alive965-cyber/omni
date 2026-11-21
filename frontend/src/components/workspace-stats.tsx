"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Icon } from "@/components/shell";
import type { SessionRecord } from "@/components/session-list";

type DocumentRecord = { id: string; status?: string; evidence_tier?: string };

export function WorkspaceStats() {
  const [sessions, setSessions] = useState<string>("—");
  const [active, setActive] = useState<string>("—");
  const [completed, setCompleted] = useState<string>("—");
  const [documents, setDocuments] = useState<string>("—");
  const [tierBreakdown, setTierBreakdown] = useState<string>("3-Tier Stratified Corpus");

  useEffect(() => {
    api
      .get<SessionRecord[]>("/sessions")
      .then((rows) => {
        setSessions(String(rows.length));
        setActive(
          String(
            rows.filter((row) =>
              ["CREATED", "CALIBRATING", "CALIBRATION_RETRY_REQUIRED", "RECORDING_LIVE"].includes(row.status)
            ).length
          )
        );
        setCompleted(String(rows.filter((row) => row.status === "COMPLETED").length));
      })
      .catch(() => {
        setSessions("0");
        setActive("0");
        setCompleted("0");
      });

    api
      .get<DocumentRecord[]>("/kb/documents")
      .then((rows) => {
        setDocuments(String(rows.length));
        const t1 = rows.filter((r) => r.evidence_tier === "TIER_1_CANONICAL_RULE").length;
        const t2 = rows.filter((r) => r.evidence_tier === "TIER_2_COMBINATORIAL_MATRIX").length;
        const t3 = rows.filter((r) => r.evidence_tier === "TIER_3_EMPIRICAL_STUDY").length;
        if (t1 + t2 + t3 > 0) {
          setTierBreakdown(`T1: ${t1} · T2: ${t2} · T3: ${t3}`);
        }
      })
      .catch(() => setDocuments("0"));
  }, []);

  return (
    <section className="grid stats-grid" aria-label="Live workspace summary">
      <div className="stat-card">
        <div className="stat-head">
          <span>Recorded sessions</span>
          <span className="stat-icon">
            <Icon name="activity" size={14} />
          </span>
        </div>
        <div className="stat-value">{sessions}</div>
        <div className="stat-foot">
          Total multimodal sessions · <em>SQLite + Timescale ready</em>
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-head">
          <span>Active captures</span>
          <span className="stat-icon">
            <Icon name="clock" size={14} />
          </span>
        </div>
        <div className="stat-value">{active}</div>
        <div className="stat-foot">Calibrating (90s) or live 2.0s telemetry</div>
      </div>

      <div className="stat-card">
        <div className="stat-head">
          <span>Completed reports</span>
          <span className="stat-icon">
            <Icon name="pulse" size={14} />
          </span>
        </div>
        <div className="stat-value">{completed}</div>
        <div className="stat-foot">
          <em>Advocate-Skeptic-Judge</em> synthesized reports
        </div>
      </div>

      <div className="stat-card">
        <div className="stat-head">
          <span>Knowledge base docs</span>
          <span className="stat-icon">
            <Icon name="book" size={14} />
          </span>
        </div>
        <div className="stat-value">{documents}</div>
        <div className="stat-foot">{tierBreakdown}</div>
      </div>
    </section>
  );
}
