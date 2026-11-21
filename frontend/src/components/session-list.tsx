"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Icon } from "@/components/shell";

export type SessionRecord = {
  id: string;
  subject_identifier: string;
  session_title: string;
  status: string;
  ambient_lux_score?: number | null;
  mic_snr_db?: number | null;
  calibration_duration_sec?: number;
  created_at: string;
  completed_at?: string | null;
};

function sessionDate(value: string) {
  const normalized = value.includes("T") ? value : value.replace(" ", "T") + "Z";
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function statusTone(status: string) {
  if (status === "COMPLETED") return "green";
  if (["RECORDING_LIVE", "CALIBRATING"].includes(status)) return "blue";
  if (status === "CALIBRATION_RETRY_REQUIRED") return "amber";
  if (["FAILED", "CANCELLED"].includes(status)) return "rose";
  return "neutral";
}

function sessionHref(session: SessionRecord) {
  return session.status === "COMPLETED" ? `/sessions/${session.id}/report` : `/sessions/${session.id}/live`;
}

export function RecentSessions({ limit, filterStatus }: { limit?: number; filterStatus?: string }) {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get<SessionRecord[]>("/sessions")
      .then(setSessions)
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Could not load sessions."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="card empty-state">
        <span className="loading-dot" />
        Loading sessions from the OmniPulse API…
      </div>
    );
  }
  if (error) {
    return <div className="card empty-state">Sessions could not be loaded: {error}</div>;
  }

  const filtered =
    filterStatus && filterStatus !== "ALL"
      ? sessions.filter((s) => s.status === filterStatus)
      : sessions;
  const visible = typeof limit === "number" ? filtered.slice(0, limit) : filtered;

  if (!visible.length) {
    return (
      <div className="card empty-state">
        No sessions match this filter yet. Start a consented live session to capture baseline-relative telemetry.
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: "6px 14px" }}>
      <div style={{ overflowX: "auto" }}>
        <table className="session-table">
          <thead>
            <tr>
              <th>SESSION TITLE</th>
              <th>PARTICIPANT</th>
              <th>ENVIRONMENT</th>
              <th>CREATED</th>
              <th>STATUS</th>
              <th style={{ textAlign: "right" }}>ACTIONS</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((session) => (
              <tr key={session.id}>
                <td>
                  <div className="session-title">{session.session_title}</div>
                  <div className="micro-label" style={{ marginTop: 3 }}>
                    ID · {session.id.slice(0, 8)}
                  </div>
                </td>
                <td>
                  <span className="code-inline">{session.subject_identifier}</span>
                </td>
                <td>
                  <span className="subtle-note">
                    {typeof session.ambient_lux_score === "number"
                      ? `Luma ${Math.round(session.ambient_lux_score)} · `
                      : ""}
                    {typeof session.mic_snr_db === "number"
                      ? `SNR ${session.mic_snr_db.toFixed(1)} dB`
                      : "Standard setup"}
                  </span>
                </td>
                <td>{sessionDate(session.created_at)}</td>
                <td>
                  <Badge tone={statusTone(session.status)}>{session.status.replaceAll("_", " ")}</Badge>
                </td>
                <td style={{ textAlign: "right" }}>
                  <div style={{ display: "inline-flex", gap: 10, alignItems: "center" }}>
                    {session.status === "COMPLETED" && (
                      <a
                        href={api.url(`/sessions/${session.id}/report/pdf`)}
                        className="button-link"
                        target="_blank"
                        rel="noreferrer"
                        title="Download PDF Report"
                      >
                        PDF <Icon name="download" size={12} />
                      </a>
                    )}
                    <Link href={sessionHref(session)} className="button-link">
                      {session.status === "COMPLETED" ? "Inspect Report" : "Open Studio"}{" "}
                      <Icon name="arrow" size={12} />
                    </Link>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
