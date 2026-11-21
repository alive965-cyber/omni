"use client";

import Link from "next/link";
import { useState } from "react";
import { Icon, PageHeading } from "@/components/shell";
import { RecentSessions } from "@/components/session-list";

const STATUS_FILTERS = [
  { key: "ALL", label: "All Sessions" },
  { key: "COMPLETED", label: "Completed Reports" },
  { key: "RECORDING_LIVE", label: "Recording Live" },
  { key: "CALIBRATING", label: "Calibrating" },
  { key: "CREATED", label: "Created" },
];

export default function SessionsPage() {
  const [statusFilter, setStatusFilter] = useState("ALL");

  return (
    <>
      <PageHeading
        eyebrow="SESSION ARCHIVE & REPORTS"
        title="Multimodal telemetry sessions"
        description="Browse recorded sessions, resume active captures, or open completed 10-minute forensic cognitive reports."
        action={
          <Link href="/sessions/new" className="button button-primary">
            <Icon name="scan" size={15} /> New session
          </Link>
        }
      />

      <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 14 }}>
        {STATUS_FILTERS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`tag-choice ${statusFilter === item.key ? "selected" : ""}`}
            onClick={() => setStatusFilter(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <RecentSessions filterStatus={statusFilter} />
    </>
  );
}
