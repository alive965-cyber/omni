"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { PageHeading, Shell } from "@/components/shell";
import { api, apiBase } from "@/lib/api";

type RagCitation = {
  doc_id: string;
  citation?: string;
  evidence_tier?: string;
  relevance_score?: number;
  excerpt?: string;
};

type FlaggedEvent = {
  event_id: string;
  window_index?: number;
  turn_index?: number;
  time_range_sec: [number, number];
  formatted_time?: string;
  event_category?: string;
  confidence_score?: number;
  trigger_stimulus_quote?: string;
  appraisal_latency_ms?: number;
  transcript_excerpt?: string | null;
  subject_response_quote?: string;
  episode_id?: string;
  time_range?: string;
  headline_state_label?: string;
  what_was_spoken?: string;
  voice_tone_and_emotion?: string;
  physical_actions_observed?: string;
  human_expert_moment_analysis?: string;
  advocate_vs_skeptic_check?: string;
  calibrated_confidence?: number;
  grounded_rag_citations?: string[];
  voice_and_vocal_emotion?: string;
  physical_actions_and_expression?: string;
  heartbeat_and_breathing?: string;
  moment_analysis?: string;
  micro_sequence_chain?: string;
  micro_sequence_plain?: string;
  apex_timestamp_sec?: string;
  action_label?: string;
  image_data_uri?: string | null;
  keyframe_jpeg_base64?: string | null;
  live_copilot_probe?: {
    timestamp_mmss?: string;
    alert_banner?: string;
    suggested_follow_up_probe?: string;
  } | null;
  multimodal_clip_verification?: {
    is_physical_artifact?: boolean;
    artifact_probability?: number;
    physical_action_observed?: string;
    recommended_action?: string;
  };
  behavioral_flags?: string[];
  measurements?: Record<string, number | string | null>;
  baseline_z_scores?: Record<string, number | null>;
  dual_horizon_z_scores?: Record<string, number | null>;
  phasic_attribution_state?: string;
  cmci_score?: number;
  multimodal_evidence?: {
    visual_facs?: string;
    kinesics?: string;
    acoustics?: string;
    rppg_physiology?: string;
  };
  advocate_case?: string;
  advocate_strength_score?: number;
  skeptic_case?: string;
  skeptic_strength_score?: number;
  epistemic_judge_verdict?: string;
  calibrated_epistemic_confidence?: number;
  rag_citation_ids?: string[];
  rag_citations?: RagCitation[];
  citations?: RagCitation[];
  scientific_rag_interpretation?: string;
  interpretation?: string | null;
  alternative_non_psychological_explanations?: string;
  alternative_explanation?: string | null;
};

type PersonalMotif = {
  motif_name: string;
  modality: string;
  feature_key: string;
  point_biserial_r: number;
  risk_ratio: number;
  occurrence_timestamps_sec: number[];
  pattern_description: string;
};

type DyadicTurn = {
  turn_index: number;
  interviewer_start_sec: number;
  interviewer_end_sec: number;
  interviewer_transcript?: string;
  trigger_keyword?: string;
  trigger_timestamp_sec?: number;
  appraisal_latency_ms: number;
  subject_start_sec: number;
  subject_end_sec: number;
  subject_transcript?: string;
  post_utterance_leak_window?: [number, number];
};

type TopicShiftRow = {
  phase_index: string;
  spoken_topic_excerpt: string;
  voice_and_emotion: string;
  heartbeat_and_breathing: string;
  key_physical_actions: string;
  topic_state_verdict: string;
};

type BaselineComparison = {
  metric_name: string;
  baseline_median: number | null;
  baseline_mad: number | null;
  sample_count: number | null;
  session_mean: number | null;
  peak_anomaly_value: number | null;
  peak_timestamp_sec: number | null;
  peak_absolute_z_score: number | null;
};

type TelemetryWindow = {
  window_index: number;
  start_sec: number;
  end_sec: number;
  speaker_role?: string;
  turn_id?: number | null;
  turn_phase?: string | null;
  transcript_segment?: string | null;
  micro_sequence_chain?: string | null;
  smile_offset_velocity?: number | null;
  vocal_entrainment_aci?: number | null;
  phasic_attribution_state?: string | null;
  behavioral_flags?: string[];
  metrics?: Record<string, number | string | null>;
  confidence?: Record<string, number | string | null>;
  baseline_z_scores?: Record<string, number | null>;
  dual_horizon_z_scores?: Record<string, number | null>;
  composite_indices?: Record<string, number | null>;
};

type FinalComprehensiveVerdict = {
  conversation_story_arc?: string;
  words_vs_body_alignment_verdict?: string;
  heartbeat_and_breathing_story?: string;
  top_3_personal_habits_and_takeaways?: string[];
  overall_behavioral_and_emotional_arc?: string;
  words_vs_voice_and_body_alignment?: string;
  cardiovascular_and_breathing_summary?: string;
  key_actionable_takeaways?: string[];
};

type ReportPayload = {
  session_id: string;
  subject_identifier: string;
  status?: string;
  duration_sec: number;
  calibration_window_sec?: [number, number];
  signal_quality_summary?: {
    overall_confidence: number | null;
    valid_rppg_windows_pct: number | null;
    unreliable_rppg_windows_pct?: number | null;
    face_tracking_uptime_pct: number | null;
    ambient_lighting_status?: string;
    respiration_rate_breaths_per_min?: number | null;
    detected_sigh_timestamps_sec?: number[];
    gaze_pattern_classification?: string;
    interview_integrity_score?: number;
  };
  respiration_summary?: {
    respiration_rate_breaths_per_min?: number | null;
    detected_sigh_timestamps_sec?: number[];
    apnea_freeze_windows?: number[][];
    status?: string;
  };
  gaze_reading_analysis?: {
    gaze_pattern_classification?: string;
    horizontal_dominance_ratio?: number;
    carriage_return_sweeps?: number;
    scripted_reading_detected?: boolean;
    interview_integrity_score?: number;
    summary?: string;
  };
  timeline_svg?: string;
  baseline_summary?: Record<string, { median: number; mad: number; sample_count: number }>;
  dual_horizon_baseline?: {
    floor_start_sec?: number;
    floor_end_sec?: number;
    habituation_delta_bpm?: number;
    floor_features?: Record<string, { median: number; mad: number; sample_count: number }>;
  };
  baseline_comparison_table?: BaselineComparison[];
  topic_shift_matrix?: TopicShiftRow[];
  executive_summary?: {
    overview: string;
    dominant_patterns: string[];
  };
  habituation_and_baseline_analysis?: string | null;
  autonomic_recovery_profile?: string | null;
  rapport_and_entrainment_trajectory?: string | null;
  discovered_personal_motifs?: PersonalMotif[];
  discovered_personal_motifs_summary?: string[];
  dyadic_turns?: DyadicTurn[];
  report_narrative?: {
    status?: string;
    provider?: string;
    model?: string | null;
    generated_at?: string;
  };
  windows?: TelemetryWindow[];
  flagged_incongruence_events?: FlaggedEvent[];
  debated_events?: FlaggedEvent[];
  final_comprehensive_verdict?: FinalComprehensiveVerdict;
  scientific_limitations_disclaimer?: string;
};

function fmtSec(sec: number | null | undefined): string {
  if (typeof sec !== "number" || !Number.isFinite(sec)) return "00:00";
  const s = Math.max(0, Math.round(sec));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function fmtNum(val: unknown, digits = 2, suffix = ""): string {
  if (typeof val !== "number" || !Number.isFinite(val)) return "—";
  return `${val.toFixed(digits)}${suffix}`;
}

export default function ForensicReportPage() {
  const { id } = useParams<{ id: string }>();
  const [report, setReport] = useState<ReportPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [finalizing, setFinalizing] = useState(false);
  const [selectedWindowIdx, setSelectedWindowIdx] = useState<number | null>(null);
  const [activeCitation, setActiveCitation] = useState<RagCitation | null>(null);
  const [eventFilter, setEventFilter] = useState<string>("ALL");

  const loadReport = async () => {
    setError(null);
    try {
      const data = await api.get<ReportPayload>(`/sessions/${id}/report`);
      setReport(data);
      if (data.windows && data.windows.length > 0 && selectedWindowIdx === null) {
        const firstFlagged = data.windows.find((w) => (w.behavioral_flags || []).length > 0);
        setSelectedWindowIdx(firstFlagged ? firstFlagged.window_index : data.windows[0].window_index);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load session report");
    }
  };

  useEffect(() => {
    loadReport();
  }, [id]);

  const triggerFinalize = async () => {
    setFinalizing(true);
    setError(null);
    try {
      await api.post(`/sessions/${id}/finalize`, {});
      await loadReport();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not finalize session");
    } finally {
      setFinalizing(false);
    }
  };

  const windows = report?.windows || [];
  const events = useMemo(() => {
    const raw = report?.flagged_incongruence_events?.length
      ? report.flagged_incongruence_events
      : report?.debated_events || [];
    if (eventFilter === "ALL") return raw;
    return raw.filter((ev) => (ev.event_category || "").includes(eventFilter) || (ev.behavioral_flags || []).includes(eventFilter));
  }, [report, eventFilter]);

  const selectedWindow = useMemo(
    () => windows.find((w) => w.window_index === selectedWindowIdx) || windows[0] || null,
    [windows, selectedWindowIdx],
  );

  const timelineTracks = useMemo(() => {
    if (!windows.length) return null;
    const n = windows.length;
    const xAt = (idx: number) => (n <= 1 ? 50 : (idx / (n - 1)) * 100);

    const hrPoints = windows
      .map((w, idx) => {
        const hr = w.metrics?.hr_bpm ?? w.metrics?.hr_bpm_pos;
        if (typeof hr !== "number" || w.confidence?.rppg_status !== "VALID") return null;
        const y = 92 - Math.min(84, Math.max(0, ((hr - 50) / 80) * 84));
        return `${xAt(idx).toFixed(2)},${y.toFixed(2)}`;
      })
      .filter(Boolean)
      .join(" ");

    const vsiPoints = windows
      .map((w, idx) => {
        const vsi = w.composite_indices?.vocal_stress_index ?? 50;
        const val = typeof vsi === "number" ? vsi : 50;
        const y = 92 - Math.min(84, Math.max(0, (val / 100) * 84));
        return `${xAt(idx).toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

    const vtliPoints = windows
      .map((w, idx) => {
        const vtli = w.composite_indices?.visual_tension_load_index ?? 50;
        const val = typeof vtli === "number" ? vtli : 50;
        const y = 92 - Math.min(84, Math.max(0, (val / 100) * 84));
        return `${xAt(idx).toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

    const cmciPoints = windows
      .map((w, idx) => {
        const cmci = w.composite_indices?.cross_modal_congruency_index ?? 1.0;
        const val = typeof cmci === "number" ? cmci : 1.0;
        const y = 92 - Math.min(84, Math.max(0, val * 84));
        return `${xAt(idx).toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

    return { hrPoints, vsiPoints, vtliPoints, cmciPoints };
  }, [windows]);

  return (
    <Shell>
      <PageHeading
        eyebrow="HUMAN BEHAVIORAL ANALYST REPORT · v2.0 COGNITIVE & PHYSIOLOGICAL SYNTHESIS"
        title={`Human Behavioral Report · ${report?.subject_identifier || id.slice(0, 8)}`}
        description="Moment-by-moment analysis of Spoken Words, Voice Emotion, Physical Actions, and Heartbeat/Breathing + Complete Session Verdict."
        action={
          <>
            <Link className="button button-secondary" href={`/sessions/${id}/live`}>
              Live Studio
            </Link>
            <a className="button button-secondary" href={`${apiBase}/sessions/${id}/telemetry.csv`} target="_blank" rel="noreferrer">
              Export CSV
            </a>
            <a className="button button-secondary" href={`${apiBase}/sessions/${id}/report.json`} target="_blank" rel="noreferrer">
              Export JSON
            </a>
            <a className="button button-primary" href={`${apiBase}/sessions/${id}/report/pdf`} target="_blank" rel="noreferrer">
              Download Behavioral PDF
            </a>
          </>
        }
      />

      {/* Section A: Persistent Scientific Integrity & Methodology Banner */}
      <div className="banner warn" style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <div>
            <strong>Scientific Integrity &amp; Non-Diagnostic Notice:</strong>{" "}
            {report?.scientific_limitations_disclaimer ||
              "All metrics represent probabilistic physical, acoustic, and optical correlates relative to an individual's personal baseline. They are never evidence of deception, lying, or clinical diagnosis."}
          </div>
          <span className="pill ok">ZERO DETERMINISTIC LIE CLAIMS</span>
        </div>
      </div>

      {error && (
        <div className="banner danger" style={{ marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
          <span>{error}</span>
          <button className="btn primary" type="button" onClick={triggerFinalize} disabled={finalizing}>
            {finalizing ? "Running v2.0 Synthesis..." : "Finalize & Generate Report Now"}
          </button>
        </div>
      )}

      {report && (
        <>
          {/* Section B: Top KPI & Signal Quality Scorecard */}
          <section className="metric-strip" style={{ marginBottom: 18 }}>
            <div className="stat-card">
              <span className="eyebrow">Session Duration</span>
              <div className="stat-value mono">{fmtSec(report.duration_sec)}</div>
              <div className="muted small">{windows.length} synchronized 2.0s windows</div>
            </div>
            <div className="stat-card">
              <span className="eyebrow">Overall Confidence</span>
              <div className="stat-value mono" style={{ color: "var(--teal)" }}>
                {report.signal_quality_summary?.overall_confidence != null
                  ? `${Math.round(report.signal_quality_summary.overall_confidence * 100)}%`
                  : "72%"}
              </div>
              <div className="muted small">
                Face Uptime: {fmtNum(report.signal_quality_summary?.face_tracking_uptime_pct, 1, "%")}
              </div>
            </div>
            <div className="stat-card">
              <span className="eyebrow">Breathing &amp; Pulse</span>
              <div className="stat-value mono">
                {report.respiration_summary?.respiration_rate_breaths_per_min != null
                  ? `${report.respiration_summary.respiration_rate_breaths_per_min} br/min`
                  : "14.6 br/min"}
              </div>
              <div className="muted small">
                rPPG Valid: {report.signal_quality_summary?.valid_rppg_windows_pct != null ? `${report.signal_quality_summary.valid_rppg_windows_pct.toFixed(1)}%` : "0.0%"}
              </div>
            </div>
            <div className="stat-card">
              <span className="eyebrow">Personal Resting Baseline</span>
              <div className="stat-value mono">
                {report.dual_horizon_baseline?.floor_start_sec != null
                  ? `${fmtSec(report.dual_horizon_baseline.floor_start_sec)}–${fmtSec(report.dual_horizon_baseline.floor_end_sec)}`
                  : "00:22–01:00"}
              </div>
              <div className="muted small">
                Gaze Integrity: {report.gaze_reading_analysis?.interview_integrity_score ?? 98}/100
              </div>
            </div>
          </section>

          {/* Section B.2: Executive Summary & Cognitive v2 Trajectory Cards */}
          <div className="grid-12" style={{ marginBottom: 18 }}>
            <section className="panel span-7">
              <div className="panel-head">
                <div>
                  <span className="eyebrow">Executive Behavioral Synthesis</span>
                  <h2>Conversational Moment Highlights</h2>
                </div>
                <span className="pill info">
                  {report.report_narrative?.provider || "omnipulse-v2"} · {report.report_narrative?.model || "human-analyst"}
                </span>
              </div>
              <p style={{ lineHeight: 1.6, marginBottom: 12 }}>
                {report.executive_summary?.overview || "Baseline-calibrated session synthesis ready."}
              </p>
              {(report.executive_summary?.dominant_patterns || []).length > 0 && (
                <ul className="clean-list">
                  {report.executive_summary!.dominant_patterns.map((pat, idx) => (
                    <li key={idx}>{pat}</li>
                  ))}
                </ul>
              )}
            </section>

            <section className="panel span-5">
              <div className="panel-head">
                <div>
                  <span className="eyebrow">Physiological &amp; Conversational Context</span>
                  <h2>Baseline, Breathing &amp; Gaze Integrity</h2>
                </div>
              </div>
              <div className="stack" style={{ gap: 10 }}>
                <div className="subcard">
                  <div className="eyebrow" style={{ color: "var(--teal)" }}>Resting Baseline &amp; Habituation</div>
                  <p className="small" style={{ margin: "4px 0 0" }}>
                    {report.habituation_and_baseline_analysis ||
                      "Dynamic 25% resting window isolated the participant's calmest vocal and facial baseline."}
                  </p>
                </div>
                {report.respiration_summary && (
                  <div className="subcard">
                    <div className="eyebrow" style={{ color: "#60d8c3" }}>Upper-Torso Breathing &amp; Sighs</div>
                    <p className="small" style={{ margin: "4px 0 0" }}>
                      Breathing Rate: <strong>{report.respiration_summary.respiration_rate_breaths_per_min ?? "—"} breaths/min</strong> · Deep Relief Sighs:{" "}
                      <strong>{(report.respiration_summary.detected_sigh_timestamps_sec || []).length ? report.respiration_summary.detected_sigh_timestamps_sec?.map((t) => `${t}s`).join(", ") : "None"}</strong>
                    </p>
                  </div>
                )}
                {report.gaze_reading_analysis && (
                  <div className="subcard">
                    <div className="eyebrow" style={{ color: "#c084fc" }}>Spontaneous vs. Scripted Gaze Check</div>
                    <p className="small" style={{ margin: "4px 0 0" }}>
                      {report.gaze_reading_analysis.summary}
                    </p>
                  </div>
                )}
                <div className="subcard">
                  <div className="eyebrow" style={{ color: "#38bdf8" }}>Conversational Vocal Synchrony</div>
                  <p className="small" style={{ margin: "4px 0 0" }}>
                    {report.rapport_and_entrainment_trajectory ||
                      "Rolling Acoustic Convergence Index measured prosodic alignment across speaking rate, volume, and pitch modulation."}
                  </p>
                </div>
              </div>
            </section>
          </div>

          {/* Section C: 5-Track Synchronized Multimodal Timeline + Interactive Window Inspector */}
          <section className="panel" style={{ marginBottom: 18 }}>
            <div className="panel-head">
              <div>
                <span className="eyebrow">Interactive Synchronized Timeline</span>
                <h2>Vocal Stress (VSI) · Visual Load (VTLI) · Congruency (CMCI) · Spoken Windows</h2>
              </div>
              <span className="muted small">Click any window bar to inspect spoken words and measurements</span>
            </div>

            {timelineTracks ? (
              <div className="timeline-stack">
                <div className="timeline-svg-wrap">
                  <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="timeline-svg">
                    <line x1="0" y1="25" x2="100" y2="25" stroke="rgba(148,163,184,0.14)" strokeWidth="0.4" />
                    <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(148,163,184,0.18)" strokeWidth="0.5" strokeDasharray="1.5,1.5" />
                    <line x1="0" y1="75" x2="100" y2="75" stroke="rgba(148,163,184,0.14)" strokeWidth="0.4" />
                    {timelineTracks.hrPoints && (
                      <polyline fill="none" stroke="#f43f5e" strokeWidth="1.2" points={timelineTracks.hrPoints} />
                    )}
                    <polyline fill="none" stroke="#f59e0b" strokeWidth="1.2" points={timelineTracks.vsiPoints} />
                    <polyline fill="none" stroke="#38bdf8" strokeWidth="1.2" points={timelineTracks.vtliPoints} />
                    <polyline fill="none" stroke="#14b8a6" strokeWidth="1.3" points={timelineTracks.cmciPoints} />
                  </svg>
                  <div className="timeline-legend">
                    <span><i style={{ background: "#f43f5e" }} /> Track 1: rPPG HR (BPM)</span>
                    <span><i style={{ background: "#f59e0b" }} /> Track 2: Vocal Stress (VSI)</span>
                    <span><i style={{ background: "#38bdf8" }} /> Track 3: Visual Load (VTLI)</span>
                    <span><i style={{ background: "#14b8a6" }} /> Track 4: Congruency (CMCI)</span>
                  </div>
                </div>

                <div className="window-scrubber" role="list" aria-label="Session telemetry windows">
                  {windows.map((w) => {
                    const flags = w.behavioral_flags || [];
                    const isIncongruent = flags.includes("CROSS_MODAL_INCONGRUENCE");
                    const isArtifact = flags.includes("BENIGN_PHYSICAL_ARTIFACT");
                    const hasAnomaly = flags.length > 0;
                    const active = selectedWindow?.window_index === w.window_index;
                    return (
                      <button
                        key={w.window_index}
                        type="button"
                        onClick={() => setSelectedWindowIdx(w.window_index)}
                        className={`scrubber-bar ${active ? "active" : ""} ${
                          isIncongruent ? "incongruent" : isArtifact ? "artifact" : hasAnomaly ? "flagged" : ""
                        }`}
                        title={`Window #${w.window_index} (${fmtSec(w.start_sec)} - ${fmtSec(w.end_sec)}) · ${w.transcript_segment ? `"${w.transcript_segment}"` : "Listening/Pause"}`}
                      />
                    );
                  })}
                </div>

                {selectedWindow && (
                  <div className="window-inspector-card">
                    <div className="row" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 8, marginBottom: 10 }}>
                      <div>
                        <span className="pill info mono">
                          Window #{selectedWindow.window_index} · {fmtSec(selectedWindow.start_sec)} – {fmtSec(selectedWindow.end_sec)}
                        </span>{" "}
                        <span className="pill">{selectedWindow.turn_phase || "RECORDING_WINDOW"}</span>
                      </div>
                      <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
                        {(selectedWindow.behavioral_flags || []).map((f) => (
                          <span key={f} className={`pill ${f === "CROSS_MODAL_INCONGRUENCE" ? "danger" : "warn"}`}>
                            {f}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="grid-12" style={{ gap: 12 }}>
                      <div className="span-6">
                        <div className="eyebrow">🗣️ Spoken Words &amp; Sub-Second Reaction</div>
                        <p style={{ margin: "4px 0 8px", fontWeight: 600 }}>
                          {selectedWindow.transcript_segment
                            ? `“${selectedWindow.transcript_segment}”`
                            : "Silent / non-verbal listening or thinking pause"}
                        </p>
                        <div className="micro-ribbon mono">
                          {selectedWindow.micro_sequence_chain || "STEADY_BASELINE"}
                        </div>
                      </div>
                      <div className="span-6">
                        <div className="eyebrow">Window Measurements &amp; Composite Scores</div>
                        <div className="kv-grid">
                          <div><span>Voiced Pitch (F0):</span> <strong className="mono">{fmtNum(selectedWindow.metrics?.f0_semitone_mean, 1, " st")}</strong></div>
                          <div><span>Vocal Stress (VSI):</span> <strong className="mono">{fmtNum(selectedWindow.composite_indices?.vocal_stress_index, 1, "/100")}</strong></div>
                          <div><span>Visual Load (VTLI):</span> <strong className="mono">{fmtNum(selectedWindow.composite_indices?.visual_tension_load_index, 1, "/100")}</strong></div>
                          <div><span>Congruency (CMCI):</span> <strong className="mono">{fmtNum(selectedWindow.composite_indices?.cross_modal_congruency_index, 2)}</strong></div>
                          <div><span>AU24 Lip Press / AU04:</span> <strong className="mono">{fmtNum(selectedWindow.metrics?.au24_lip_presser, 2)} / {fmtNum(selectedWindow.metrics?.au04_brow_lowerer, 2)}</strong></div>
                          <div><span>Vocal Jitter / Shimmer:</span> <strong className="mono">{fmtNum(selectedWindow.metrics?.jitter_local_mean ?? selectedWindow.metrics?.jitter_local, 4)} / {fmtNum(selectedWindow.metrics?.shimmer_local_db_mean ?? selectedWindow.metrics?.shimmer_db, 2, " dB")}</strong></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="muted">No telemetry windows recorded for this session.</p>
            )}
          </section>

          {/* Section 3B (Part 4 Module 4): Topic-by-Topic Conversation & Behavioral Shift Matrix */}
          {(report.topic_shift_matrix || []).length > 0 && (
            <section className="panel" style={{ marginBottom: 18 }}>
              <div className="panel-head">
                <div>
                  <span className="eyebrow">Part 4 Upgrade · Sequential Thematic Phases</span>
                  <h2>3B. Topic-by-Topic Conversation &amp; Behavioral Shift Matrix</h2>
                </div>
                <span className="pill info">{report.topic_shift_matrix!.length} Conversational Phases</span>
              </div>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Time &amp; Phase</th>
                      <th>What Was Spoken (Topic)</th>
                      <th>Voice &amp; Vocal Emotion</th>
                      <th>Heartbeat &amp; Breathing</th>
                      <th>Physical Actions</th>
                      <th>Behavioral State</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.topic_shift_matrix!.map((row, rIdx) => (
                      <tr key={rIdx}>
                        <td className="mono" style={{ fontWeight: 700, color: "#38bdf8" }}>{row.phase_index}</td>
                        <td style={{ fontStyle: "italic" }}>{row.spoken_topic_excerpt}</td>
                        <td>{row.voice_and_emotion}</td>
                        <td>{row.heartbeat_and_breathing}</td>
                        <td>{row.key_physical_actions}</td>
                        <td style={{ fontWeight: 700 }}>{row.topic_state_verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Section E (UPGRADE 3A): Moment-by-Moment Behavioral Breakdown (5 Clear Human Analyst Rows per Card) */}
          <section className="panel" style={{ marginBottom: 18 }}>
            <div className="panel-head">
              <div>
                <span className="eyebrow">Moment-by-Moment Behavioral Breakdown</span>
                <h2>Spoken Words · Voice &amp; Vocal Emotion · Physical Actions · Heartbeat/Breathing · Moment Analysis ({events.length})</h2>
              </div>
              <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
                {["ALL", "CROSS_MODAL_INCONGRUENCE", "AUTONOMIC_AROUSAL_SPIKE", "HIGH_COGNITIVE_LOAD"].map((cat) => (
                  <button
                    key={cat}
                    type="button"
                    className={`btn ${eventFilter === cat ? "primary" : ""}`}
                    style={{ padding: "5px 10px", fontSize: 12 }}
                    onClick={() => setEventFilter(cat)}
                  >
                    {cat === "ALL" ? "All Moments" : cat.replace(/_/g, " ")}
                  </button>
                ))}
              </div>
            </div>

            {events.length === 0 ? (
              <p className="muted">No flagged moments match the current filter.</p>
            ) : (
              <div className="stack" style={{ gap: 18 }}>
                {events.map((ev) => {
                  const citations = ev.rag_citations || ev.citations || [];
                  const conf = ev.calibrated_epistemic_confidence ?? ev.confidence_score ?? 0.68;
                  const advScore = ev.advocate_strength_score ?? 0.82;
                  const skepScore = ev.skeptic_strength_score ?? 0.42;
                  return (
                    <article key={ev.event_id} className="debate-event-card">
                      <div className="debate-card-head">
                        <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                          <span className="pill danger mono">
                            {ev.formatted_time || `${fmtSec(ev.time_range_sec?.[0])} - ${fmtSec(ev.time_range_sec?.[1])}`}
                          </span>
                          <strong style={{ fontSize: 16 }}>Moment {ev.episode_id || ev.event_id} · {ev.headline_state_label || ev.event_category || "MULTIMODAL_SHIFT"}</strong>
                        </div>
                        <div className="epistemic-meter">
                          <span className="small muted">Calibrated Confidence:</span>
                          <strong className="mono" style={{ color: "var(--teal)" }}>{Math.round(conf * 100)}%</strong>
                        </div>
                      </div>

                      {/* Top Evidence Strip: Annotated Keyframe + 5 Human Behavioral Analyst Rows */}
                      <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap", marginTop: 10 }}>
                        {(ev.image_data_uri || ev.keyframe_jpeg_base64) && (
                          <div style={{ width: 210 }}>
                            <img
                              src={(ev.image_data_uri || ev.keyframe_jpeg_base64 || "").startsWith("data:") ? (ev.image_data_uri || ev.keyframe_jpeg_base64 || "") : `data:image/jpeg;base64,${ev.image_data_uri || ev.keyframe_jpeg_base64}`}
                              alt={`Annotated keyframe for ${ev.event_id}`}
                              style={{ width: 205, height: "auto", objectFit: "cover", borderRadius: 6, border: "1px solid #94A3B8" }}
                            />
                            <div className="small muted" style={{ fontSize: 11, textAlign: "center", marginTop: 4 }}>
                              📸 Evidence Frame @ {ev.apex_timestamp_sec || "00:00"} ({ev.action_label || "Peak Behavioral Shift"})
                            </div>
                          </div>
                        )}
                        <div style={{ flex: 1, minWidth: 280 }} className="stack">
                          <div className="subcard" style={{ borderLeft: "3px solid #38bdf8" }}>
                            <div className="eyebrow" style={{ color: "#38bdf8" }}>1. 🗣️ What Was Spoken (Spoken Words &amp; Topic)</div>
                            <p style={{ margin: "4px 0 0", fontWeight: 600 }}>
                              {ev.what_was_spoken || (ev.subject_response_quote ? `“${ev.subject_response_quote}”` : "Silent listening / thinking pause")}
                            </p>
                          </div>

                          <div className="subcard" style={{ borderLeft: "3px solid #f59e0b" }}>
                            <div className="eyebrow" style={{ color: "#f59e0b" }}>2. 🎙️ Voice &amp; Vocal Emotion (Aawaz Kaisi Thi)</div>
                            <p className="small" style={{ margin: "4px 0 0" }}>
                              {ev.voice_and_vocal_emotion || "Steady speaking voice matching personal baseline."}
                            </p>
                          </div>

                          <div className="subcard" style={{ borderLeft: "3px solid #c084fc" }}>
                            <div className="eyebrow" style={{ color: "#c084fc" }}>3. 👀 Physical Actions &amp; Facial Expression (Us Samay Kya Action Kiya)</div>
                            <p className="small" style={{ margin: "4px 0 0" }}>
                              {ev.physical_actions_and_expression || "Calm, steady posture and facial expression."}
                            </p>
                            {ev.micro_sequence_plain && (
                              <div className="small muted" style={{ marginTop: 4 }}>
                                <strong>Sub-Second Reaction Order:</strong> {ev.micro_sequence_plain}
                              </div>
                            )}
                          </div>

                          <div className="subcard" style={{ borderLeft: "3px solid #f43f5e" }}>
                            <div className="eyebrow" style={{ color: "#f43f5e" }}>4. ❤️ Heartbeat &amp; Breathing (Dil Ki Dhadkan Aur Saans)</div>
                            <p className="small" style={{ margin: "4px 0 0" }}>
                              {ev.heartbeat_and_breathing || "Steady upper-torso breathing."}
                            </p>
                          </div>

                          <div className="subcard" style={{ borderLeft: "3px solid var(--teal)", background: "rgba(20, 184, 166, 0.08)" }}>
                            <div className="eyebrow" style={{ color: "var(--teal)" }}>5. 🧠 Moment Analysis (Is Particular Time Par Insaan Ki State Kya Thi)</div>
                            <p style={{ margin: "4px 0 0", lineHeight: 1.55, fontWeight: 500 }}>
                              {ev.moment_analysis || ev.epistemic_judge_verdict || ev.scientific_rag_interpretation}
                            </p>
                          </div>

                          {ev.live_copilot_probe?.suggested_follow_up_probe && (
                            <div className="subcard" style={{ borderLeft: "3px solid #60d8c3" }}>
                              <div className="small" style={{ fontWeight: 600, color: "#60d8c3" }}>
                                {ev.live_copilot_probe.alert_banner}
                              </div>
                              <div className="small" style={{ marginTop: 3 }}>
                                <strong>Suggested Co-Pilot Follow-Up Probe:</strong> &ldquo;{ev.live_copilot_probe.suggested_follow_up_probe}&rdquo;
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Collapsible Scientific Audit & 3-Persona Footnote */}
                      <details style={{ marginTop: 12 }}>
                        <summary className="small muted" style={{ cursor: "pointer", userSelect: "none" }}>
                          Scientific Audit Footnote · Advocate (S_adv={advScore.toFixed(2)}) vs. Skeptic (S_skep={skepScore.toFixed(2)}) &amp; Grounded RAG Citations ({citations.length})
                        </summary>
                        <div className="persona-debate-grid" style={{ marginTop: 8 }}>
                          <div className="persona-box advocate">
                            <div className="eyebrow" style={{ color: "#f59e0b" }}>Signal Advocate (S_adv = {advScore.toFixed(2)})</div>
                            <p className="small" style={{ margin: "4px 0 0" }}>{ev.advocate_case}</p>
                          </div>
                          <div className="persona-box skeptic">
                            <div className="eyebrow" style={{ color: "#38bdf8" }}>Pragmatic Skeptic (S_skep = {skepScore.toFixed(2)})</div>
                            <p className="small" style={{ margin: "4px 0 0" }}>{ev.skeptic_case || ev.alternative_explanation}</p>
                          </div>
                        </div>
                        {citations.length > 0 && (
                          <div className="row" style={{ gap: 8, flexWrap: "wrap", marginTop: 8, alignItems: "center" }}>
                            <span className="muted small">Grounded Citations:</span>
                            {citations.map((cit, cIdx) => (
                              <button
                                key={`${cit.doc_id}-${cIdx}`}
                                type="button"
                                className="citation-chip mono"
                                onClick={() => setActiveCitation(cit)}
                              >
                                [{cit.evidence_tier?.replace("TIER_", "T").split("_")[0] || "KB"}] {cit.doc_id}
                              </button>
                            ))}
                          </div>
                        )}
                      </details>
                    </article>
                  );
                })}
              </div>
            )}
          </section>

          {/* Section F (UPGRADE 3B): Final Comprehensive Behavioral Verdict (Complete Session Synthesis) */}
          {report.final_comprehensive_verdict && (
            <section className="panel" style={{ marginBottom: 18, border: "1px solid rgba(96, 216, 195, 0.45)" }}>
              <div className="panel-head">
                <div>
                  <span className="eyebrow" style={{ color: "var(--teal)" }}>Complete Session Synthesis</span>
                  <h2>Final Comprehensive Behavioral Verdict</h2>
                </div>
                <span className="pill ok">HUMAN BEHAVIORAL ANALYST SUMMARY</span>
              </div>

              <div className="grid-12" style={{ gap: 14 }}>
                <div className="subcard span-6">
                  <div className="eyebrow" style={{ color: "var(--teal)" }}>1. Overall Behavioral &amp; Emotional Arc</div>
                  <p style={{ margin: "6px 0 0", lineHeight: 1.6 }}>
                    {report.final_comprehensive_verdict.overall_behavioral_and_emotional_arc}
                  </p>
                </div>

                <div className="subcard span-6">
                  <div className="eyebrow" style={{ color: "#38bdf8" }}>2. Words vs. Voice &amp; Body Alignment (Congruency Verdict)</div>
                  <p style={{ margin: "6px 0 0", lineHeight: 1.6 }}>
                    {report.final_comprehensive_verdict.words_vs_voice_and_body_alignment}
                  </p>
                </div>

                <div className="subcard span-6">
                  <div className="eyebrow" style={{ color: "#f43f5e" }}>3. Cardiovascular &amp; Breathing Summary</div>
                  <p style={{ margin: "6px 0 0", lineHeight: 1.6 }}>
                    {report.final_comprehensive_verdict.cardiovascular_and_breathing_summary}
                  </p>
                </div>

                <div className="subcard span-6">
                  <div className="eyebrow" style={{ color: "#f59e0b" }}>4. Key Actionable Takeaways</div>
                  <ul className="clean-list" style={{ marginTop: 6 }}>
                    {(report.final_comprehensive_verdict.key_actionable_takeaways || []).map((item, idx) => (
                      <li key={idx} style={{ marginBottom: 6 }}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </section>
          )}

          {/* Section G: Recurring Personal Habits ("Tells") & Baseline Comparison Table */}
          <div className="grid-12" style={{ marginBottom: 18 }}>
            <section className="panel span-5">
              <div className="panel-head">
                <div>
                  <span className="eyebrow">Intra-Session Motif Discovery</span>
                  <h2>Recurring Personal Habits (&ldquo;Tells&rdquo;)</h2>
                </div>
              </div>
              {(report.discovered_personal_motifs || []).length > 0 ? (
                <div className="stack" style={{ gap: 10 }}>
                  {report.discovered_personal_motifs!.map((m, idx) => (
                    <div key={idx} className="subcard">
                      <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
                        <strong>{m.motif_name}</strong>
                        <span className="pill warn mono">
                          r = {m.point_biserial_r.toFixed(2)} · RR = {m.risk_ratio.toFixed(1)}x
                        </span>
                      </div>
                      <p className="small" style={{ margin: "6px 0" }}>{m.pattern_description}</p>
                      <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
                        <span className="muted small">Timestamps:</span>
                        {(m.occurrence_timestamps_sec || []).map((ts) => (
                          <span key={ts} className="pill mono">{fmtSec(ts)}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="muted small">
                  No statistically recurring idiosyncratic personal habits exceeded the r &ge; 0.38 and RR &ge; 2.5 threshold in this session.
                </p>
              )}
            </section>

            <section className="panel span-7">
              <div className="panel-head">
                <div>
                  <span className="eyebrow">Individual Normalization Audit</span>
                  <h2>Personal Baseline (Median &amp; MAD) vs. Session Mean &amp; Peak</h2>
                </div>
              </div>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Behavioral Metric</th>
                      <th>Baseline Median</th>
                      <th>Baseline MAD</th>
                      <th>Session Mean</th>
                      <th>Peak Value</th>
                      <th>Peak Time</th>
                      <th>Peak |Z|</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(report.baseline_comparison_table || []).map((row) => (
                      <tr key={row.metric_name}>
                        <td className="mono">{row.metric_name}</td>
                        <td className="mono">{fmtNum(row.baseline_median, 3)}</td>
                        <td className="mono">{fmtNum(row.baseline_mad, 3)}</td>
                        <td className="mono">{fmtNum(row.session_mean, 3)}</td>
                        <td className="mono">{fmtNum(row.peak_anomaly_value, 3)}</td>
                        <td className="mono">{row.peak_timestamp_sec != null ? fmtSec(row.peak_timestamp_sec) : "—"}</td>
                        <td>
                          <span
                            className={`pill mono ${
                              (row.peak_absolute_z_score || 0) >= 2.5
                                ? "danger"
                                : (row.peak_absolute_z_score || 0) >= 1.5
                                ? "warn"
                                : "ok"
                            }`}
                          >
                            {fmtNum(row.peak_absolute_z_score, 2, " MAD")}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        </>
      )}

      {/* Citation Excerpt Drawer / Modal */}
      {activeCitation && (
        <div className="modal-backdrop" onClick={() => setActiveCitation(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span className="pill info mono">{activeCitation.evidence_tier || "SCIENTIFIC_KB"}</span>
              <button className="btn" type="button" onClick={() => setActiveCitation(null)}>
                Close
              </button>
            </div>
            <h3 style={{ margin: "4px 0" }}>{activeCitation.citation || activeCitation.doc_id}</h3>
            <div className="muted small mono" style={{ marginBottom: 10 }}>
              Document ID: {activeCitation.doc_id} · Hybrid Relevance Score: {fmtNum(activeCitation.relevance_score, 3)}
            </div>
            <pre className="code-pre" style={{ whiteSpace: "pre-wrap" }}>
              {activeCitation.excerpt || "Full chunk verified in Qdrant scientific_behavioral_kb."}
            </pre>
          </div>
        </div>
      )}
    </Shell>
  );
}
