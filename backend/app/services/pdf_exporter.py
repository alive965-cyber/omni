"""Human Behavioral Analyst & Forensic Multi-Section PDF Report Renderer for OmniPulse MBS v2.0 (pdf_exporter.py).
Implements Part 4 Technical Specification (Modules 1-5):
  - Page 1: Embedded 3-Track Synchronized Vector Timeline Chart + Section 1 Executive Summary + Section 2 Signal Quality Scorecard
  - Section 3: Personal Baseline (Median / MAD) vs. Session Behavioral Comparison (Zero Blacklisted Camera Artifacts)
  - Section 3B: Topic-by-Topic Conversation & Behavioral Shift Matrix (4 Conversational Phases comparing Spoken Topic, Voice & Emotion, Heartbeat & Breathing, Physical Actions, and Behavioral State Verdict)
  - Section 4: Recurring Personal Behavioral Habits ('Tells')
  - Section 5: Side-by-Side Annotated Video Evidence Snapshot (`Evidence Frame @ MM:SS.ms (Action Label)`) + 5-Channel Human Analysis Card (Spoken Words, Voice Tone & Emotion, Physical Action (Video), Heartbeat & Breathing, Amber-Bordered Human Expert Analysis Box, and Advocate vs. Skeptic Check + Grounded Citations)
  - Section 6: Compact Synchronized Telemetry Windows Matrix
  - Section 7: Final Comprehensive Behavioral Verdict (Complete Session Synthesis)
"""
from __future__ import annotations

import base64
from io import BytesIO
from math import isfinite
from typing import Any

from app.services.human_signal_translator import HumanSignalTranslator


def _build_vector_timeline_drawing(windows: list[dict[str, Any]], width: float = 504.0, height: float = 126.0):
    from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
    from reportlab.lib import colors

    d = Drawing(width, height)
    d.add(
        Rect(
            0,
            0,
            width,
            height,
            fillColor=colors.HexColor("#FFFFFF"),
            strokeColor=colors.HexColor("#CBD5E1"),
            strokeWidth=0.8,
            rx=4,
            ry=4,
        )
    )
    if not windows:
        return d

    ordered = sorted(windows, key=lambda x: float(x.get("start_sec", 0.0)))
    max_t = max(10.0, max(float(w.get("end_sec", 2.0)) for w in ordered))
    pad_l, pad_r, pad_t, pad_b = 36.0, 14.0, 22.0, 22.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    d.add(
        String(
            pad_l,
            height - 14,
            "Synchronized 3-Track Multimodal Timeline",
            fontName="Helvetica-Bold",
            fontSize=8.0,
            fillColor=colors.HexColor("#0F172A"),
        )
    )
    d.add(Rect(width - 210, height - 15, 7, 7, fillColor=colors.HexColor("#0284C7"), strokeColor=None))
    d.add(String(width - 200, height - 14, "Vocal Stress (VSI)", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#334155")))
    d.add(Rect(width - 125, height - 15, 7, 7, fillColor=colors.HexColor("#D97706"), strokeColor=None))
    d.add(String(width - 115, height - 14, "Visual Tension (VTLI)", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#334155")))
    d.add(Rect(width - 36, height - 15, 7, 7, fillColor=colors.HexColor("#14B8A6"), strokeColor=None))
    d.add(String(width - 26, height - 14, "CMCI", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#334155")))

    for w in ordered:
        ci = w.get("composite_indices") or {}
        cmci = w.get("cmci_score") if w.get("cmci_score") is not None else ci.get("cross_modal_congruency_index", 1.0)
        cmci_f = float(cmci) if isinstance(cmci, (int, float)) else 1.0
        passes_r3 = bool(w.get("passes_rule_of_3", False))
        flags = w.get("behavioral_flags") or []
        if passes_r3 or cmci_f < 0.55 or len(flags) >= 2:
            x1 = pad_l + (float(w.get("start_sec", 0.0)) / max_t) * plot_w
            x2 = pad_l + (float(w.get("end_sec", 2.0)) / max_t) * plot_w
            d.add(
                Rect(
                    x1,
                    pad_b,
                    max(2.0, x2 - x1),
                    plot_h,
                    fillColor=colors.HexColor("#FEE2E2"),
                    strokeColor=None,
                )
            )

    d.add(
        Line(
            pad_l,
            pad_b + plot_h * 0.5,
            width - pad_r,
            pad_b + plot_h * 0.5,
            strokeColor=colors.HexColor("#CBD5E1"),
            strokeWidth=0.6,
            strokeDashArray=[3, 3],
        )
    )
    d.add(
        Line(
            pad_l,
            pad_b,
            width - pad_r,
            pad_b,
            strokeColor=colors.HexColor("#475569"),
            strokeWidth=0.9,
        )
    )

    def _pt(t_val: float, score_0_100: float) -> tuple[float, float]:
        x = pad_l + (t_val / max_t) * plot_w
        v = max(0.0, min(100.0, float(score_0_100)))
        y = pad_b + (v / 100.0) * plot_h
        return (x, y)

    vsi_coords: list[float] = []
    vtli_coords: list[float] = []
    cmci_coords: list[float] = []
    for w in ordered:
        t_s = float(w.get("start_sec", 0.0))
        ci = w.get("composite_indices") or {}
        vsi = w.get("vsi_score") if w.get("vsi_score") is not None else ci.get("vocal_stress_index", 50.0)
        vtli = w.get("vtli_score") if w.get("vtli_score") is not None else ci.get("visual_tension_load_index", 50.0)
        cmci = w.get("cmci_score") if w.get("cmci_score") is not None else ci.get("cross_modal_congruency_index", 0.85)
        vsi_f = float(vsi) if isinstance(vsi, (int, float)) and isfinite(float(vsi)) else 50.0
        vtli_f = float(vtli) if isinstance(vtli, (int, float)) and isfinite(float(vtli)) else 50.0
        cmci_f = (float(cmci) * 100.0) if isinstance(cmci, (int, float)) and isfinite(float(cmci)) else 85.0
        vsi_coords.extend(_pt(t_s, vsi_f))
        vtli_coords.extend(_pt(t_s, vtli_f))
        cmci_coords.extend(_pt(t_s, cmci_f))

    if len(cmci_coords) >= 4:
        d.add(PolyLine(cmci_coords, strokeColor=colors.HexColor("#14B8A6"), strokeWidth=1.1))
    if len(vsi_coords) >= 4:
        d.add(PolyLine(vsi_coords, strokeColor=colors.HexColor("#0284C7"), strokeWidth=1.5))
    if len(vtli_coords) >= 4:
        d.add(PolyLine(vtli_coords, strokeColor=colors.HexColor("#D97706"), strokeWidth=1.5))

    d.add(String(pad_l, 7, "0.0s", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#475569")))
    d.add(String(pad_l + plot_w * 0.5 - 12, 7, f"{max_t * 0.5:.1f}s", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#475569")))
    d.add(String(width - pad_r - 28, 7, f"{max_t:.1f}s", fontName="Helvetica", fontSize=7.5, fillColor=colors.HexColor("#475569")))
    return d


def render_report_pdf(report: dict[str, Any]) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image as RLImage,
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("PDF export requires reportlab; install the backend 'documents' extra.") from exc

    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=letter,
        leftMargin=0.50 * inch,
        rightMargin=0.50 * inch,
        topMargin=0.50 * inch,
        bottomMargin=0.50 * inch,
        title=f"OmniPulse MBS Human Behavioral Analyst Report - {report.get('session_id', '')}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "OmniTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14.5,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "OmniSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceAfter=7,
    )
    h2_style = ParagraphStyle(
        "OmniH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.2,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=9,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "OmniBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.0,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=3.5,
    )
    small_style = ParagraphStyle(
        "OmniSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=10.4,
        textColor=colors.HexColor("#334155"),
        spaceAfter=2.5,
    )
    banner_style = ParagraphStyle(
        "OmniBanner",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#7C2D12"),
        backColor=colors.HexColor("#FFF7ED"),
        borderPadding=5,
        spaceAfter=7,
    )
    copilot_style = ParagraphStyle(
        "OmniCopilot",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#1E3A8A"),
        backColor=colors.HexColor("#EFF6FF"),
        borderPadding=4.5,
        spaceBefore=3,
        spaceAfter=3,
    )
    verdict_box_style = ParagraphStyle(
        "OmniVerdictBox",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.4,
        leading=11.8,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F0FDFA"),
        borderPadding=6,
        spaceBefore=3,
        spaceAfter=5,
    )

    story: list[Any] = []

    def _esc(text: Any) -> str:
        s = str(text if text is not None else "—")
        for emoji_char in ("🗣️", "🎙️", "👀", "❤️", "🧠", "⚡", "⏱️", "📸", "⚖️"):
            s = s.replace(emoji_char, "")
        return s.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Header & Methodology Banner
    story.append(Paragraph("OMNIPULSE MBS v2.0 · HUMAN BEHAVIORAL ANALYST &amp; PHYSIOLOGICAL REPORT", title_style))
    story.append(
        Paragraph(
            f"<b>Session ID:</b> {_esc(report.get('session_id'))} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Participant:</b> {_esc(report.get('subject_identifier'))} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Duration:</b> {_esc(report.get('duration_sec', 0))}s",
            subtitle_style,
        )
    )
    disclaimer = report.get("scientific_limitations_disclaimer") or (
        "All metrics represent probabilistic physical, acoustic, and optical correlates relative to an individual's "
        "personal baseline. Never equate behavioral signals with deception, lying, or clinical diagnosis."
    )
    story.append(Paragraph(f"<b>Scientific Integrity &amp; Methodology Notice:</b> {_esc(disclaimer)}", banner_style))

    # Page 1 Embedded 3-Track Vector Timeline Chart
    windows = report.get("windows") or []
    if windows:
        story.append(_build_vector_timeline_drawing(windows, width=510.0, height=120.0))
        story.append(Spacer(1, 5))

    # Section 1: Executive Behavioral Summary & Session Overview
    story.append(Paragraph("1. Executive Behavioral Summary &amp; Conversational Highlights", h2_style))
    summary = report.get("executive_summary") or {}
    story.append(Paragraph(_esc(summary.get("overview") or "No narrative summary available."), body_style))
    for pattern in summary.get("dominant_patterns") or []:
        story.append(Paragraph(f"&bull; {_esc(pattern)}", small_style))

    if report.get("habituation_and_baseline_analysis"):
        story.append(Paragraph(f"<b>Personal Resting Baseline &amp; Habituation:</b> {_esc(report.get('habituation_and_baseline_analysis'))}", small_style))
    if report.get("autonomic_recovery_profile"):
        story.append(Paragraph(f"<b>Autonomic &amp; Vagal Recovery:</b> {_esc(report.get('autonomic_recovery_profile'))}", small_style))

    resp_summary = report.get("respiration_summary") or {}
    if resp_summary.get("respiration_rate_breaths_per_min") is not None:
        sighs = resp_summary.get("detected_sigh_timestamps_sec") or []
        apneas = resp_summary.get("apnea_freeze_windows") or []
        story.append(
            Paragraph(
                f"<b>Upper-Torso Breathing &amp; Sigh Profile:</b> "
                f"Measured Thoracic Respiration Rate = {_esc(resp_summary.get('respiration_rate_breaths_per_min'))} breaths/min "
                f"({_esc(resp_summary.get('status', 'VALID_TORSO_KINEMATICS'))}); "
                f"Deep Post-Effort Relief Sighs at {_esc(sighs if sighs else 'none detected')}; "
                f"Brief Breath-Hold Pauses: {_esc(apneas if apneas else 'none')}.",
                small_style,
            )
        )

    gaze_analysis = report.get("gaze_reading_analysis") or {}
    if gaze_analysis.get("summary"):
        story.append(
            Paragraph(
                f"<b>Spontaneous vs. Scripted Gaze Assessment:</b> {_esc(gaze_analysis.get('summary'))}",
                small_style,
            )
        )

    if report.get("rapport_and_entrainment_trajectory"):
        story.append(Paragraph(f"<b>Conversational Vocal Synchrony:</b> {_esc(report.get('rapport_and_entrainment_trajectory'))}", small_style))

    # Section 2: Signal Quality Scorecard
    story.append(Paragraph("2. Signal Confidence &amp; Quality Scorecard", h2_style))
    sq = dict(report.get("signal_quality_summary") or {})
    valid_rppg_pct = sq.get("valid_rppg_windows_pct")
    if valid_rppg_pct is None or (isinstance(valid_rppg_pct, (int, float)) and float(valid_rppg_pct) < 20.0):
        curr_conf = sq.get("overall_confidence")
        sq["overall_confidence"] = round(min(0.72, float(curr_conf)), 2) if isinstance(curr_conf, (int, float)) else 0.72

    sq_rows = [["Quality Indicator", "Measured Value"]]
    for k, v in sq.items():
        sq_rows.append([k.replace("_", " ").title(), "Unavailable" if v is None else f"{v}"])
    t_sq = Table(sq_rows, colWidths=[3.3 * inch, 3.8 * inch])
    t_sq.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("PADDING", (0, 0), (-1, -1), 3.2),
        ])
    )
    story.append(t_sq)

    # Section 3: Personal Baseline Comparison Table (Camera Artifacts Excluded)
    story.append(Paragraph("3. Personal Baseline (Median / MAD) vs. Session Behavioral Comparison", h2_style))
    base_table = report.get("baseline_comparison_table") or []
    if base_table:
        b_rows = [["Behavioral Metric", "Baseline Median", "Baseline MAD", "Session Mean", "Peak Value", "Peak |Z|"]]
        for item in base_table:
            peak_z = item.get("peak_absolute_z_score")
            clamped_peak_z = min(8.0, float(peak_z)) if isinstance(peak_z, (int, float)) else None
            b_rows.append([
                str(item.get("metric_name", "—")),
                f"{item['baseline_median']:.3f}" if isinstance(item.get("baseline_median"), (int, float)) else "—",
                f"{item['baseline_mad']:.3f}" if isinstance(item.get("baseline_mad"), (int, float)) else "—",
                f"{item['session_mean']:.3f}" if isinstance(item.get("session_mean"), (int, float)) else "—",
                f"{item['peak_anomaly_value']:.3f}" if isinstance(item.get("peak_anomaly_value"), (int, float)) else "—",
                f"{clamped_peak_z:.2f}" if clamped_peak_z is not None else "—",
            ])
        t_base = Table(b_rows, colWidths=[2.1 * inch, 1.0 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch])
        t_base.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 3.0),
            ])
        )
        story.append(t_base)

    # Section 3B (Part 4 Module 4 & 5): Topic-by-Topic Conversation & Behavioral Shift Matrix
    topic_matrix = report.get("topic_shift_matrix") or []
    if topic_matrix:
        story.append(Paragraph("3B. Topic-by-Topic Conversation &amp; Behavioral Shift Matrix", h2_style))
        cell_p_style = ParagraphStyle("TopicCell", parent=small_style, fontSize=7.2, leading=9.4, spaceAfter=0)
        cell_bold_blue = ParagraphStyle("TopicPhase", parent=cell_p_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#0284C7"))
        cell_italic = ParagraphStyle("TopicQuote", parent=cell_p_style, fontName="Helvetica-Oblique", textColor=colors.HexColor("#1E293B"))
        cell_verdict = ParagraphStyle("TopicVerdict", parent=cell_p_style, fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"))

        tm_rows: list[list[Any]] = [
            ["Time & Phase", "What Was Spoken (Topic)", "Voice & Vocal Emotion", "Heartbeat & Breathing", "Physical Actions", "Behavioral State"]
        ]
        for row in topic_matrix:
            tm_rows.append([
                Paragraph(_esc(row.get("phase_index", "")), cell_bold_blue),
                Paragraph(_esc(row.get("spoken_topic_excerpt", "")), cell_italic),
                Paragraph(_esc(row.get("voice_and_emotion", "")), cell_p_style),
                Paragraph(_esc(row.get("heartbeat_and_breathing", "")), cell_p_style),
                Paragraph(_esc(row.get("key_physical_actions", "")), cell_p_style),
                Paragraph(_esc(row.get("topic_state_verdict", "")), cell_verdict),
            ])
        t_topic = Table(
            tm_rows,
            colWidths=[0.95 * inch, 1.63 * inch, 1.22 * inch, 1.20 * inch, 1.05 * inch, 1.05 * inch],
            repeatRows=1,
        )
        t_topic.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7.0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 4.5),
            ])
        )
        story.append(t_topic)

    # Section 4: Discovered Personal Behavioral Motifs ("Tells")
    motifs = report.get("discovered_personal_motifs") or []
    if motifs:
        story.append(Paragraph("4. Recurring Personal Behavioral Habits ('Tells')", h2_style))
        for m in motifs:
            story.append(
                Paragraph(
                    f"&bull; <b>{_esc(m.get('motif_name'))}</b> ({_esc(m.get('modality'))}): "
                    f"{_esc(m.get('pattern_description'))} [Timestamps: {_esc(m.get('occurrence_timestamps_sec'))}]",
                    small_style,
                )
            )

    # Section 5 (Part 4 Module 5): Human-Analyst Moment Card with Side-by-Side Video Evidence Snapshot & 5-Channel Breakdown
    events = report.get("moment_by_moment_cards") or report.get("flagged_incongruence_events") or []
    story.append(Paragraph("5. Moment-by-Moment Behavioral Breakdown (Video Snapshot &amp; 5-Channel Human Analysis)", h2_style))
    base_hr = (report.get("dual_horizon_baseline") or {}).get("hr_bpm_pos_median")
    if not events:
        story.append(Paragraph("No high-divergence or Rule-of-3 multi-channel convergence events were flagged in this session.", small_style))
    else:
        for ev in events:
            ep_id = str(ev.get("episode_id") or ev.get("event_id") or "EPISODE").upper()
            fmt_time = ev.get("time_range") or ev.get("formatted_time") or f"{(ev.get('time_range_sec') or [0, 0])[0]:.1f}s - {(ev.get('time_range_sec') or [0, 0])[-1]:.1f}s"
            headline_lbl = ev.get("headline_state_label") or ev.get("event_category", "Focused Conversational Moment")
            conf_val = float(ev.get("calibrated_confidence") or ev.get("calibrated_epistemic_confidence") or ev.get("confidence_score") or 0.68)
            conf_pct = int(round(conf_val * 100))

            human_rows = HumanSignalTranslator.build_human_moment_breakdown(
                ev, baseline_hr=base_hr, resp_summary=resp_summary
            )
            what_spoken = ev.get("what_was_spoken") or human_rows["what_was_spoken"]
            voice_emotion = ev.get("voice_tone_and_emotion") or ev.get("voice_and_vocal_emotion") or human_rows["voice_tone_and_emotion"]
            phys_action = ev.get("physical_actions_observed") or ev.get("physical_actions_and_expression") or human_rows["physical_actions_observed"]
            hb_breath = ev.get("heartbeat_and_breathing") or human_rows["heartbeat_and_breathing"]
            moment_analysis = (
                ev.get("human_expert_moment_analysis")
                or ev.get("moment_analysis")
                or ev.get("epistemic_judge_verdict")
                or "Multimodal shift relative to personal baseline."
            )
            adv_vs_skep = ev.get("advocate_vs_skeptic_check") or (
                f"Advocate (S_adv={ev.get('advocate_strength_score', 0.82)}): {ev.get('advocate_case', '')} "
                f"Skeptic (S_skep={ev.get('skeptic_strength_score', 0.42)}): {ev.get('skeptic_case', '')}"
            )
            cites = ev.get("grounded_rag_citations") or ev.get("rag_citation_ids") or [
                c.get("doc_id") for c in (ev.get("rag_citations") or ev.get("citations") or []) if isinstance(c, dict)
            ]
            cite_str = ", ".join(str(c) for c in cites) if cites else "Tier-1/2/3 KB"

            # Header row inside the Moment Card
            hdr_left = Paragraph(
                f"<b>{ep_id} [{_esc(fmt_time)}] — {_esc(headline_lbl)}</b>",
                ParagraphStyle("CardHdrL", parent=body_style, fontName="Helvetica-Bold", fontSize=9.2, leading=11.5, textColor=colors.HexColor("#0F172A"), spaceAfter=0),
            )
            hdr_right = Paragraph(
                f"<b>Confidence: {conf_pct}%</b>",
                ParagraphStyle("CardHdrR", parent=small_style, fontName="Helvetica-Bold", fontSize=7.8, leading=10.0, alignment=2, textColor=colors.HexColor("#0369A1"), spaceAfter=0),
            )
            hdr_table = Table([[hdr_left, hdr_right]], colWidths=[5.45 * inch, 1.40 * inch])
            hdr_table.setStyle(
                TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )

            # Left column: Annotated Video Evidence Snapshot + Apex Frame Caption
            left_col_flow: list[Any] = []
            b64_img = ev.get("image_data_uri") or ev.get("keyframe_jpeg_base64")
            apex_ts = ev.get("apex_timestamp_sec") or fmt_time.split("-")[0].strip()
            act_lbl = ev.get("action_label") or "Peak Behavioral Shift"
            if isinstance(b64_img, str) and len(b64_img) > 100:
                try:
                    raw_img = base64.b64decode(b64_img.split(",")[-1])
                    rl_img = RLImage(BytesIO(raw_img), width=2.02 * inch, height=1.26 * inch)
                    left_col_flow.append(rl_img)
                except Exception:
                    pass
            left_col_flow.append(
                Paragraph(
                    f"<b>Evidence Frame @ {_esc(apex_ts)}</b><br/>({_esc(act_lbl)})",
                    ParagraphStyle("CapP", parent=small_style, fontSize=7.0, leading=9.0, alignment=1, textColor=colors.HexColor("#475569"), spaceBefore=3, spaceAfter=0),
                )
            )

            # Right column: 5-Channel Human Behavioral Breakdown
            ch_style = ParagraphStyle("ChP", parent=small_style, fontSize=7.8, leading=10.6, textColor=colors.HexColor("#1E293B"), spaceAfter=2.5)
            analysis_box = Table(
                [[
                    Paragraph(
                        f"<b>Human Expert Analysis:</b> {_esc(moment_analysis)}",
                        ParagraphStyle("ExpBox", parent=small_style, fontSize=7.8, leading=10.6, textColor=colors.HexColor("#0F172A"), spaceAfter=0),
                    )
                ]],
                colWidths=[4.62 * inch],
            )
            analysis_box.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                    ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor("#F59E0B")),
                    ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#E2E8F0")),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ])
            )

            foot_p = Paragraph(
                f"<b>Advocate vs. Skeptic Check:</b> {_esc(adv_vs_skep)} &nbsp;|&nbsp; <b>Citations:</b> {_esc(cite_str)}",
                ParagraphStyle("FootP", parent=small_style, fontSize=6.9, leading=9.2, textColor=colors.HexColor("#64748B"), spaceBefore=3, spaceAfter=0),
            )

            right_col_flow: list[Any] = [
                Paragraph(f"<b>Spoken Words:</b> <i>{_esc(what_spoken)}</i>", ch_style),
                Paragraph(f"<b>Voice Tone &amp; Emotion:</b> {_esc(voice_emotion)}", ch_style),
                Paragraph(f"<b>Physical Action (Video):</b> {_esc(phys_action)}", ch_style),
                Paragraph(f"<b>Heartbeat &amp; Breathing:</b> {_esc(hb_breath)}", ch_style),
                Spacer(1, 2),
                analysis_box,
                foot_p,
            ]

            probe = ev.get("live_copilot_probe")
            if isinstance(probe, dict) and probe.get("suggested_follow_up_probe"):
                right_col_flow.append(
                    Paragraph(
                        f"<b>[LIVE CO-PILOT] {_esc(probe.get('alert_banner', 'Live Behavioral Alert'))}:</b> "
                        f"&ldquo;{_esc(probe.get('suggested_follow_up_probe'))}&rdquo;",
                        copilot_style,
                    )
                )

            body_two_col = Table(
                [[left_col_flow, right_col_flow]],
                colWidths=[2.15 * inch, 4.70 * inch],
            )
            body_two_col.setStyle(
                TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, -1), 8),
                    ("RIGHTPADDING", (1, 0), (1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ])
            )

            outer_card = Table(
                [[[hdr_table, body_two_col]]],
                colWidths=[7.10 * inch],
            )
            outer_card.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                    ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor("#0284C7")),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ])
            )

            story.append(KeepTogether([outer_card, Spacer(1, 6)]))

    # Section 6: Compact Scannable Telemetry Matrix
    story.append(Paragraph(f"6. Synchronized Telemetry Windows ({len(windows)} Windows Recorded · 0.0s–{report.get('duration_sec', 0)}s)", h2_style))
    if not windows:
        story.append(Paragraph("No measurements were supplied. No signal estimates are available.", small_style))
    else:
        w_rows = [["Window", "Turn Phase", "Voice Emotion", "Voiced F0 / Jitter", "Resp / HR", "Spoken Words & Active Behavioral Cues"]]
        resp_bpm = resp_summary.get("respiration_rate_breaths_per_min")
        resp_str = f"{resp_bpm} br/m" if resp_bpm is not None else "—"
        for window in windows:
            metrics = window.get("metrics") or {}
            hr = metrics.get("hr_bpm", metrics.get("hr_bpm_pos"))
            hr_text = "unavail" if hr is None else f"{hr} BPM"
            f0 = metrics.get("f0_semitone_mean")
            jit = metrics.get("jitter_local_mean", metrics.get("jitter_local"))
            f0_str = f"{float(f0):.1f}st" if isinstance(f0, (int, float)) and float(f0) > 0 else "silent"
            jit_str = f"{float(jit):.3f}" if isinstance(jit, (int, float)) and float(jit) > 0 else "—"
            dom_em = str(window.get("dominant_vocal_emotion") or ("Calm & Assured" if f0_str != "silent" else "Silent"))
            flags = [
                f.replace("SIGNIFICANT_DEVIATION_", "SIG_").replace("MODERATE_DEVIATION_", "MOD_")
                for f in (window.get("behavioral_flags") or [])
                if "illumination_change" not in f and "head_velocity" not in f
            ]
            flag_short = ", ".join(flags[:3]) or "baseline"
            tr_txt = str(window.get("transcript_segment") or "").strip()
            summary_cell = (f"<b>“{_esc(tr_txt)}”</b> · " if tr_txt else "") + _esc(flag_short)
            w_rows.append([
                f"{window.get('start_sec', 0):.1f}–{window.get('end_sec', 0):.1f}s",
                str(window.get("turn_phase") or "RECORDING").replace("POST_UTTERANCE_LEAK", "POST_UTTERANCE").replace("STIMULUS_LISTENING", "LISTENING"),
                dom_em.replace(" / Listening Pause", ""),
                f"{f0_str} / {jit_str}",
                f"{resp_str} / {hr_text}",
                Paragraph(summary_cell, ParagraphStyle("CellP", parent=small_style, fontSize=6.8, leading=8.5, spaceAfter=0)),
            ])
        t_win = Table(w_rows, colWidths=[0.80 * inch, 0.98 * inch, 1.05 * inch, 0.85 * inch, 0.85 * inch, 2.57 * inch], repeatRows=1)
        t_win.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("PADDING", (0, 0), (-1, -1), 2.5),
            ])
        )
        story.append(t_win)

    # Section 7: Final Comprehensive Behavioral Verdict (Complete Session Synthesis)
    verdict_block = report.get("final_overall_verdict") or report.get("final_comprehensive_verdict") or {}
    if verdict_block:
        arc_txt = verdict_block.get("conversation_story_arc") or verdict_block.get("overall_behavioral_and_emotional_arc", "")
        align_txt = verdict_block.get("words_vs_body_alignment_verdict") or verdict_block.get("words_vs_voice_and_body_alignment", "")
        hb_txt = verdict_block.get("heartbeat_and_breathing_story") or verdict_block.get("cardiovascular_and_breathing_summary", "")
        takeaways_list = verdict_block.get("top_3_personal_habits_and_takeaways") or verdict_block.get("key_actionable_takeaways") or []
        verdict_flow: list[Any] = [
            Paragraph("7. Final Comprehensive Behavioral Verdict (Complete Session Synthesis)", h2_style),
            Paragraph(
                f"<b>1. Conversation Story Arc (Overall Behavioral &amp; Emotional Arc):</b><br/>{_esc(arc_txt)}",
                verdict_box_style,
            ),
            Paragraph(
                f"<b>2. Words vs. Voice &amp; Body Alignment Verdict:</b><br/>{_esc(align_txt)}",
                verdict_box_style,
            ),
            Paragraph(
                f"<b>3. Heartbeat &amp; Breathing Story:</b><br/>{_esc(hb_txt)}",
                verdict_box_style,
            ),
            Paragraph("<b>4. Top 3 Personal Habits &amp; Actionable Takeaways:</b>", body_style),
        ]
        for idx, bullet in enumerate(takeaways_list, 1):
            verdict_flow.append(Paragraph(f"{idx}. {_esc(bullet)}", small_style))
        story.append(KeepTogether(verdict_flow))

    doc.build(story)
    return output.getvalue()
