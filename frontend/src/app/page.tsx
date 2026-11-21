import Link from "next/link";
import { Badge, Icon, PageHeading } from "@/components/shell";
import { WorkspaceStats } from "@/components/workspace-stats";
import { RecentSessions } from "@/components/session-list";

export default function Home() {
  return (
    <>
      <PageHeading
        eyebrow="OMNIPULSE MBS v2.0 · COGNITIVE & DYADIC TELEMETRY LAB"
        title="Multimodal behavioral & physiological signal studio"
        description="Baseline-relative 30 FPS MediaPipe FACS/Ocular/Kinesics, POS+CHROM rPPG cardiovascular pulse, openSMILE eGeMAPSv02 acoustics, and 3-Tier Stratified RAG synthesis."
        action={
          <div style={{ display: "flex", gap: 8 }}>
            <Link className="button button-secondary" href="/knowledge-base">
              <Icon name="book" size={14} /> 3-Tier Knowledge Base
            </Link>
            <Link className="button button-primary" href="/sessions/new">
              <Icon name="scan" size={15} /> New Session
            </Link>
          </div>
        }
      />

      <section className="overview-hero">
        <div className="hero-copy">
          <div className="hero-kicker">
            <i /> DUAL-HORIZON BASELINE · 100MS MICRO-SEQUENCES · ADVOCATE-SKEPTIC DEBATE
          </div>
          <h2>
            Capture real multimodal signals.
            <br />
            Synthesize them with scientific rigor.
          </h2>
          <p>
            Run a 90-second personal baseline + retroactive resting floor calibration, stream 2.0-second trimodal
            windows with 100ms sub-second micro-sequence encoding, and generate forensic reports grounded in
            peer-reviewed literature.
          </p>
          <div className="hero-actions">
            <Link href="/sessions/new" className="button button-primary">
              Launch Calibration Wizard <Icon name="arrow" size={14} />
            </Link>
            <Link href="/sessions" className="button button-secondary">
              Browse Session Archive
            </Link>
          </div>
        </div>

        <div className="hero-graphic" aria-hidden="true">
          <div className="hero-glow" />
          <svg className="hero-signal" viewBox="0 0 420 120">
            <circle className="hero-orbit" cx="310" cy="60" r="52" />
            <circle className="hero-orbit-2" cx="310" cy="60" r="76" />
            <path
              className="hero-line-soft"
              d="M 10 75 Q 55 40, 100 72 T 190 58 T 280 70 T 370 48 L 410 55"
            />
            <path
              className="hero-line"
              d="M 10 62 L 65 62 L 82 34 L 98 92 L 114 48 L 130 62 L 205 62 L 222 26 L 238 96 L 254 50 L 272 62 L 410 62"
            />
            <circle className="hero-point" cx="82" cy="34" r="3.5" />
            <circle className="hero-point" cx="222" cy="26" r="4" />
          </svg>
        </div>
      </section>

      <WorkspaceStats />

      <div className="grid overview-columns" style={{ marginBottom: 18 }}>
        <div className="card">
          <div className="card-heading">
            <div>
              <h2 className="card-title">OmniPulse v2.0 Cognitive &amp; Dyadic Architecture</h2>
              <div className="card-subtitle">
                All 7 specification upgrades active across browser capture and backend synthesis
              </div>
            </div>
            <Badge tone="green">7/7 UPGRADES ACTIVE</Badge>
          </div>
          <div className="grid" style={{ gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10 }}>
            <div className="metric-card">
              <div className="metric-label">
                <span>01 · DUAL-HORIZON BASELINE</span>
                <Badge tone="blue">HORIZON A + B</Badge>
              </div>
              <div className="metric-help" style={{ marginTop: 6 }}>
                Combines retroactive 90s resting floor (S*) with local 15s pre-stimulus phasic delta (ΔZ_phasic) to
                separate acute reactions from carryover rumination.
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">
                <span>02 · 100MS MICRO-SEQUENCES</span>
                <Badge tone="green">10 HZ SLICES</Badge>
              </div>
              <div className="metric-help" style={{ marginTop: 6 }}>
                Tracks sub-second onset ordering (e.g. Gaze Shift → Lip Press → Masked Smile) and Smile Snap-Off
                velocity (v_drop &gt; 1.80 s⁻¹).
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">
                <span>03 · ADVOCATE vs. SKEPTIC ToM</span>
                <Badge tone="amber">3-PERSONA DEBATE</Badge>
              </div>
              <div className="metric-help" style={{ marginTop: 6 }}>
                Every anomaly is debated by a Signal Advocate, a Pragmatic/L2 Cultural Skeptic, and an Epistemic Judge
                with calibrated confidence.
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">
                <span>04 · PERSONAL MOTIFS &amp; ACI</span>
                <Badge tone="neutral">r_pb ≥ 0.38 · RR ≥ 2.5</Badge>
              </div>
              <div className="metric-help" style={{ marginTop: 6 }}>
                Unsupervised Point-Biserial mining of speaker-specific behavioral habits plus dyadic Acoustic
                Convergence Index (ACI).
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-heading">
            <div>
              <h2 className="card-title">Scientific Integrity &amp; Anti-Fabrication Gate</h2>
              <div className="card-subtitle">Strict mathematical invariants enforced at runtime</div>
            </div>
            <Icon name="shield" size={16} />
          </div>
          <div className="activity-list">
            <div className="activity-row">
              <div className="activity-icon">
                <Icon name="pulse" size={15} />
              </div>
              <div className="activity-copy">
                <b>POS + CHROM rPPG Cross-Check</b>
                <small>Masks HR to null when SQI &lt; 0.38, |POS−CHROM| &gt; 12 BPM, or head velocity &gt; 25°/s</small>
              </div>
            </div>
            <div className="activity-row">
              <div className="activity-icon blue">
                <Icon name="eye" size={15} />
              </div>
              <div className="activity-copy">
                <b>3D Head Pose Reliability Gate</b>
                <small>Suspends facial Z-score flagging when |yaw| &gt; 35° or |pitch| &gt; 25°</small>
              </div>
            </div>
            <div className="activity-row">
              <div className="activity-icon amber">
                <Icon name="book" size={15} />
              </div>
              <div className="activity-copy">
                <b>Stratified 4-Chunk RAG Quota</b>
                <small>1× Tier-1 Canonical + 1× Tier-2 Matrix (648 perms) + 2× Tier-3 Empirical Studies</small>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="section-heading">
        <h2>Recent sessions</h2>
        <Link href="/sessions" className="button-link">
          All sessions <Icon name="arrow" size={13} />
        </Link>
      </div>
      <RecentSessions limit={6} />

      <div className="sample-banner" style={{ marginTop: 14 }}>
        <Icon name="shield" size={14} />
        <span>
          <b>Zero Synthetic Data Policy:</b> Signal values are never populated with mock examples. Reports display only
          real measurements captured from your session or verified processor ingestion.
        </span>
      </div>
    </>
  );
}
