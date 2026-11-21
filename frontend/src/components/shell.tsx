"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { api } from "@/lib/api";

const sections = [
  {
    title: "TELEMETRY LAB",
    links: [
      { href: "/", label: "Overview", icon: "grid" },
      { href: "/sessions/new", label: "New session", icon: "scan" },
      { href: "/sessions", label: "Sessions", icon: "chart" },
    ],
  },
  {
    title: "SCIENTIFIC RAG",
    links: [{ href: "/knowledge-base", label: "Knowledge base", icon: "book" }],
  },
];

export type HealthPayload = {
  status: string;
  version?: string;
  cognitive_v2_modules_active?: boolean;
  gemini_report_configured?: boolean;
  groq_asr_configured?: boolean;
  audio_inference?: {
    opensmile_installed?: boolean;
    dsp_fallback_available?: boolean;
    groq_sdk_installed?: boolean;
    groq_api_configured?: boolean;
    groq_model?: string;
  };
};

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.75,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };
  const paths: Record<string, ReactNode> = {
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    scan: (
      <>
        <path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" />
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v2m0 16v2M2 12h2m16 0h2" />
      </>
    ),
    chart: (
      <>
        <path d="M4 19V5m0 14h17" />
        <path d="m7 15 4-4 3 2 5-7" />
        <circle cx="19" cy="6" r="1" />
      </>
    ),
    book: (
      <>
        <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21z" />
        <path d="M4 5.5v13A2.5 2.5 0 0 1 6.5 16H20M8 7h8m-8 3h6" />
      </>
    ),
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    pulse: <path d="M3 12h4l2-7 4 14 2-7h6" />,
    shield: (
      <>
        <path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4m-5 5 5-5 5 5" />
        <path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" />
      </>
    ),
    download: (
      <>
        <path d="M12 4v12m-5-5 5 5 5-5" />
        <path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" />
      </>
    ),
    close: <path d="m18 6-12 12M6 6l12 12" />,
    activity: <path d="M3 12h4l3-8 4 16 3-8h4" />,
    spark: (
      <>
        <path d="M12 3v4m0 10v4M3 12h4m10 0h4" />
        <circle cx="12" cy="12" r="3" />
      </>
    ),
    eye: (
      <>
        <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z" />
        <circle cx="12" cy="12" r="2.5" />
      </>
    ),
  };
  return (
    <svg {...common} aria-hidden="true">
      {paths[name] || paths.activity}
    </svg>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [apiStatus, setApiStatus] = useState<"checking" | "connected" | "degraded" | "offline">("checking");
  const [apiLabel, setApiLabel] = useState("CHECKING API");

  useEffect(() => {
    const check = () => {
      api
        .get<HealthPayload>("/health")
        .then((health) => {
          if (health.status !== "ok") {
            setApiStatus("offline");
            setApiLabel("API OFFLINE");
            return;
          }
          const acousticReady = Boolean(
            health.audio_inference?.opensmile_installed || health.audio_inference?.dsp_fallback_available
          );
          if (acousticReady && health.cognitive_v2_modules_active) {
            setApiStatus("connected");
            setApiLabel(
              health.groq_asr_configured
                ? "v2.0 COGNITIVE + GROQ ASR READY"
                : "v2.0 COGNITIVE ENGINE READY"
            );
          } else if (acousticReady) {
            setApiStatus("connected");
            setApiLabel("API + DSP AUDIO READY");
          } else {
            setApiStatus("degraded");
            setApiLabel("AUDIO CONFIG REQUIRED");
          }
        })
        .catch(() => {
          setApiStatus("offline");
          setApiLabel("API OFFLINE");
        });
    };
    check();
    const timer = window.setInterval(check, 12000);
    return () => window.clearInterval(timer);
  }, []);

  const crumbName =
    pathname === "/"
      ? "Overview"
      : pathname.includes("knowledge")
        ? "3-Tier Knowledge Base"
        : pathname.includes("new")
          ? "Calibration & Setup"
          : pathname.includes("live")
            ? "Live Telemetry Studio"
            : pathname.includes("report")
              ? "Forensic Cognitive Report"
              : "Sessions";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <Link href="/" className="brand">
          <span className="brand-mark">
            <Icon name="pulse" size={20} />
          </span>
          <span className="brand-copy">
            <strong>omnipulse</strong>
            <small>MBS v2.0 · COGNITIVE LAB</small>
          </span>
        </Link>

        <div className="workspace-switch">
          <span className="workspace-avatar">OP</span>
          <span>
            <strong>Trimodal + rPPG Lab</strong>
            <small>Dual-Horizon &amp; 3-Tier RAG</small>
          </span>
          <span className="switch-chevron">⌄</span>
        </div>

        {sections.map((section) => (
          <div className="nav-section" key={section.title}>
            <div className="nav-label">{section.title}</div>
            {section.links.map((link) => {
              const active =
                link.href === "/"
                  ? pathname === "/"
                  : link.href === "/sessions"
                    ? pathname === "/sessions" ||
                      (pathname.startsWith("/sessions/") && !pathname.startsWith("/sessions/new"))
                    : pathname === link.href;
              return (
                <Link href={link.href} className={`nav-link ${active ? "active" : ""}`} key={link.href}>
                  <Icon name={link.icon} />
                  <span>{link.label}</span>
                  {link.label === "New session" && <span className="nav-shortcut">LIVE</span>}
                </Link>
              );
            })}
          </div>
        ))}

        <div className="sidebar-spacer" />

        <div className="privacy-card">
          <div className="privacy-icon">
            <Icon name="shield" size={16} />
          </div>
          <div>
            <b>Scientific &amp; Privacy Guardrails</b>
            <p>
              478-point Face, Pose &amp; Hand landmarks run in-browser. Missing rPPG signals are never fabricated;
              behavioral shifts are never equated with deception.
            </p>
          </div>
        </div>

        <div className="profile-row">
          <span className="profile-avatar">v2</span>
          <span className="profile-name">
            <b>Cognitive &amp; Dyadic Engine</b>
            <small>Local SQLite + Qdrant Hybrid</small>
          </span>
          <span className="profile-menu">●</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>OmniPulse MBS</span>
            <span className="crumb-slash">/</span>
            <b>{crumbName}</b>
          </div>
          <div className="topbar-actions">
            <span className="environment-pill">
              <i
                style={{
                  background:
                    apiStatus === "connected"
                      ? "#6ed2a3"
                      : apiStatus === "offline"
                        ? "#ee7585"
                        : "#e4b45a",
                }}
              />
              {apiLabel}
            </span>
            <span className="topbar-divider" />
            <Link href="/sessions/new" className="button button-ghost" style={{ minHeight: 28, padding: "0 10px" }}>
              <Icon name="scan" size={13} /> New Capture
            </Link>
          </div>
        </header>
        <div className="page-wrap">{children}</div>
      </main>
    </div>
  );
}

export function PageHeading({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {action && <div className="heading-action">{action}</div>}
    </div>
  );
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: string }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
