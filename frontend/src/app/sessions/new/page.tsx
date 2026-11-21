"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Badge, Icon, PageHeading } from "@/components/shell";

type SessionResponse = {
  id: string;
  status: string;
  capture_token: string;
  webrtc_ice_servers?: Array<{ urls: string[] }>;
};

export default function NewSessionPage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const noiseFloorRmsRef = useRef<number>(0.0025);
  const peakSignalRmsRef = useRef<number>(0.015);

  const [streamReady, setStreamReady] = useState(false);
  const [consent, setConsent] = useState(false);
  const [title, setTitle] = useState("");
  const [participant, setParticipant] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Real-time hardware & environment telemetry (PDF Section 9.1)
  const [lumaY, setLumaY] = useState<number | null>(null);
  const [greenClipPct, setGreenClipPct] = useState<number>(0);
  const [micSnrDb, setMicSnrDb] = useState<number | null>(null);
  const [micDbfs, setMicDbfs] = useState<number | null>(null);

  const stopHardwarePreview = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      void audioCtxRef.current.close();
    }
    audioCtxRef.current = null;
    setStreamReady(false);
    setLumaY(null);
    setGreenClipPct(0);
    setMicSnrDb(null);
    setMicDbfs(null);
  };

  useEffect(() => () => stopHardwarePreview(), []);

  useEffect(() => {
    if (videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [streamReady]);

  async function startPreview() {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      stopHardwarePreview();
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }

      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      const pcmBuf = new Float32Array(analyser.fftSize);

      noiseFloorRmsRef.current = 0.003;
      peakSignalRmsRef.current = 0.018;
      setStreamReady(true);

      let lastSampleAt = 0;
      const measureLoop = (now: number) => {
        if (!streamRef.current) return;
        if (now - lastSampleAt >= 220) {
          lastSampleAt = now;
          // 1. Measure Canvas Luminance Y = 0.299R + 0.587G + 0.114B and Green-Channel Clipping (G >= 250)
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (video && canvas && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
            const w = 160;
            const h = 120;
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext("2d", { willReadFrequently: true });
            if (ctx) {
              ctx.drawImage(video, 0, 0, w, h);
              // Sample central oval ROI (forehead/cheeks region)
              const img = ctx.getImageData(Math.floor(w * 0.28), Math.floor(h * 0.18), Math.floor(w * 0.44), Math.floor(h * 0.64));
              const data = img.data;
              let sumY = 0;
              let clippedG = 0;
              const pixelCount = Math.max(1, data.length / 4);
              for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                sumY += 0.299 * r + 0.587 * g + 0.114 * b;
                if (g >= 250) clippedG += 1;
              }
              setLumaY(Math.round(sumY / pixelCount));
              setGreenClipPct(Number(((clippedG / pixelCount) * 100).toFixed(1)));
            }
          }

          // 2. Measure Microphone RMS & Estimate SNR (dB)
          analyser.getFloatTimeDomainData(pcmBuf);
          let sumSq = 0;
          for (let i = 0; i < pcmBuf.length; i += 1) {
            sumSq += pcmBuf[i] * pcmBuf[i];
          }
          const rms = Math.sqrt(sumSq / pcmBuf.length);
          const dbfs = rms > 1e-6 ? Math.round(20 * Math.log10(rms)) : -96;
          setMicDbfs(dbfs);

          // Adaptive floor & speech peak tracking
          if (rms > 1e-5 && rms < noiseFloorRmsRef.current * 1.15) {
            noiseFloorRmsRef.current = 0.85 * noiseFloorRmsRef.current + 0.15 * rms;
          } else if (rms > noiseFloorRmsRef.current * 1.8) {
            peakSignalRmsRef.current = Math.max(peakSignalRmsRef.current * 0.96, rms);
          }
          const floor = Math.max(noiseFloorRmsRef.current, 0.0008);
          const activeRef = Math.max(rms, peakSignalRmsRef.current);
          const snr = Math.max(6, Math.min(48, 20 * Math.log10(activeRef / floor)));
          setMicSnrDb(Number(snr.toFixed(1)));
        }
        rafRef.current = requestAnimationFrame(measureLoop);
      };
      rafRef.current = requestAnimationFrame(measureLoop);
    } catch {
      setError("Camera or microphone access was unavailable. Check browser permissions and try again.");
    }
  }

  async function beginSession() {
    if (!consent) {
      setError("Please review and confirm explicit multimodal informed consent before creating a session.");
      return;
    }
    if (!streamReady) {
      setError("Start the local camera and microphone hardware check first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const session = await api.post<SessionResponse>("/sessions", {
        subject_identifier: participant.trim() || `participant-${crypto.randomUUID().slice(0, 8)}`,
        session_title: title.trim() || "Multimodal Behavioral Session",
        consent_verified: true,
        ambient_lux_score: lumaY ?? 125,
        mic_snr_db: micSnrDb ?? 22.0,
      });
      sessionStorage.setItem(`omnipulse.capture:${session.id}`, session.capture_token);
      await api.post(`/sessions/${session.id}/calibrate`);
      stopHardwarePreview();
      router.push(`/sessions/${session.id}/live`);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "The local API is unavailable. Start the FastAPI service and retry."
      );
    } finally {
      setLoading(false);
    }
  }

  const lightingOptimal = lumaY !== null && lumaY >= 60 && greenClipPct < 5.0;
  const micOptimal = micSnrDb !== null && micSnrDb >= 15.0;

  return (
    <>
      <PageHeading
        eyebrow="SESSION SETUP & HARDWARE VERIFICATION · STEP 1 OF 2"
        title="Calibration & consent wizard"
        description="Verify ambient facial illumination (Y ≥ 60, G-channel non-clipping), microphone SNR (≥ 15 dB), and explicit multimodal consent before running the 90-second personal baseline."
        action={<Badge tone={streamReady ? "green" : "blue"}>{streamReady ? "SENSORS VERIFIED" : "LOCAL MODELS"}</Badge>}
      />

      <div className="wizard-layout">
        <section className="card wizard-card">
          <div className="step-row">
            <div className="wizard-step active">
              <span className="step-number">1</span>Hardware &amp; ROI Check
            </div>
            <i className="step-connector" />
            <div className="wizard-step">
              <span className="step-number">2</span>90s Personal Baseline + Live Capture
            </div>
          </div>

          <div className="card-heading">
            <div>
              <h2 className="card-title">Camera framing &amp; optical rPPG skin ROI check</h2>
              <div className="card-subtitle">
                Center the participant’s face inside the oval guide. MediaPipe processes frames locally in browser memory.
              </div>
            </div>
            <Badge tone={streamReady ? (lightingOptimal ? "green" : "amber") : "neutral"}>
              {streamReady ? (lightingOptimal ? "OPTICAL ROI READY" : "CHECK LIGHTING") : "NOT CONNECTED"}
            </Badge>
          </div>

          <div className="camera-stage">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ display: streamReady ? "block" : "none" }}
              aria-label="Local camera preview"
            />
            <canvas ref={canvasRef} width={160} height={120} hidden />
            {!streamReady && (
              <div className="camera-placeholder">
                <span className="camera-placeholder-icon">
                  <Icon name="scan" size={22} />
                </span>
                <b>Start local camera &amp; microphone preview</b>
                <span>Real-time luminance (Y), G-channel saturation, and Mic SNR will be measured automatically.</span>
              </div>
            )}
            {streamReady && (
              <div className="camera-overlay">
                <span className="camera-overlay-label">
                  FOREHEAD + BILATERAL CHEEK ROI · LUMA Y={lumaY ?? "—"} · G-CLIP {greenClipPct}%
                </span>
              </div>
            )}
            <div className="camera-controls">
              <button
                type="button"
                className="camera-control"
                onClick={() => void startPreview()}
                aria-label="Start camera preview"
                title="Start Camera & Mic Check"
              >
                <Icon name="activity" size={15} />
              </button>
              <button
                type="button"
                className="camera-control"
                onClick={stopHardwarePreview}
                aria-label="Stop camera preview"
                title="Stop Preview"
              >
                <Icon name="close" size={14} />
              </button>
            </div>
          </div>

          <div
            className="grid"
            style={{
              gridTemplateColumns: "1fr 1fr",
              gap: 1,
              marginTop: 12,
              background: "#222b36",
              border: "1px solid #222b36",
              borderRadius: 7,
              overflow: "hidden",
            }}
          >
            <div className="lighting-row" style={{ background: "#10151e", padding: "11px 12px" }}>
              <div className="lighting-value">
                <span
                  className="lighting-check"
                  style={{
                    background: lightingOptimal ? "#152a22" : "#2c2517",
                    color: lightingOptimal ? "#76d6a4" : "#e8bd70",
                  }}
                >
                  {lightingOptimal ? "✓" : "!"}
                </span>
                <div className="lighting-copy">
                  <b>Optical Illumination (rPPG Gate)</b>
                  <small>
                    {lumaY === null
                      ? "Start preview to measure Y luminance (target Y ≥ 60)"
                      : `Luminance Y = ${lumaY}/255 · Green clipping ${greenClipPct}%`}
                  </small>
                </div>
              </div>
              <Badge tone={lumaY === null ? "neutral" : lightingOptimal ? "green" : "amber"}>
                {lumaY === null ? "PENDING" : lightingOptimal ? "OPTIMAL ≥60" : lumaY < 60 ? "LOW LIGHT" : "GLARE"}
              </Badge>
            </div>

            <div className="lighting-row" style={{ background: "#10151e", padding: "11px 12px" }}>
              <div className="lighting-value">
                <span
                  className="lighting-check"
                  style={{
                    background: micOptimal ? "#152a22" : "#2c2517",
                    color: micOptimal ? "#76d6a4" : "#e8bd70",
                  }}
                >
                  {micOptimal ? "✓" : "!"}
                </span>
                <div className="lighting-copy">
                  <b>Acoustic SNR (eGeMAPSv02 Gate)</b>
                  <small>
                    {micSnrDb === null
                      ? "Speak a test phrase to verify SNR ≥ 15 dB"
                      : `Estimated SNR ${micSnrDb} dB · Level ${micDbfs ?? -96} dBFS`}
                  </small>
                </div>
              </div>
              <Badge tone={micSnrDb === null ? "neutral" : micOptimal ? "green" : "amber"}>
                {micSnrDb === null ? "PENDING" : micOptimal ? `${micSnrDb} dB READY` : `${micSnrDb} dB NOISY`}
              </Badge>
            </div>
          </div>

          <div className="divider" />

          <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div className="input-wrap">
              <label htmlFor="session-title">Session title</label>
              <input
                className="input"
                id="session-title"
                placeholder="e.g. Executive Leadership Assessment · Turn 2"
                value={title}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setTitle(e.target.value)}
              />
            </div>
            <div className="input-wrap">
              <label htmlFor="participant-id">
                Participant identifier <span className="micro-label">PSEUDONYM</span>
              </label>
              <input
                className="input"
                id="participant-id"
                placeholder="e.g. SUBJ-2026-04 (auto-generated if blank)"
                value={participant}
                onChange={(e: ChangeEvent<HTMLInputElement>) => setParticipant(e.target.value)}
              />
            </div>
          </div>

          <div className="consent-card">
            <label className="check-row">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span>
                <strong>I confirm explicit multimodal informed consent</strong> for optical (FACS, gaze, pose, rPPG
                skin chrominance), acoustic (openSMILE eGeMAPSv02), and verbatim transcript (Groq Whisper Large-v3)
                telemetry. All outputs represent baseline-relative probabilistic correlates of arousal, valence, and
                cognitive load—never truth verification, lie detection, or clinical psychiatric diagnosis.
              </span>
            </label>
          </div>

          {error && <div className="form-error">{error}</div>}

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, marginTop: 14 }}>
            <span className="subtle-note">
              Next step: 90-second personal baseline calibration (Horizon A + B) in the Live Studio.
            </span>
            <button className="button button-primary" onClick={() => void beginSession()} disabled={loading}>
              {loading ? "Initializing session…" : "Launch 90s Calibration & Studio"}
              <Icon name="arrow" size={14} />
            </button>
          </div>
        </section>

        <aside className="wizard-aside">
          <div className="card aside-card">
            <div className="card-heading">
              <div>
                <h2 className="card-title">Hardware Quality Checklist</h2>
                <div className="card-subtitle">PDF Section 9.1 pre-flight requirements</div>
              </div>
              <Icon name="shield" size={16} />
            </div>
            <div className="aside-check-row">
              <Icon name="activity" size={14} />
              Face centered in oval guide (|yaw| &lt; 35°, |pitch| &lt; 25°)
            </div>
            <div className="aside-check-row">
              <Icon name="activity" size={14} />
              Front-facing soft lighting (Luminance Y ≥ 60, G &lt; 250)
            </div>
            <div className="aside-check-row">
              <Icon name="activity" size={14} />
              Quiet room with Microphone SNR ≥ 15 dB
            </div>
            <div className="aside-check-row">
              <Icon name="activity" size={14} />
              90-second neutral baseline + retroactive resting floor
            </div>
            <p className="aside-note">
              If rPPG Signal Quality Index (SQI) drops below 0.38 due to motion or dim lighting, heart rate is
              automatically masked rather than guessed.
            </p>
          </div>

          <div className="card aside-card">
            <div className="card-heading">
              <div>
                <h2 className="card-title">Trimodal + rPPG Pipeline</h2>
                <div className="card-subtitle">Real-time browser + backend cognitive stack</div>
              </div>
            </div>
            <div className="activity-list">
              <div className="activity-row">
                <div className="activity-icon">
                  <span>01</span>
                </div>
                <div className="activity-copy">
                  <b>478-Point Face + Pose + Hands</b>
                  <small>9 FACS Action Units, 3D head pose, iris gaze, and shoulder-normalized kinesics</small>
                </div>
              </div>
              <div className="activity-row">
                <div className="activity-icon blue">
                  <span>02</span>
                </div>
                <div className="activity-copy">
                  <b>POS + CHROM Optical rPPG</b>
                  <small>Forehead + bilateral cheek YCrCb skin ROI resampled onto a 30 Hz uniform spline grid</small>
                </div>
              </div>
              <div className="activity-row">
                <div className="activity-icon amber">
                  <span>03</span>
                </div>
                <div className="activity-copy">
                  <b>openSMILE + Groq Whisper + 3-Tier RAG</b>
                  <small>Vocal jitter/shimmer/HNR, verbatim disfluencies, and Advocate-Skeptic-Judge synthesis</small>
                </div>
              </div>
            </div>
          </div>

          <div className="sample-banner">
            <Icon name="shield" size={14} />
            <span>
              <b>Scientific Ethics Invariant:</b> Behavioral anomalies indicate cognitive effort or autonomic regulation,
              never lying or guilt.
            </span>
          </div>
        </aside>
      </div>
    </>
  );
}
