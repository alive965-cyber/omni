"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Badge, Icon, PageHeading } from "@/components/shell";

const CALIBRATION_SEC = 90;
const VISION_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";
const FACE_MODEL = "/models/face_landmarker.task";
const POSE_MODEL = "/models/pose_landmarker.task";
const HAND_MODEL = "/models/hand_landmarker.task";

const BASELINE_FEATURES = new Set([
  "hr_bpm",
  "f0_semitone_mean",
  "f0_semitone_std",
  "jitter_local_mean",
  "shimmer_local_db_mean",
  "hnr_db_mean",
  "loudness_mean",
  "au01_inner_brow",
  "au02_outer_brow",
  "au04_brow_lowerer",
  "au06_cheek_raiser",
  "au12_lip_corner_puller",
  "au15_lip_corner_depressor",
  "au20_lip_stretcher",
  "au24_lip_presser",
  "au25_jaw_drop",
  "facial_asymmetry_index",
  "self_touch_ratio",
  "hand_kinetic_energy",
  "postural_lean_z",
  "blink_rate_per_min",
  "blink_duration_ms_mean",
  "gaze_aversion_ratio",
  "saccade_velocity_deg_s",
  "speech_rate_wpm",
  "pause_ratio",
  "response_latency_ms",
  "disfluency_count",
  "first_person_pronoun_ratio",
]);

const FOREHEAD_ROI = [10, 67, 69, 104, 108, 151, 337, 333, 299, 297];
const LEFT_CHEEK_ROI = [116, 117, 118, 100, 126, 209];
const RIGHT_CHEEK_ROI = [345, 346, 347, 329, 355, 429];
const FINGERTIP_IDS = [4, 8, 12, 16, 20];

type Session = { id: string; session_title: string; status: string; subject_identifier: string };
type Landmark = { x: number; y: number; z?: number };
type VisionTasks = {
  face: import("@mediapipe/tasks-vision").FaceLandmarker;
  pose: import("@mediapipe/tasks-vision").PoseLandmarker;
  hands: import("@mediapipe/tasks-vision").HandLandmarker;
};
type RGBSample = { timestamp_sec: number; red: number; green: number; blue: number };
type Slice100ms = {
  offset_ms: number;
  au04: number;
  au06: number;
  au12: number;
  au15: number;
  au20: number;
  au24: number;
  gaze_aversion: number;
  self_touch: number;
};

type VisualAccumulator = {
  faceFrames: number;
  trackedFrames: number;
  poseFrames: number;
  handFrames: number;
  au01: number[];
  au02: number[];
  au04: number[];
  au06: number[];
  au12: number[];
  au15: number[];
  au20: number[];
  au24: number[];
  au25: number[];
  fai: number[];
  headPitch: number[];
  headYaw: number[];
  headRoll: number[];
  gazeAway: number;
  gazeSamples: number;
  prevGaze: { gx: number; gy: number; time: number } | null;
  saccadeVelocities: number[];
  selfTouch: number;
  handSamples: number;
  handKeValues: number[];
  prevWrists: { left?: Landmark; right?: Landmark; time: number } | null;
  baselineShoulderZ: number | null;
  posturalLeanZ: number[];
  openEarSamples: number[];
  blinkDurations: number[];
  prolongedClosures: number;
  closedSince: number | null;
  earClosed: boolean;
  nose: { yaw: number; pitch: number; time: number } | null;
  headVelocity: number[];
  luma: number[];
  slices100ms: Slice100ms[];
  lastSliceAtMs: number;
  gazeXTrace: number[];
  gazeYTrace: number[];
  shoulderYSamples: Array<{ t: number; y: number }>;
  peakKeyframeJpegB64: string | null;
};

function newAccumulator(baselineShoulderZ: number | null = null, openEarSamples: number[] = []): VisualAccumulator {
  return {
    faceFrames: 0,
    trackedFrames: 0,
    poseFrames: 0,
    handFrames: 0,
    au01: [],
    au02: [],
    au04: [],
    au06: [],
    au12: [],
    au15: [],
    au20: [],
    au24: [],
    au25: [],
    fai: [],
    headPitch: [],
    headYaw: [],
    headRoll: [],
    gazeAway: 0,
    gazeSamples: 0,
    prevGaze: null,
    saccadeVelocities: [],
    selfTouch: 0,
    handSamples: 0,
    handKeValues: [],
    prevWrists: null,
    baselineShoulderZ,
    posturalLeanZ: [],
    openEarSamples,
    blinkDurations: [],
    prolongedClosures: 0,
    closedSince: null,
    earClosed: false,
    nose: null,
    headVelocity: [],
    luma: [],
    slices100ms: [],
    lastSliceAtMs: 0,
    gazeXTrace: [],
    gazeYTrace: [],
    shoulderYSamples: [],
    peakKeyframeJpegB64: null,
  };
}

const average = (values: number[]) =>
  values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;

const dist2d = (a: Landmark, b: Landmark) => Math.hypot(a.x - b.x, a.y - b.y);
const dist3d = (a: Landmark, b: Landmark) =>
  Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));

function singleEyeEar(points: Landmark[], ids: number[]) {
  const [a, b, c, d, e, f] = ids.map((index) => points[index]);
  if (!a || !b || !c || !d || !e || !f) return null;
  return (dist2d(b, f) + dist2d(c, e)) / Math.max(2 * dist2d(a, d), 1e-6);
}

function getBlendshapeScore(categories: Array<{ categoryName?: string; score?: number }>, name: string): number {
  const hit = categories.find((c) => c.categoryName === name);
  return typeof hit?.score === "number" ? hit.score : 0;
}

function estimateEulerPose(points: Landmark[]) {
  const leftEye = points[33];
  const rightEye = points[263];
  const nose = points[1];
  const chin = points[152];
  if (!leftEye || !rightEye || !nose || !chin) {
    return { pitch: 0, yaw: 0, roll: 0 };
  }
  const dx = rightEye.x - leftEye.x;
  const dy = rightEye.y - leftEye.y;
  const roll = (Math.atan2(dy, Math.abs(dx) > 1e-5 ? dx : 1e-5) * 180) / Math.PI;
  const eyeMidX = 0.5 * (leftEye.x + rightEye.x);
  const eyeMidY = 0.5 * (leftEye.y + rightEye.y);
  const iod = Math.max(dist2d(leftEye, rightEye), 1e-5);
  const yaw = Math.max(-75, Math.min(75, ((nose.x - eyeMidX) / iod) * 90));
  const faceV = Math.max(Math.abs(chin.y - eyeMidY), 1e-5);
  const pitch = Math.max(-60, Math.min(60, (((nose.y - eyeMidY) / faceV) - 0.36) * 110));
  return { pitch, yaw, roll };
}

function extractMultiRegionSkinRgb(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  points: Landmark[]
): [number, number, number] | null {
  const roiIndices = [...FOREHEAD_ROI, ...LEFT_CHEEK_ROI, ...RIGHT_CHEEK_ROI];
  const validMarks = roiIndices.map((idx) => points[idx]).filter((p): p is Landmark => Boolean(p));
  if (!validMarks.length) return null;

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let skinCount = 0;

  let fallbackR = 0;
  let fallbackG = 0;
  let fallbackB = 0;
  let fallbackCount = 0;

  for (const mark of validMarks) {
    const cx = Math.max(2, Math.min(width - 3, Math.round(mark.x * width)));
    const cy = Math.max(2, Math.min(height - 3, Math.round(mark.y * height)));
    for (let dy = -2; dy <= 2; dy += 1) {
      for (let dx = -2; dx <= 2; dx += 1) {
        const offset = ((cy + dy) * width + cx + dx) * 4;
        const r = data[offset];
        const g = data[offset + 1];
        const b = data[offset + 2];
        const y = 0.299 * r + 0.587 * g + 0.114 * b;
        const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
        const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        if (y >= 30 && y <= 240) {
          fallbackR += r;
          fallbackG += g;
          fallbackB += b;
          fallbackCount += 1;
          if (cr >= 130 && cr <= 178 && cb >= 75 && cb <= 130) {
            rSum += r;
            gSum += g;
            bSum += b;
            skinCount += 1;
          }
        }
      }
    }
  }

  if (skinCount >= 12) {
    return [rSum / skinCount, gSum / skinCount, bSum / skinCount];
  }
  if (fallbackCount >= 8) {
    return [fallbackR / fallbackCount, fallbackG / fallbackCount, fallbackB / fallbackCount];
  }
  return null;
}

function resampleTo16k(samples: Float32Array, inputRate: number) {
  const outputLength = Math.max(1, Math.floor((samples.length * 16000) / inputRate));
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const sourceIndex = (i * inputRate) / 16000;
    const left = Math.min(samples.length - 1, Math.floor(sourceIndex));
    const right = Math.min(samples.length - 1, left + 1);
    output[i] = samples[left] + (samples[right] - samples[left]) * (sourceIndex - left);
  }
  return output;
}

function normalizeAudioSamples(samples: Float32Array): Float32Array {
  if (!samples.length) return samples;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sumSq += samples[i] * samples[i];
  }
  const rms = Math.sqrt(sumSq / samples.length);
  // Boost quiet spoken audio toward -16 dBFS (~0.1585 RMS) while leaving background silence untouched
  if (rms > 0.002 && rms < 0.14) {
    const gain = Math.min(18.0, 0.1585 / rms);
    const out = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i += 1) {
      out[i] = Math.tanh(samples[i] * gain);
    }
    return out;
  }
  return samples;
}

function encodeWav(samples: Float32Array, inputRate: number) {
  const outputRate = 16000;
  const converted = normalizeAudioSamples(resampleTo16k(samples, inputRate));
  const outputLength = converted.length;
  const pcm = new ArrayBuffer(44 + outputLength * 2);
  const view = new DataView(pcm);
  const write = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + outputLength * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, outputRate, true);
  view.setUint32(28, outputRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, outputLength * 2, true);
  for (let i = 0; i < outputLength; i += 1) {
    const value = Math.max(-1, Math.min(1, converted[i]));
    view.setInt16(44 + i * 2, value < 0 ? value * 32768 : value * 32767, true);
  }
  return new Blob([pcm], { type: "audio/wav" });
}

let mediaPipeConsoleFilterInstalled = false;
function installMediaPipeConsoleFilter() {
  if (mediaPipeConsoleFilterInstalled || typeof window === "undefined") return;
  mediaPipeConsoleFilterInstalled = true;
  const isBenignMediaPipeLog = (args: unknown[]) => {
    const msg = args.map((a) => (typeof a === "string" ? a : "")).join(" ");
    return (
      msg.includes("Created TensorFlow Lite XNNPACK delegate") ||
      msg.includes("Sets FaceBlendshapesGraph acceleration") ||
      msg.includes("OpenGL error checking is disabled") ||
      msg.includes("NORM_RECT without IMAGE_DIMENSIONS") ||
      msg.includes("vision_wasm_internal.js") ||
      msg.includes("gl_context.cc") ||
      msg.includes("face_landmarker_graph.cc") ||
      msg.includes("landmark_projection_calculator.cc")
    );
  };
  const origError = console.error.bind(console);
  const origWarn = console.warn.bind(console);
  console.error = (...args: unknown[]) => {
    if (isBenignMediaPipeLog(args)) return;
    origError(...args);
  };
  console.warn = (...args: unknown[]) => {
    if (isBenignMediaPipeLog(args)) return;
    origWarn(...args);
  };
}

async function createLandmarkers(): Promise<VisionTasks> {
  installMediaPipeConsoleFilter();
  const vision = await import("@mediapipe/tasks-vision");
  const fileset = await vision.FilesetResolver.forVisionTasks(VISION_CDN);
  const make = async <T,>(factory: (delegate: "GPU" | "CPU") => Promise<T>) => {
    try {
      return await factory("GPU");
    } catch {
      return factory("CPU");
    }
  };
  const [face, pose, hands] = await Promise.all([
    make((delegate) =>
      vision.FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: false,
      })
    ),
    make((delegate) =>
      vision.PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate },
        runningMode: "VIDEO",
        numPoses: 1,
        outputSegmentationMasks: false,
      })
    ),
    make((delegate) =>
      vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL, delegate },
        runningMode: "VIDEO",
        numHands: 2,
      })
    ),
  ]);
  return { face, pose, hands };
}

function drawHudOverlay(
  hudCanvas: HTMLCanvasElement,
  facePoints: Landmark[] | null,
  posePoints: Landmark[] | null,
  handLandmarks: Landmark[][] | null
) {
  const ctx = hudCanvas.getContext("2d");
  if (!ctx) return;
  const w = hudCanvas.width;
  const h = hudCanvas.height;
  ctx.clearRect(0, 0, w, h);

  if (facePoints && facePoints.length) {
    ctx.fillStyle = "rgba(96, 216, 195, 0.65)";
    for (let i = 0; i < facePoints.length; i += 3) {
      const pt = facePoints[i];
      ctx.beginPath();
      ctx.arc((1 - pt.x) * w, pt.y * h, 1.3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = "#65a9ff";
    ctx.lineWidth = 1.5;
    for (const irisIdx of [468, 473]) {
      const iris = facePoints[irisIdx];
      if (iris) {
        ctx.beginPath();
        ctx.arc((1 - iris.x) * w, iris.y * h, 5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.strokeStyle = "rgba(110, 210, 163, 0.78)";
    ctx.lineWidth = 1.2;
    for (const poly of [FOREHEAD_ROI, LEFT_CHEEK_ROI, RIGHT_CHEEK_ROI]) {
      const pts = poly.map((idx) => facePoints[idx]).filter((p): p is Landmark => Boolean(p));
      if (pts.length >= 3) {
        ctx.beginPath();
        ctx.moveTo((1 - pts[0].x) * w, pts[0].y * h);
        for (let k = 1; k < pts.length; k += 1) {
          ctx.lineTo((1 - pts[k].x) * w, pts[k].y * h);
        }
        ctx.closePath();
        ctx.stroke();
      }
    }
  }

  if (posePoints && posePoints[11] && posePoints[12]) {
    ctx.strokeStyle = "rgba(228, 180, 90, 0.75)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo((1 - posePoints[11].x) * w, posePoints[11].y * h);
    ctx.lineTo((1 - posePoints[12].x) * w, posePoints[12].y * h);
    ctx.stroke();
  }

  if (handLandmarks && handLandmarks.length) {
    ctx.fillStyle = "rgba(238, 117, 133, 0.85)";
    for (const hand of handLandmarks) {
      for (const tipIdx of FINGERTIP_IDS) {
        const pt = hand[tipIdx];
        if (pt) {
          ctx.beginPath();
          ctx.arc((1 - pt.x) * w, pt.y * h, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }
}

export default function LiveSessionPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hudCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const speechRecognitionRef = useRef<{ stop: () => void } | null>(null);
  const browserTranscriptBufferRef = useRef<string>("");
  const captureTimerRef = useRef<number | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const asrAudioSamplesRef = useRef<Float32Array>(new Float32Array(0));
  const analyserRef = useRef<AnalyserNode | null>(null);
  const visionRef = useRef<VisionTasks | null>(null);
  const latestFaceRef = useRef<Landmark[] | null>(null);
  const rgbSamplesRef = useRef<RGBSample[]>([]);
  const visualRef = useRef<VisualAccumulator>(newAccumulator());
  const captureStartedRef = useRef(0);
  const windowStartedRef = useRef(0);
  const windowIndexRef = useRef(0);
  const inFlightRef = useRef(false);
  const captureLoopRef = useRef(false);
  const phaseRef = useRef<"CALIBRATING" | "RECORDING_LIVE">("CALIBRATING");
  const baselineRef = useRef<Record<string, number[]>>({});
  const calibrationCompletedRef = useRef(false);
  const trackingFramesRef = useRef(0);
  const faceFramesRef = useRef(0);

  const [session, setSession] = useState<Session | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [capturing, setCapturing] = useState(false);
  const [showHud, setShowHud] = useState(true);
  const [cameraFps, setCameraFps] = useState<number | null>(null);
  const [microphoneDb, setMicrophoneDb] = useState<number | null>(null);
  const [visionStatus, setVisionStatus] = useState("Waiting for camera");
  const [audioStatus, setAudioStatus] = useState("Waiting for audio pipeline");
  const [windows, setWindows] = useState(0);
  const [latestMetrics, setLatestMetrics] = useState<Record<string, number | string | null>>({});
  const [latestConfidence, setLatestConfidence] = useState<Record<string, unknown>>({});
  const [latestFlags, setLatestFlags] = useState<string[]>([]);
  const [latestMicroChain, setLatestMicroChain] = useState<string>("STEADY_BASELINE");
  const [latestCopilotProbe, setLatestCopilotProbe] = useState<{
    timestamp_mmss?: string;
    alert_banner?: string;
    suggested_follow_up_probe?: string;
  } | null>(null);
  const [vsiHistory, setVsiHistory] = useState<number[]>([]);
  const [vtliHistory, setVtliHistory] = useState<number[]>([]);
  const [cmciHistory, setCmciHistory] = useState<number[]>([]);
  const [bvpHistory, setBvpHistory] = useState<number[]>([]);
  const [transcript, setTranscript] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    api
      .get<Session>(`/sessions/${id}`)
      .then((value) => {
        setSession(value);
        if (value.status === "RECORDING_LIVE") phaseRef.current = "RECORDING_LIVE";
        if (value.status === "CALIBRATION_RETRY_REQUIRED") phaseRef.current = "CALIBRATING";
        const token = sessionStorage.getItem(`omnipulse.capture:${id}`);
        if (token) {
          api
            .get<{ window_count: number }>(`/sessions/${id}/capture/state`, { "X-Capture-Token": token })
            .then((state) => setWindows(state.window_count))
            .catch(() => undefined);
        }
      })
      .catch((e) => setError(e.message));
  }, [id]);

  const releaseCapture = useCallback(() => {
    captureLoopRef.current = false;
    if (captureTimerRef.current !== null) window.clearInterval(captureTimerRef.current);
    captureTimerRef.current = null;
    try {
      speechRecognitionRef.current?.stop();
    } catch {
      // ignore speech recognition stop error
    }
    speechRecognitionRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    audioProcessorRef.current?.disconnect();
    audioProcessorRef.current = null;
    analyserRef.current = null;
    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context && context.state !== "closed") void context.close();
    for (const task of [visionRef.current?.face, visionRef.current?.pose, visionRef.current?.hands]) task?.close();
    visionRef.current = null;
    latestFaceRef.current = null;
    setCapturing(false);
    setCameraFps(null);
    setMicrophoneDb(null);
  }, []);

  useEffect(() => () => releaseCapture(), [releaseCapture]);

  const appendCalibrationMetrics = useCallback((metrics: Record<string, number | string | null>) => {
    for (const [name, value] of Object.entries(metrics)) {
      if (BASELINE_FEATURES.has(name) && typeof value === "number" && Number.isFinite(value)) {
        (baselineRef.current[name] ||= []).push(value);
      }
    }
  }, []);

  const completeCalibration = useCallback(async () => {
    if (calibrationCompletedRef.current || phaseRef.current !== "CALIBRATING") return;
    const elapsedSeconds = (Date.now() - captureStartedRef.current) / 1000;
    if (elapsedSeconds < CALIBRATION_SEC) return;
    calibrationCompletedRef.current = true;
    const baselineFeatures = Object.fromEntries(
      Object.entries(baselineRef.current).filter(([, values]) => values.length >= 3)
    );
    try {
      const token = sessionStorage.getItem(`omnipulse.capture:${id}`);
      if (!token) throw new Error("The session capture token is missing. Reopen the session from setup.");
      const result = await api.post<{ status: string; baseline_features_available: string[] }>(
        `/sessions/${id}/capture/calibration/complete`,
        {
          face_tracking_uptime_pct: Math.round(
            (trackingFramesRef.current / Math.max(faceFramesRef.current, 1)) * 100
          ),
          duration_sec: Math.min(120, Math.max(60, Math.round(elapsedSeconds))),
          baseline_features: baselineFeatures,
        },
        { "X-Capture-Token": token }
      );
      if (result.status !== "RECORDING_LIVE") {
        setSession((current) => (current ? { ...current, status: result.status } : current));
        setNotice(
          "Face tracking uptime was below 75%. Press Start capture to retry with the participant's face centered and well lit."
        );
        releaseCapture();
        return;
      }
      phaseRef.current = "RECORDING_LIVE";
      setSession((current) => (current ? { ...current, status: "RECORDING_LIVE" } : current));
      setNotice(
        `Personal baseline locked (${result.baseline_features_available.length} biomarkers calibrated). Live 2.0s telemetry windows are now recording.`
      );
    } catch (caught) {
      calibrationCompletedRef.current = false;
      setError(caught instanceof Error ? caught.message : "Could not complete baseline calibration.");
    }
  }, [id, releaseCapture]);

  const sendWindow = useCallback(async () => {
    if (!captureLoopRef.current || inFlightRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    inFlightRef.current = true;
    const nowSeconds = (Date.now() - captureStartedRef.current) / 1000;
    const startSeconds = windowStartedRef.current;
    const endSeconds = Math.max(startSeconds + 0.1, nowSeconds);
    windowStartedRef.current = endSeconds;
    const index = windowIndexRef.current++;
    const visual = visualRef.current;
    const totalFaceFrames = Math.max(visual.faceFrames, 1);
    const earDurations = visual.blinkDurations.slice();

    const selfTouchRatio = visual.handSamples ? visual.selfTouch / visual.handSamples : 0;
    const handKe = average(visual.handKeValues) ?? 0;
    let gestureType = "NONE";
    if (selfTouchRatio >= 0.25) gestureType = "SELF_TOUCH";
    else if (handKe > 0.12) gestureType = "ILLUSTRATIVE";
    else if (handKe > 0.05) gestureType = "FIDGET_ADAPTOR";

    const metrics: Record<string, number | string | null> = {
      au01_inner_brow: average(visual.au01),
      au02_outer_brow: average(visual.au02),
      au04_brow_lowerer: average(visual.au04),
      au06_cheek_raiser: average(visual.au06),
      au12_lip_corner_puller: average(visual.au12),
      au15_lip_corner_depressor: average(visual.au15),
      au20_lip_stretcher: average(visual.au20),
      au24_lip_presser: average(visual.au24),
      au25_jaw_drop: average(visual.au25),
      facial_asymmetry_index: average(visual.fai),
      head_pitch_deg: average(visual.headPitch),
      head_yaw_deg: average(visual.headYaw),
      head_roll_deg: average(visual.headRoll),
      self_touch_ratio: selfTouchRatio,
      hand_kinetic_energy: handKe,
      gesture_type: gestureType,
      postural_lean_z: average(visual.posturalLeanZ),
      blink_rate_per_min: (visual.blinkDurations.length * 60) / Math.max(endSeconds - startSeconds, 0.1),
      blink_duration_ms_mean: average(earDurations),
      gaze_aversion_ratio: visual.gazeSamples ? visual.gazeAway / visual.gazeSamples : null,
      saccade_velocity_deg_s: average(visual.saccadeVelocities),
      head_velocity_deg_s: average(visual.headVelocity),
      illumination_change_pct: visual.luma.length
        ? ((Math.max(...visual.luma) - Math.min(...visual.luma)) / Math.max(average(visual.luma) || 1, 1)) * 100
        : 0,
    };

    const confidence = {
      face_tracking_uptime_pct: Math.round((visual.trackedFrames / totalFaceFrames) * 100),
      pose_tracking_pct: Math.round((visual.poseFrames / totalFaceFrames) * 100),
      hand_tracking_pct: Math.round((visual.handFrames / totalFaceFrames) * 100),
      overall: Number((visual.trackedFrames / totalFaceFrames).toFixed(3)),
    };

    const slices100ms = visual.slices100ms.slice(0, 24);
    const gazeXTrace = visual.gazeXTrace.slice(-60);
    const gazeYTrace = visual.gazeYTrace.slice(-60);
    const shoulderYSamples = visual.shoulderYSamples.slice(-60);
    const keyframeJpegB64 = visual.peakKeyframeJpegB64;
    visualRef.current = newAccumulator(visual.baselineShoulderZ, visual.openEarSamples.slice(-90));

    const rgbSamples = rgbSamplesRef.current.slice(-256);
    const audioParts = audioChunksRef.current;
    audioChunksRef.current = [];
    const sampleCount = audioParts.reduce((sum, part) => sum + part.length, 0);
    const joinedAudio = new Float32Array(sampleCount);
    let offset = 0;
    for (const part of audioParts) {
      joinedAudio.set(part, offset);
      offset += part.length;
    }

    const browserSpokenText = browserTranscriptBufferRef.current.trim();
    browserTranscriptBufferRef.current = "";

    const capture = {
      window: {
        window_index: index,
        start_sec: startSeconds,
        end_sec: endSeconds,
        confidence,
        metrics,
        transcript_segment: browserSpokenText || null,
        slices_100ms: slices100ms,
        gaze_x_trace: gazeXTrace,
        gaze_y_trace: gazeYTrace,
        shoulder_y_samples: shoulderYSamples,
        keyframe_jpeg_base64: keyframeJpegB64,
      },
      rgb_samples: rgbSamples,
      face_tracking_uptime_pct: confidence.face_tracking_uptime_pct,
      transcribe_audio: true,
    };

    try {
      const token = sessionStorage.getItem(`omnipulse.capture:${id}`);
      if (!token) throw new Error("The session capture token is missing. Reopen this session from setup.");
      const form = new FormData();
      form.set("capture", JSON.stringify(capture));
      if (joinedAudio.length) {
        form.set(
          "audio",
          encodeWav(joinedAudio, audioContextRef.current?.sampleRate || 48000),
          `window-${index}.wav`
        );
      }
      if (joinedAudio.length) {
        const previous = asrAudioSamplesRef.current;
        const convertedAudio = resampleTo16k(joinedAudio, audioContextRef.current?.sampleRate || 48000);
        const combined = new Float32Array(previous.length + convertedAudio.length);
        combined.set(previous);
        combined.set(convertedAudio, previous.length);
        asrAudioSamplesRef.current = combined.slice(Math.max(0, combined.length - 16000 * 6));
      }
      const asrStartSec = Math.max(0, endSeconds - asrAudioSamplesRef.current.length / 16000);
      if (capture.transcribe_audio && asrAudioSamplesRef.current.length) {
        form.set("asr_audio", encodeWav(asrAudioSamplesRef.current, 16000), `asr-${index}.wav`);
        form.set("asr_start_sec", String(asrStartSec));
      }

      const result = await api.upload<{
        phase: string;
        metrics?: Record<string, number | string | null>;
        confidence?: Record<string, unknown>;
        live_copilot_probe?: {
          timestamp_mmss?: string;
          alert_banner?: string;
          suggested_follow_up_probe?: string;
        } | null;
        window?: {
          metrics: Record<string, number | string | null>;
          confidence?: Record<string, unknown>;
          composite_indices?: Record<string, number | null>;
          behavioral_flags?: string[];
          micro_sequence_chain?: string;
          transcript_segment?: string | null;
          live_copilot_probe?: {
            timestamp_mmss?: string;
            alert_banner?: string;
            suggested_follow_up_probe?: string;
          } | null;
        };
      }>(`/sessions/${id}/capture/window`, form, { "X-Capture-Token": token });

      const actualMetrics = result.window?.metrics || result.metrics || {};
      const actualConf = result.window?.confidence || result.confidence || {};
      setLatestMetrics(actualMetrics);
      setLatestConfidence(actualConf);

      if (result.window?.behavioral_flags) setLatestFlags(result.window.behavioral_flags);
      if (result.window?.micro_sequence_chain) setLatestMicroChain(result.window.micro_sequence_chain);
      const probe = result.live_copilot_probe || result.window?.live_copilot_probe;
      if (probe?.suggested_follow_up_probe) setLatestCopilotProbe(probe);
      if (result.window?.transcript_segment) {
        setTranscript((value) => `${value} ${result.window?.transcript_segment}`.trim());
      }

      const indices = result.window?.composite_indices || {};
      if (typeof indices.vocal_stress_index === "number") {
        setVsiHistory((prev) => [...prev.slice(-29), indices.vocal_stress_index as number]);
      }
      if (typeof indices.visual_tension_load_index === "number") {
        setVtliHistory((prev) => [...prev.slice(-29), indices.visual_tension_load_index as number]);
      }
      if (typeof indices.cross_modal_congruency_index === "number") {
        setCmciHistory((prev) => [...prev.slice(-29), (indices.cross_modal_congruency_index as number) * 100]);
      }

      appendCalibrationMetrics(actualMetrics);
      setWindows((value) => value + 1);

      const acoustic = String(actualConf.acoustic_status || "ready");
      const asr = String(actualConf.asr_status || "standby");
      const rppgSt = String(actualConf.rppg_status || "INSUFFICIENT_SIGNAL");
      setAudioStatus(`Acoustic: ${acoustic} · Whisper ASR: ${asr} · rPPG: ${rppgSt}`);

      if (
        phaseRef.current === "CALIBRATING" &&
        (Date.now() - captureStartedRef.current) / 1000 >= CALIBRATION_SEC
      ) {
        await completeCalibration();
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "This capture window could not be processed.");
    } finally {
      inFlightRef.current = false;
    }
  }, [appendCalibrationMetrics, completeCalibration, id]);

  const startCapture = useCallback(async () => {
    setError("");
    setNotice("");
    setTranscript("");
    setLatestMetrics({});
    try {
      const token = sessionStorage.getItem(`omnipulse.capture:${id}`);
      if (!token) throw new Error("The session capture token is missing. Reopen the session from setup.");
      const captureState = await api.get<{
        status: string;
        next_window_index: number;
        window_count: number;
        last_end_sec: number;
      }>(`/sessions/${id}/capture/state`, { "X-Capture-Token": token });

      if (["CALIBRATION_RETRY_REQUIRED", "CREATED"].includes(captureState.status)) {
        await api.post(`/sessions/${id}/calibrate`);
        captureState.status = "CALIBRATING";
        setSession((current) => (current ? { ...current, status: "CALIBRATING" } : current));
      }
      if (!["CALIBRATING", "RECORDING_LIVE"].includes(captureState.status)) {
        throw new Error(`This session cannot capture in ${captureState.status} state.`);
      }

      phaseRef.current = captureState.status === "RECORDING_LIVE" ? "RECORDING_LIVE" : "CALIBRATING";
      windowIndexRef.current = captureState.next_window_index;
      windowStartedRef.current = captureState.last_end_sec;
      setWindows(captureState.window_count);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Camera preview is not ready.");
      video.srcObject = stream;
      await video.play();
      setCapturing(true);
      setVisionStatus("Loading local MediaPipe Face, Pose & Hand models…");
      const tasks = await createLandmarkers();
      visionRef.current = tasks;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        if (!captureLoopRef.current) return;
        const input = event.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(input));
        let excess =
          audioChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0) -
          event.inputBuffer.sampleRate * 8;
        while (excess > 0 && audioChunksRef.current.length) {
          if (audioChunksRef.current[0].length <= excess) {
            excess -= audioChunksRef.current.shift()!.length;
          } else {
            audioChunksRef.current[0] = audioChunksRef.current[0].slice(excess);
            excess = 0;
          }
        }
        event.outputBuffer.getChannelData(0).fill(0);
      };
      source.connect(processor);
      processor.connect(audioContext.destination);
      audioProcessorRef.current = processor;
      await audioContext.resume();

      // Upgrade 1: Start continuous browser Web Speech API fallback alongside Whisper ASR
      browserTranscriptBufferRef.current = "";
      if (typeof window !== "undefined") {
        const SpeechRec =
          (window as unknown as { SpeechRecognition?: new () => unknown; webkitSpeechRecognition?: new () => unknown })
            .SpeechRecognition ||
          (window as unknown as { webkitSpeechRecognition?: new () => unknown }).webkitSpeechRecognition;
        if (SpeechRec) {
          try {
            const rec = new SpeechRec() as {
              continuous: boolean;
              interimResults: boolean;
              lang: string;
              onresult: ((ev: { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null;
              onend: (() => void) | null;
              start: () => void;
              stop: () => void;
            };
            rec.continuous = true;
            rec.interimResults = false;
            rec.lang = "en-US";
            rec.onresult = (ev) => {
              let chunkText = "";
              for (let i = ev.resultIndex; i < ev.results.length; i += 1) {
                if (ev.results[i].isFinal && ev.results[i][0]?.transcript) {
                  chunkText += ` ${ev.results[i][0].transcript}`;
                }
              }
              if (chunkText.trim()) {
                browserTranscriptBufferRef.current = `${browserTranscriptBufferRef.current} ${chunkText.trim()}`.trim();
              }
            };
            rec.onend = () => {
              if (captureLoopRef.current) {
                try {
                  rec.start();
                } catch {
                  // ignore restart error
                }
              }
            };
            rec.start();
            speechRecognitionRef.current = rec;
          } catch {
            // browser speech recognition optional fallback
          }
        }
      }

      captureStartedRef.current =
        Date.now() - (phaseRef.current === "RECORDING_LIVE" ? captureState.last_end_sec * 1000 : 0);
      calibrationCompletedRef.current = false;
      baselineRef.current = {};
      trackingFramesRef.current = 0;
      faceFramesRef.current = 0;
      visualRef.current = newAccumulator();
      rgbSamplesRef.current = [];
      audioChunksRef.current = [];
      asrAudioSamplesRef.current = new Float32Array(0);
      captureLoopRef.current = true;
      setVisionStatus("478-Point Face Mesh + 33-Point Pose + 21-Point Hand tracking active");
      setAudioStatus("16kHz Normalized PCM Audio + Whisper ASR + eGeMAPSv02 DSP active");

      let previousDetection = 0;
      let lastFpsAt = performance.now();
      let decoded = 0;
      let lastDecodedTime = -1;
      let lastRgbAt = 0;
      let lastUiAt = 0;
      let visionFailure = "";
      const levelSamples = new Float32Array(analyser.fftSize);

      const track = (time: number) => {
        if (!captureLoopRef.current) return;
        const visual = visualRef.current;
        const currentVideo = videoRef.current;
        const canvas = canvasRef.current;
        const hudCanvas = hudCanvasRef.current;

        if (
          currentVideo &&
          canvas &&
          currentVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          currentVideo.videoWidth > 0 &&
          currentVideo.videoHeight > 0
        ) {
          if (currentVideo.currentTime !== lastDecodedTime) {
            decoded += 1;
            lastDecodedTime = currentVideo.currentTime;
          }
          if (time - previousDetection >= 95) {
            previousDetection = time;
            try {
              if (currentVideo.width !== currentVideo.videoWidth) currentVideo.width = currentVideo.videoWidth;
              if (currentVideo.height !== currentVideo.videoHeight) currentVideo.height = currentVideo.videoHeight;
              const result = tasks.face.detectForVideo(currentVideo, time);
              const points = (result.faceLandmarks?.[0] || null) as Landmark[] | null;
              latestFaceRef.current = points;
              visual.faceFrames += 1;
              faceFramesRef.current += 1;

              let poseLandmarks: Landmark[] | null = null;
              let handLandmarkSets: Landmark[][] | null = null;

              if (points?.length) {
                visual.trackedFrames += 1;
                trackingFramesRef.current += 1;
                const categories = result.faceBlendshapes?.[0]?.categories || [];

                const au01 = getBlendshapeScore(categories, "browInnerUp");
                const au02 =
                  0.5 *
                  (getBlendshapeScore(categories, "browOuterUpLeft") +
                    getBlendshapeScore(categories, "browOuterUpRight"));
                const au04 =
                  0.5 *
                  (getBlendshapeScore(categories, "browDownLeft") +
                    getBlendshapeScore(categories, "browDownRight"));
                const au06 =
                  0.5 *
                  (getBlendshapeScore(categories, "cheekSquintLeft") +
                    getBlendshapeScore(categories, "cheekSquintRight"));
                const au12 =
                  0.5 *
                  (getBlendshapeScore(categories, "mouthSmileLeft") +
                    getBlendshapeScore(categories, "mouthSmileRight"));
                const au15 =
                  0.5 *
                  (getBlendshapeScore(categories, "mouthFrownLeft") +
                    getBlendshapeScore(categories, "mouthFrownRight"));
                const au20 =
                  0.5 *
                  (getBlendshapeScore(categories, "mouthStretchLeft") +
                    getBlendshapeScore(categories, "mouthStretchRight"));
                const au24 =
                  0.5 *
                  (getBlendshapeScore(categories, "mouthPressLeft") +
                    getBlendshapeScore(categories, "mouthPressRight"));
                const au25 = getBlendshapeScore(categories, "jawOpen");

                const fai =
                  0.25 *
                  (Math.abs(
                    getBlendshapeScore(categories, "mouthSmileLeft") -
                      getBlendshapeScore(categories, "mouthSmileRight")
                  ) +
                    Math.abs(
                      getBlendshapeScore(categories, "browDownLeft") -
                        getBlendshapeScore(categories, "browDownRight")
                    ) +
                    Math.abs(
                      getBlendshapeScore(categories, "cheekSquintLeft") -
                        getBlendshapeScore(categories, "cheekSquintRight")
                    ) +
                    Math.abs(
                      getBlendshapeScore(categories, "mouthPressLeft") -
                        getBlendshapeScore(categories, "mouthPressRight")
                    ));

                visual.au01.push(au01);
                visual.au02.push(au02);
                visual.au04.push(au04);
                visual.au06.push(au06);
                visual.au12.push(au12);
                visual.au15.push(au15);
                visual.au20.push(au20);
                visual.au24.push(au24);
                visual.au25.push(au25);
                visual.fai.push(fai);

                const poseEuler = estimateEulerPose(points);
                visual.headPitch.push(poseEuler.pitch);
                visual.headYaw.push(poseEuler.yaw);
                visual.headRoll.push(poseEuler.roll);
                if (visual.nose) {
                  const dt = Math.max((time - visual.nose.time) / 1000, 0.01);
                  const angVel =
                    Math.hypot(poseEuler.yaw - visual.nose.yaw, poseEuler.pitch - visual.nose.pitch) / dt;
                  visual.headVelocity.push(angVel);
                }
                visual.nose = { yaw: poseEuler.yaw, pitch: poseEuler.pitch, time };

                const leftEye = singleEyeEar(points, [33, 160, 158, 133, 153, 144]);
                const rightEye = singleEyeEar(points, [362, 385, 387, 263, 373, 380]);
                const rawEar = average([leftEye, rightEye].filter((v): v is number => v !== null));
                const pitchComp = Math.max(0.5, Math.abs(Math.cos((poseEuler.pitch * Math.PI) / 180)));
                const compEar = rawEar !== null ? rawEar * pitchComp : null;

                if (compEar !== null && compEar > 0.22) {
                  visual.openEarSamples.push(compEar);
                }
                const medianOpenEar = visual.openEarSamples.length >= 8 ? average(visual.openEarSamples) || 0.28 : 0.28;
                const tauBlink = Math.min(0.21, 0.72 * medianOpenEar);

                if (compEar !== null && compEar < tauBlink && !visual.earClosed) {
                  visual.earClosed = true;
                  visual.closedSince = time;
                } else if (compEar !== null && compEar >= tauBlink && visual.earClosed) {
                  const duration = time - (visual.closedSince || time);
                  if (duration >= 66 && duration <= 400) {
                    visual.blinkDurations.push(duration);
                  } else if (duration > 400) {
                    visual.prolongedClosures += 1;
                  }
                  visual.earClosed = false;
                  visual.closedSince = null;
                }

                const irisLeft = points[468];
                const irisRight = points[473];
                let frameGazeAway = 0;
                if (
                  irisLeft &&
                  irisRight &&
                  points[33] &&
                  points[133] &&
                  points[159] &&
                  points[145] &&
                  points[362] &&
                  points[263]
                ) {
                  const gxL = (irisLeft.x - points[33].x) / Math.max(Math.abs(points[133].x - points[33].x), 1e-4);
                  const gxR = (irisRight.x - points[362].x) / Math.max(Math.abs(points[263].x - points[362].x), 1e-4);
                  const gyL = (irisLeft.y - points[159].y) / Math.max(Math.abs(points[145].y - points[159].y), 1e-4);
                  const gx = Math.max(0, Math.min(1, 0.5 * (gxL + gxR)));
                  const gy = Math.max(0, Math.min(1, gyL));
                  visual.gazeSamples += 1;
                  visual.gazeXTrace.push(Number(gx.toFixed(4)));
                  visual.gazeYTrace.push(Number(gy.toFixed(4)));
                  if (Math.abs(gx - 0.5) > 0.14 || Math.abs(gy - 0.5) > 0.16) {
                    visual.gazeAway += 1;
                    frameGazeAway = 1;
                  }
                  if (visual.prevGaze) {
                    const dtG = Math.max((time - visual.prevGaze.time) / 1000, 0.01);
                    const sacVel = (Math.hypot(gx - visual.prevGaze.gx, gy - visual.prevGaze.gy) / dtG) * 30;
                    visual.saccadeVelocities.push(sacVel);
                  }
                  visual.prevGaze = { gx, gy, time };
                }

                const poseResult = tasks.pose.detectForVideo(currentVideo, time);
                poseLandmarks = (poseResult.landmarks?.[0] || null) as Landmark[] | null;
                let dSw = 0.28;
                let neckPt: Landmark = points[152] || points[1];
                if (poseLandmarks?.length && poseLandmarks[11] && poseLandmarks[12]) {
                  visual.poseFrames += 1;
                  const p11 = poseLandmarks[11];
                  const p12 = poseLandmarks[12];
                  dSw = Math.max(dist3d(p11, p12), 0.05);
                  neckPt = {
                    x: 0.5 * (p11.x + p12.x),
                    y: 0.5 * (p11.y + p12.y),
                    z: 0.5 * ((p11.z || 0) + (p12.z || 0)),
                  };
                  const meanZ = neckPt.z || 0;
                  if (visual.baselineShoulderZ === null) {
                    visual.baselineShoulderZ = meanZ;
                  }
                  visual.posturalLeanZ.push((visual.baselineShoulderZ - meanZ) / dSw);
                  const relSec = (Date.now() - captureStartedRef.current) / 1000;
                  visual.shoulderYSamples.push({
                    t: Number(relSec.toFixed(3)),
                    y: Number((0.5 * (p11.y + p12.y)).toFixed(5)),
                  });
                }

                const handResult = tasks.hands.detectForVideo(currentVideo, time);
                handLandmarkSets = (handResult.landmarks || null) as Landmark[][] | null;
                let frameSelfTouch = 0;
                if (handLandmarkSets?.length) {
                  visual.handFrames += 1;
                  visual.handSamples += 1;
                  const targetZones = [points[1], points[152], points[234], points[454], neckPt].filter(
                    (p): p is Landmark => Boolean(p)
                  );
                  let minAdaptorNorm = Infinity;
                  for (const hand of handLandmarkSets) {
                    for (const tipIdx of FINGERTIP_IDS) {
                      const tip = hand[tipIdx];
                      if (tip) {
                        for (const zone of targetZones) {
                          minAdaptorNorm = Math.min(minAdaptorNorm, dist3d(tip, zone) / dSw);
                        }
                      }
                    }
                  }
                  if (minAdaptorNorm < 0.22) {
                    visual.selfTouch += 1;
                    frameSelfTouch = 1;
                  }

                  const leftWrist = handLandmarkSets[0]?.[0];
                  const rightWrist = handLandmarkSets[1]?.[0];
                  if (visual.prevWrists) {
                    const dtW = Math.max((time - visual.prevWrists.time) / 1000, 0.01);
                    let vSq = 0;
                    if (leftWrist && visual.prevWrists.left) {
                      vSq += Math.pow(dist3d(leftWrist, visual.prevWrists.left) / dtW, 2);
                    }
                    if (rightWrist && visual.prevWrists.right) {
                      vSq += Math.pow(dist3d(rightWrist, visual.prevWrists.right) / dtW, 2);
                    }
                    visual.handKeValues.push(vSq / (2 * dSw * dSw));
                  }
                  visual.prevWrists = { left: leftWrist, right: rightWrist, time };
                } else {
                  visual.handSamples += 1;
                }

                const windowElapsedMs = Math.max(
                  0,
                  Math.round(Date.now() - captureStartedRef.current - windowStartedRef.current * 1000)
                );
                if (windowElapsedMs - visual.lastSliceAtMs >= 95 && visual.slices100ms.length < 24) {
                  visual.lastSliceAtMs = windowElapsedMs;
                  visual.slices100ms.push({
                    offset_ms: Math.min(2000, windowElapsedMs),
                    au04: Number(au04.toFixed(3)),
                    au06: Number(au06.toFixed(3)),
                    au12: Number(au12.toFixed(3)),
                    au15: Number(au15.toFixed(3)),
                    au20: Number(au20.toFixed(3)),
                    au24: Number(au24.toFixed(3)),
                    gaze_aversion: frameGazeAway,
                    self_touch: frameSelfTouch,
                  });
                }
              }

              if (hudCanvas && showHud) {
                hudCanvas.width = 640;
                hudCanvas.height = 360;
                drawHudOverlay(hudCanvas, points, poseLandmarks, handLandmarkSets);
              } else if (hudCanvas) {
                const hCtx = hudCanvas.getContext("2d");
                hCtx?.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
              }

              if (visionFailure) {
                visionFailure = "";
                setVisionStatus("478-Point Face Mesh + 33-Point Pose + 21-Point Hand tracking active");
              }
            } catch (caught) {
              const detail = caught instanceof Error ? caught.message : "video inference failed";
              const message = `MediaPipe tracking note: ${detail}`;
              if (message !== visionFailure) {
                visionFailure = message;
                setVisionStatus(message);
              }
            }
          }

          const rgbPoints = latestFaceRef.current;
          if (time - lastRgbAt >= 33 && rgbPoints?.length) {
            lastRgbAt = time;
            const width = 320;
            const height = Math.max(
              180,
              Math.round((width * currentVideo.videoHeight) / Math.max(currentVideo.videoWidth, 1))
            );
            canvas.width = width;
            canvas.height = height;
            const context = canvas.getContext("2d", { willReadFrequently: true });
            if (context) {
              context.drawImage(currentVideo, 0, 0, width, height);
              const image = context.getImageData(0, 0, width, height);
              const rgb = extractMultiRegionSkinRgb(image.data, width, height, rgbPoints);
              if (rgb) {
                const rawTs = (Date.now() - captureStartedRef.current) / 1000;
                const lastSample = rgbSamplesRef.current[rgbSamplesRef.current.length - 1];
                const timestamp = lastSample ? Math.max(rawTs, lastSample.timestamp_sec + 0.002) : rawTs;
                rgbSamplesRef.current.push({
                  timestamp_sec: timestamp,
                  red: rgb[0],
                  green: rgb[1],
                  blue: rgb[2],
                });
                while (
                  rgbSamplesRef.current.length > 512 ||
                  (rgbSamplesRef.current[0] && timestamp - rgbSamplesRef.current[0].timestamp_sec > 8.8)
                ) {
                  rgbSamplesRef.current.shift();
                }
                if (!visualRef.current.peakKeyframeJpegB64 && visualRef.current.slices100ms.length >= 2) {
                  try {
                    visualRef.current.peakKeyframeJpegB64 = canvas.toDataURL("image/jpeg", 0.68);
                  } catch {
                    // ignore canvas export errors
                  }
                }
                const yVal = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
                visualRef.current.luma.push(yVal);
                const bvpProxy = rgb[1] - 0.5 * (rgb[0] + rgb[2]);
                setBvpHistory((prev) => [...prev.slice(-49), bvpProxy]);
              }
            }
          }

          if (time - lastUiAt >= 250) {
            if (time - lastFpsAt >= 1000) {
              setCameraFps(decoded);
              decoded = 0;
              lastFpsAt = time;
            }
            analyser.getFloatTimeDomainData(levelSamples);
            const rms = Math.sqrt(
              levelSamples.reduce((sum, value) => sum + value * value, 0) / levelSamples.length
            );
            setMicrophoneDb(rms > 0 ? Math.round(20 * Math.log10(rms)) : -100);
            lastUiAt = time;
            setElapsed(Math.floor((Date.now() - captureStartedRef.current) / 1000));
          }
        }
        requestAnimationFrame(track);
      };
      requestAnimationFrame(track);
      captureTimerRef.current = window.setInterval(() => {
        void sendWindow();
      }, 2000);
    } catch (caught) {
      releaseCapture();
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not initialize the camera, microphone, or MediaPipe models."
      );
      setVisionStatus("MediaPipe unavailable");
    }
  }, [id, releaseCapture, sendWindow, showHud]);

  function stopCapture() {
    captureLoopRef.current = false;
    releaseCapture();
    setNotice(
      "Capture paused. Resume capture to complete the 90-second baseline and record live telemetry windows."
    );
  }

  async function finishSession() {
    if (!session || session.status !== "RECORDING_LIVE" || windows < 1) {
      setError(
        session?.status !== "RECORDING_LIVE"
          ? "Complete the 90-second personal baseline before finalizing this session."
          : "Keep capture running until at least one live telemetry window is recorded."
      );
      return;
    }
    setBusy(true);
    setError("");
    try {
      if (capturing) stopCapture();
      while (inFlightRef.current) await new Promise((resolve) => window.setTimeout(resolve, 100));
      await api.post(`/sessions/${id}/finalize`);
      sessionStorage.removeItem(`omnipulse.capture:${id}`);
      router.push(`/sessions/${id}/report`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not finalize this session.");
    } finally {
      setBusy(false);
    }
  }

  const clock = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`;
  const calibrationRemaining = Math.max(0, CALIBRATION_SEC - elapsed);
  const calibrated = phaseRef.current === "RECORDING_LIVE";
  const canFinish = session?.status === "RECORDING_LIVE" && windows > 0;
  const metric = (key: string, digits = 2) =>
    typeof latestMetrics[key] === "number" ? (latestMetrics[key] as number).toFixed(digits) : "—";

  const rppgStatus = String(latestConfidence.rppg_status || "INSUFFICIENT_SIGNAL");
  const rppgValid = rppgStatus === "VALID" && typeof latestMetrics.hr_bpm === "number";

  const renderSparkline = (values: number[], strokeColor: string) => {
    if (values.length < 2) {
      return (
        <svg className="sparkline" viewBox="0 0 220 48">
          <line x1="0" y1="24" x2="220" y2="24" stroke="#26313e" strokeDasharray="3 3" />
        </svg>
      );
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, 1e-3);
    const pts = values
      .map((v, i) => {
        const x = (i / Math.max(values.length - 1, 1)) * 216 + 2;
        const y = 42 - ((v - min) / span) * 36;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return (
      <svg className="sparkline" viewBox="0 0 220 48">
        <polyline fill="none" stroke={strokeColor} strokeWidth="2" points={pts} />
      </svg>
    );
  };

  return (
    <>
      <PageHeading
        eyebrow="LIVE MULTIMODAL TELEMETRY STUDIO · 30 FPS VISION + 16KHZ AUDIO"
        title={session?.session_title || "Preparing session studio…"}
        description={`Participant ${session?.subject_identifier || "—"} · ${
          calibrated
            ? "Personal baseline locked · Recording 2.0s trimodal + rPPG windows"
            : `Calibrating personal baseline (${Math.floor(calibrationRemaining / 60)}:${String(
                calibrationRemaining % 60
              ).padStart(2, "0")} remaining)`
        }`}
        action={
          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              className="button button-ghost"
              onClick={() => setShowHud((v) => !v)}
              style={{ minHeight: 32 }}
            >
              <Icon name="eye" size={14} /> {showHud ? "Hide HUD Mesh" : "Show HUD Mesh"}
            </button>
            <Badge tone={capturing ? "green" : "amber"}>
              {capturing ? "30 FPS INFERENCE ACTIVE" : "CAPTURE PAUSED"}
            </Badge>
          </div>
        }
      />

      <div className="live-header">
        <div className="live-meta">
          <div>
            <div className="live-timer">
              {clock}
              <small>SESSION CLOCK</small>
            </div>
          </div>
          <div className="live-chips">
            <Badge tone={calibrated ? "green" : "amber"}>
              {calibrated ? "DUAL-HORIZON BASELINE LOCKED" : `CALIBRATING (${calibrationRemaining}S LEFT)`}
            </Badge>
            <Badge tone="blue">{windows} TELEMETRY WINDOWS</Badge>
            <Badge tone={rppgValid ? "green" : "neutral"}>
              {rppgValid ? `rPPG SQI ${Number(latestConfidence.rppg_sqi || 0).toFixed(2)}` : `rPPG · ${rppgStatus}`}
            </Badge>
          </div>
        </div>

        <div className="heading-action">
          <button
            className="button button-secondary"
            onClick={capturing ? stopCapture : () => void startCapture()}
          >
            <Icon name="activity" size={14} />
            {capturing ? "Pause capture" : "Start camera + mic"}
          </button>
          <button
            className="button button-primary"
            onClick={() => void finishSession()}
            disabled={busy || !canFinish}
            title={!canFinish ? "Complete 90s calibration and record at least one window first." : undefined}
          >
            {busy ? "Synthesizing v2.0 Report…" : "Finalize & Generate Report"}
            <Icon name="arrow" size={14} />
          </button>
        </div>
      </div>

      {!canFinish && (
        <div className="subtle-note" style={{ marginBottom: 12 }}>
          Keep capture running through the 90-second baseline calibration and at least one live 2.0s window to enable
          report generation.
        </div>
      )}
      {error && (
        <div className="form-error" style={{ marginBottom: 12 }}>
          {error}
        </div>
      )}
      {notice && (
        <div className="sample-banner" style={{ marginBottom: 12 }}>
          <Icon name="shield" size={14} />
          <span>{notice}</span>
        </div>
      )}

      <div className="grid live-layout">
        <section className="card">
          <div className="card-heading">
            <div>
              <h2 className="card-title">Local 478-Point Face + Pose + Hand HUD</h2>
              <div className="card-subtitle">{visionStatus}</div>
            </div>
            <Badge tone={capturing ? "green" : "neutral"}>
              {capturing ? `FPS · ${cameraFps ?? 30}` : "CAMERA OFF"}
            </Badge>
          </div>

          <div className="camera-stage live-video">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              style={{ display: capturing ? "block" : "none" }}
            />
            <canvas
              ref={hudCanvasRef}
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                pointerEvents: "none",
                zIndex: 2,
                display: capturing && showHud ? "block" : "none",
              }}
            />
            {!capturing && (
              <div className="camera-placeholder">
                <span className="camera-placeholder-icon">
                  <Icon name="scan" size={22} />
                </span>
                <b>Camera &amp; microphone idle</b>
                <span>Click &ldquo;Start camera + mic&rdquo; to launch 30 FPS MediaPipe &amp; rPPG tracking</span>
              </div>
            )}
            {capturing && (
              <div className="camera-overlay">
                <span className="camera-overlay-label">
                  YCrCb SKIN ROI + 3D POSE · PITCH {metric("head_pitch_deg", 1)}° · YAW {metric("head_yaw_deg", 1)}°
                </span>
              </div>
            )}
            <div className="video-footer">
              <span>{capturing ? "ON-DEVICE MEDIAPIPE · ZERO VIDEO UPLOAD" : "VIDEO · OFF"}</span>
              <span>GESTURE · {String(latestMetrics.gesture_type || "NONE")}</span>
            </div>
          </div>
          <canvas ref={canvasRef} width="320" height="240" hidden />

          <div className="card-heading" style={{ marginTop: 16, marginBottom: 6 }}>
            <div>
              <h2 className="card-title">Optical rPPG Blood Volume Pulse (POS + CHROM)</h2>
              <div className="card-subtitle">
                {rppgValid
                  ? `Valid Pulse · POS ${metric("hr_bpm_pos", 1)} BPM vs CHROM ${metric("hr_bpm_chrom", 1)} BPM`
                  : "Anti-Fabrication Gate Active: Waiting for 256 stable frames (SQI ≥ 0.38, |POS−CHROM| ≤ 12 BPM)"}
              </div>
            </div>
            <Badge tone={rppgValid ? "green" : "amber"}>
              {rppgValid ? `${metric("hr_bpm", 1)} BPM` : "SIGNAL PAUSED / GATED"}
            </Badge>
          </div>
          {renderSparkline(bvpHistory, rppgValid ? "#60d8c3" : "#596578")}

          <div className="card-heading" style={{ marginTop: 14, marginBottom: 6 }}>
            <div>
              <h2 className="card-title">Verbatim Speech &amp; Disfluency Stream (Whisper Large-v3 + Web Speech)</h2>
              <div className="card-subtitle">{audioStatus}</div>
            </div>
            <Badge tone="blue">EVERY WINDOW TRANSCRIBED</Badge>
          </div>
          <div className="empty-state" style={{ maxHeight: 130, overflowY: "auto", textAlign: "left", padding: 14 }}>
            {transcript ||
              "Spoken utterances (including verbatim hesitations 'um', 'uh', 'haan', 'matlab') will stream here as windows are transcribed."}
          </div>
        </section>

        <div className="telemetry-stack">
          <section className="card">
            <div className="card-heading">
              <div>
                <h2 className="card-title">Real-time multimodal biomarker matrix</h2>
                <div className="card-subtitle">Live window measurements prior to Dual-Horizon Z-score fusion</div>
              </div>
              <Icon name="shield" size={16} />
            </div>

            <div className="metric-grid">
              <div className={`metric-card ${!rppgValid ? "paused" : ""}`}>
                <div className="metric-label">
                  HEART RATE (rPPG) <Badge tone={rppgValid ? "green" : "neutral"}>POS+CHROM</Badge>
                </div>
                <div className="metric-value">
                  {rppgValid ? `${metric("hr_bpm", 1)} BPM` : "Signal Paused — Low SQI / Warming"}
                </div>
                <div className="metric-help">Never fabricated when motion &gt; 25°/s or SQI &lt; 0.38</div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  VOCAL PITCH &amp; JITTER <Badge tone="blue">eGeMAPSv02</Badge>
                </div>
                <div className="metric-value">
                  {metric("f0_semitone_mean", 1)} <span className="metric-unit">st</span>
                </div>
                <div className="metric-help">
                  Jitter {metric("jitter_local_mean", 4)} · Shimmer {metric("shimmer_local_db_mean", 2)} dB · HNR{" "}
                  {metric("hnr_db_mean", 1)} dB
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  BROW LOWER (AU04) &amp; LIP PRESS (AU24) <Badge tone="neutral">FACS</Badge>
                </div>
                <div className="metric-value">
                  {metric("au04_brow_lowerer")} / {metric("au24_lip_presser")}
                </div>
                <div className="metric-help">
                  AU06 Cheek {metric("au06_cheek_raiser")} · AU12 Smile {metric("au12_lip_corner_puller")} · FAI{" "}
                  {metric("facial_asymmetry_index")}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  BLINK &amp; GAZE TELEMETRY <Badge tone="neutral">OCULAR</Badge>
                </div>
                <div className="metric-value">
                  {metric("blink_rate_per_min", 1)} <span className="metric-unit">/min</span>
                </div>
                <div className="metric-help">
                  Gaze Aversion {metric("gaze_aversion_ratio")} · Saccade {metric("saccade_velocity_deg_s", 1)}°/s
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  SELF-ADAPTOR &amp; HAND KE <Badge tone="neutral">KINESICS</Badge>
                </div>
                <div className="metric-value">{metric("self_touch_ratio")}</div>
                <div className="metric-help">
                  Hand KE {metric("hand_kinetic_energy", 3)} · Lean Z {metric("postural_lean_z", 2)}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label">
                  SPEECH RATE &amp; MIC LEVEL <Badge tone="neutral">DISCOURSE</Badge>
                </div>
                <div className="metric-value">
                  {metric("speech_rate_wpm", 0)} <span className="metric-unit">WPM</span>
                </div>
                <div className="metric-help">
                  Pause Ratio {metric("pause_ratio")} · Mic {microphoneDb === null ? "—" : `${microphoneDb} dBFS`}
                </div>
              </div>
            </div>
          </section>

          <section className="card">
            <div className="card-heading">
              <div>
                <h2 className="card-title">Composite Indices (VSI · VTLI · CMCI) &amp; 100ms Micro-Sequence</h2>
                <div className="card-subtitle">
                  Rolling 60-second baseline-relative trajectories &amp; sub-second onset chain
                </div>
              </div>
              <Badge tone={latestFlags.includes("CROSS_MODAL_INCONGRUENCE") ? "rose" : "green"}>
                {latestFlags.includes("CROSS_MODAL_INCONGRUENCE") ? "INCONGRUENCE FLAGGED" : "CONCORDANT"}
              </Badge>
            </div>

            <div className="grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8 }}>
              <div className="metric-card" style={{ padding: 10 }}>
                <div className="metric-label">VOCAL STRESS (VSI)</div>
                {renderSparkline(vsiHistory, "#e4b45a")}
              </div>
              <div className="metric-card" style={{ padding: 10 }}>
                <div className="metric-label">VISUAL LOAD (VTLI)</div>
                {renderSparkline(vtliHistory, "#65a9ff")}
              </div>
              <div className="metric-card" style={{ padding: 10 }}>
                <div className="metric-label">CONGRUENCY (CMCI)</div>
                {renderSparkline(cmciHistory, "#60d8c3")}
              </div>
            </div>

            <div className="notice-block" style={{ marginTop: 10 }}>
              <div className="micro-label" style={{ marginBottom: 4 }}>
                LATEST 100MS SUB-SECOND MICRO-SEQUENCE
              </div>
              <code style={{ fontSize: 10, color: "#8ce5ca" }}>{latestMicroChain}</code>
              {latestFlags.length > 0 && (
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
                  {latestFlags.map((flag) => (
                    <Badge key={flag} tone={flag.includes("INCONGRUENCE") ? "rose" : "amber"}>
                      {flag}
                    </Badge>
                  ))}
                </div>
              )}
            </div>

            {latestCopilotProbe && (
              <div
                className="notice-block"
                style={{
                  marginTop: 10,
                  border: "1px solid rgba(96, 216, 195, 0.45)",
                  background: "rgba(15, 118, 110, 0.16)",
                }}
              >
                <div className="micro-label" style={{ color: "#60d8c3", marginBottom: 4 }}>
                  LIVE INTERVIEWER CO-PILOT · REAL-TIME FOLLOW-UP GENERATOR
                </div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#f8fafc", marginBottom: 4 }}>
                  {latestCopilotProbe.alert_banner}
                </div>
                <div style={{ fontSize: 12, color: "#bae6fd" }}>
                  <strong>Suggested Follow-Up Probe:</strong> &ldquo;{latestCopilotProbe.suggested_follow_up_probe}&rdquo;
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </>
  );
}
