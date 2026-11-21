-- Migration 001: Complete Initial Schema for OmniPulse MBS (PostgreSQL 16 + TimescaleDB)
-- Implements PDF Specification Section 6

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 1. Sessions Table
CREATE TABLE IF NOT EXISTS analysis_sessions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  subject_identifier VARCHAR(128) NOT NULL,
  session_title VARCHAR(255) NOT NULL,
  status VARCHAR(40) NOT NULL DEFAULT 'CREATED',
  -- Enum: CREATED, CALIBRATING, CALIBRATION_RETRY_REQUIRED, RECORDING_LIVE, PROCESSING_ALIGNMENT, SYNTHESIZING_RAG_REPORT, COMPLETED, FAILED
  consent_verified BOOLEAN NOT NULL DEFAULT FALSE,
  calibration_duration_sec DOUBLE PRECISION NOT NULL DEFAULT 90.0,
  total_duration_sec DOUBLE PRECISION,
  video_s3_key TEXT,
  audio_s3_key TEXT,
  report_pdf_s3_key TEXT,
  ambient_lux_score DOUBLE PRECISION,
  mic_snr_db DOUBLE PRECISION,
  overall_signal_confidence DOUBLE PRECISION,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

-- 2. Personal Baseline Parameters Table (Computed from 0s to 90s)
CREATE TABLE IF NOT EXISTS session_baselines (
  session_id UUID PRIMARY KEY REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  f0_semitone_median DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  f0_semitone_mad DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  jitter_median DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  jitter_mad DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  shimmer_median DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  shimmer_mad DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  hnr_median DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  hnr_mad DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  speech_rate_wpm_median DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  ear_open_median DOUBLE PRECISION NOT NULL DEFAULT 0.28,
  blink_rate_median DOUBLE PRECISION NOT NULL DEFAULT 15.0,
  blink_rate_mad DOUBLE PRECISION NOT NULL DEFAULT 3.0,
  gaze_aversion_median DOUBLE PRECISION NOT NULL DEFAULT 0.15,
  hr_bpm_baseline_median DOUBLE PRECISION,
  hr_bpm_baseline_mad DOUBLE PRECISION,
  hand_ke_median DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  hand_ke_mad DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  feature_medians JSONB NOT NULL DEFAULT '{}'::jsonb,
  feature_mads JSONB NOT NULL DEFAULT '{}'::jsonb,
  sample_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
  calibrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. TimescaleDB Hypertable for 2.0s Multimodal Telemetry Windows
CREATE TABLE IF NOT EXISTS telemetry_windows (
  time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  window_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  session_id UUID NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  window_index INTEGER NOT NULL,
  start_sec DOUBLE PRECISION NOT NULL,
  end_sec DOUBLE PRECISION NOT NULL,

  -- Visual / FACS / Gaze / Kinesics
  face_confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  head_pitch_deg DOUBLE PRECISION,
  head_yaw_deg DOUBLE PRECISION,
  head_roll_deg DOUBLE PRECISION,
  ear_mean DOUBLE PRECISION,
  blink_rate_per_min DOUBLE PRECISION,
  blink_duration_ms DOUBLE PRECISION,
  gaze_aversion_ratio DOUBLE PRECISION,
  saccade_velocity_deg_s DOUBLE PRECISION,
  au01_inner_brow DOUBLE PRECISION,
  au02_outer_brow DOUBLE PRECISION,
  au04_brow_lowerer DOUBLE PRECISION,
  au06_cheek_raiser DOUBLE PRECISION,
  au12_lip_corner_puller DOUBLE PRECISION,
  au15_lip_corner_depressor DOUBLE PRECISION,
  au20_lip_stretcher DOUBLE PRECISION,
  au24_lip_presser DOUBLE PRECISION,
  au25_jaw_drop DOUBLE PRECISION,
  facial_asymmetry_index DOUBLE PRECISION,
  self_touch_ratio DOUBLE PRECISION,
  hand_kinetic_energy DOUBLE PRECISION,
  gesture_type VARCHAR(48),
  postural_lean_z DOUBLE PRECISION,

  -- rPPG Heart Rate Telemetry
  hr_bpm_pos DOUBLE PRECISION,
  hr_bpm_chrom DOUBLE PRECISION,
  rppg_sqi DOUBLE PRECISION NOT NULL DEFAULT 0.0,
  rppg_status VARCHAR(32) NOT NULL DEFAULT 'INSUFFICIENT_SIGNAL',

  -- Acoustic openSMILE eGeMAPSv02
  f0_semitone_mean DOUBLE PRECISION,
  f0_semitone_std DOUBLE PRECISION,
  jitter_local DOUBLE PRECISION,
  shimmer_local_db DOUBLE PRECISION,
  hnr_db DOUBLE PRECISION,
  loudness_mean DOUBLE PRECISION,

  -- Linguistic & ASR
  transcript_segment TEXT,
  speech_rate_wpm DOUBLE PRECISION,
  pause_ratio DOUBLE PRECISION,
  disfluency_count INTEGER DEFAULT 0,
  text_valence_score DOUBLE PRECISION,

  -- Baseline Z-Scores & Composite Indices
  vsi_score DOUBLE PRECISION NOT NULL DEFAULT 50.0,
  vtli_score DOUBLE PRECISION NOT NULL DEFAULT 50.0,
  cmci_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  confidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
  baseline_z_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
  z_scores_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  composite_indices JSONB NOT NULL DEFAULT '{}'::jsonb,
  behavioral_flags TEXT[] NOT NULL DEFAULT '{}',

  PRIMARY KEY (time, session_id, window_index),
  CHECK (end_sec > start_sec)
);

SELECT create_hypertable('telemetry_windows', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS telemetry_windows_session_time_idx ON telemetry_windows(session_id, time DESC);

-- 4. Flagged Multimodal Events & RAG Annotations Table
CREATE TABLE IF NOT EXISTS session_events (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  start_sec DOUBLE PRECISION NOT NULL,
  end_sec DOUBLE PRECISION NOT NULL,
  event_category VARCHAR(64) NOT NULL,
  confidence_score DOUBLE PRECISION NOT NULL DEFAULT 0.8,
  transcript_excerpt TEXT,
  observed_signals_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  rag_citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
  llm_scientific_interpretation TEXT NOT NULL DEFAULT '',
  alternative_explanations TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (end_sec > start_sec)
);

-- 5. RAG Knowledge Base Documents & Parent-Child Chunks Tables
CREATE TABLE IF NOT EXISTS kb_documents (
  id VARCHAR(128) PRIMARY KEY,
  file_name VARCHAR(255) NOT NULL,
  title VARCHAR(255),
  source_title VARCHAR(512) NOT NULL,
  authors TEXT[] DEFAULT '{}',
  publication_year INTEGER,
  sha256_hash CHAR(64) UNIQUE NOT NULL,
  modality_tags TEXT[] NOT NULL DEFAULT '{}',
  evidence_tier VARCHAR(64) NOT NULL DEFAULT 'TIER_1_CANONICAL_RULE',
  file_s3_key TEXT,
  ingestion_status VARCHAR(32) NOT NULL DEFAULT 'UPLOADED',
  rejection_reason TEXT,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kb_parents (
  id VARCHAR(128) PRIMARY KEY,
  document_id VARCHAR(128) NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
  section TEXT,
  content_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_chunks (
  id VARCHAR(128) PRIMARY KEY,
  document_id VARCHAR(128) NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
  parent_id VARCHAR(128) NOT NULL REFERENCES kb_parents(id) ON DELETE CASCADE,
  section TEXT,
  content_text TEXT NOT NULL
);
