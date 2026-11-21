-- Migration 002: Cognitive & Dyadic Intelligence Upgrades for OmniPulse MBS v2.0
-- Implements PDF Specification Addendum Section 3 (Pages 35-36)

-- 1. Upgrade `session_baselines` to store Dual-Horizon Retroactive Floor parameters
ALTER TABLE session_baselines
ADD COLUMN IF NOT EXISTS floor_start_sec DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS floor_end_sec DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS hr_bpm_floor_median DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS hr_bpm_floor_mad DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS jitter_floor_median DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS jitter_floor_mad DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS au04_floor_median DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS au04_floor_mad DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS au24_floor_median DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS au24_floor_mad DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS habituation_delta_bpm DOUBLE PRECISION;

-- 2. Upgrade `telemetry_windows` with Phasic Delta, Micro-Sequences, and Entrainment
ALTER TABLE telemetry_windows
ADD COLUMN IF NOT EXISTS speaker_role VARCHAR(24) DEFAULT 'SUBJECT',
ADD COLUMN IF NOT EXISTS turn_id INTEGER,
ADD COLUMN IF NOT EXISTS turn_phase VARCHAR(32),
ADD COLUMN IF NOT EXISTS phasic_delta_z_json JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS micro_sequence_tokens TEXT[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS smile_offset_velocity DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS vocal_entrainment_aci DOUBLE PRECISION;

-- 3. Dyadic Conversational Turns Table
CREATE TABLE IF NOT EXISTS dyadic_turns (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  turn_index INTEGER NOT NULL,
  interviewer_start_sec DOUBLE PRECISION NOT NULL,
  interviewer_end_sec DOUBLE PRECISION NOT NULL,
  interviewer_transcript TEXT,
  trigger_keyword VARCHAR(128),
  trigger_timestamp_sec DOUBLE PRECISION,
  appraisal_latency_ms DOUBLE PRECISION NOT NULL,
  subject_start_sec DOUBLE PRECISION NOT NULL,
  subject_end_sec DOUBLE PRECISION NOT NULL,
  subject_transcript TEXT,
  post_utterance_leak_flags TEXT[] DEFAULT '{}',
  acoustic_convergence_aci DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Discovered Personal Motifs ("Tells") Table
CREATE TABLE IF NOT EXISTS discovered_personal_motifs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  motif_name VARCHAR(128) NOT NULL,
  modality VARCHAR(32) NOT NULL,
  feature_key VARCHAR(64) NOT NULL,
  point_biserial_r DOUBLE PRECISION NOT NULL,
  risk_ratio DOUBLE PRECISION NOT NULL,
  occurrence_timestamps_sec DOUBLE PRECISION[] NOT NULL,
  pattern_description TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Upgrade `session_events` with Video Verification & Advocate-Skeptic-Judge ToM Output
ALTER TABLE session_events
ADD COLUMN IF NOT EXISTS turn_id UUID REFERENCES dyadic_turns(id) ON DELETE SET NULL,
ADD COLUMN IF NOT EXISTS trigger_stimulus_quote TEXT,
ADD COLUMN IF NOT EXISTS micro_sequence_chain TEXT,
ADD COLUMN IF NOT EXISTS clip_s3_key TEXT,
ADD COLUMN IF NOT EXISTS artifact_probability DOUBLE PRECISION NOT NULL DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS artifact_verification_note TEXT,
ADD COLUMN IF NOT EXISTS advocate_hypothesis TEXT,
ADD COLUMN IF NOT EXISTS advocate_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS skeptic_counter_hypothesis TEXT,
ADD COLUMN IF NOT EXISTS skeptic_score DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS judge_synthesis_verdict TEXT,
ADD COLUMN IF NOT EXISTS epistemic_confidence DOUBLE PRECISION;
