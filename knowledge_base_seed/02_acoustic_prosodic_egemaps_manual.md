---
doc_id: "doc_acoustic_egemaps_2026_v1"
source_title: "openSMILE eGeMAPSv02 Paralinguistic Biomarkers and Psycholinguistic Disfluency Manual"
authors: ["Eyben, F.", "Scherer, K.R.", "Schuller, B.W.", "Giddens, C.L."]
publication_year: 2024
modality_tags: ["AUDIO", "ACOUSTIC", "PROSODY", "LINGUISTIC"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Fundamental Frequency (F0) & Pitch Contour Dynamics

## Mean Pitch (`f0_semitone_mean`) & Pitch Variability (`f0_semitone_std`)
- **Physiological Mechanism:** Tension of the cricothyroid and thyroarytenoid muscles innervated by the superior and recurrent laryngeal nerves.
- **Signal Triggers:** `f0_semitone_mean`, `f0_semitone_std`
- **Behavioral Correlates:**
  1. **Sympathetic Arousal Spike (Z_F0 >= +1.5):** Autonomic arousal increases laryngeal muscle tone, raising fundamental frequency relative to baseline.
  2. **Pitch Flattening / Monotone Constriction (Z_F0_std <= -1.5):** A marked drop in pitch variability correlates with guarded delivery, emotional dampening, or cognitive fatigue.

# Section 2: Voice Quality Micro-Perturbations (Jitter, Shimmer, HNR)

## Local Jitter (`jitter_local`) & Local Shimmer (`shimmer_local_db`)
- **Physiological Mechanism:** Micro-instabilities in vocal fold vibration period (Jitter) and glottal closure amplitude (Shimmer).
- **Signal Triggers:** `jitter_local`, `jitter_local_mean`, `shimmer_local_db`, `shimmer_local_db_mean`
- **Behavioral Correlates:**
  1. **Cognitive & Emotional Load (Z_jitter >= +1.8):** Under high cognitive load or emotional regulation, neuromuscular control of the larynx exhibits micro-tremor.
  2. **Incongruent Vocal Strain:** When a speaker delivers a positive verbal assurance (`text_valence_score > 0.35`) while vocal jitter spikes, cross-modal congruency drops (`CMCI < 0.55`).
- **Mandatory Confounders:** Low microphone SNR (< 15 dB), phrase-final vocal fry, or throat dryness.

## Harmonic-to-Noise Ratio (`hnr_db`)
- **Signal Triggers:** `hnr_db`, `hnr_db_mean`
- **Behavioral Correlates:**
  1. A drop in HNR (Z_HNR <= -1.5) paired with elevated pitch and jitter reflects pressed or breathy phonation under vocal tension.

# Section 3: Temporal Prosody, Response Latency, & Disfluencies

## Speech Rate (`speech_rate_wpm`), Pause Ratio (`pause_ratio`), & Fillers (`disfluency_count`)
- **Signal Triggers:** `speech_rate_wpm`, `pause_ratio`, `disfluency_count`, `response_latency_ms`
- **Behavioral Correlates:**
  1. **Pre-Utterance Response Latency (> 800 ms above baseline):** An abnormally long silent planning gap before answering indicates high cognitive appraisal load.
  2. **Hesitation & Disfluency Cluster (`pause_ratio > 0.35` + `disfluency_count >= 2`):** Elevated unfilled pauses and filler tokens reflect real-time speech planning conflict or lexical search.
