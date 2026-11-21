---
doc_id: "doc_acoustic_egemaps_2026_v1"
source_title: "openSMILE eGeMAPSv02 Paralinguistic Biomarkers and Psycholinguistic Disfluency Manual"
authors: ["Eyben, F.", "Scherer, K.R.", "Schuller, B.W.", "Giddens, C.L."]
publication_year: 2024
modality_tags: ["AUDIO", "ACOUSTIC", "PROSODY", "LINGUISTIC"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Fundamental Frequency ($F_0$) & Pitch Contour Dynamics

## Mean Pitch (`f0_semitone_mean`) & Pitch Variability (`f0_semitone_std`)
- **Physiological Mechanism:** Tension of the cricothyroid and thyroarytenoid muscles innervated by the superior and recurrent laryngeal branches of the vagus nerve (Cranial Nerve X).
- **Signal Triggers:** `f0_semitone_mean`, `f0_semitone_std`, `f0_z > 1.5`, `f0_std_z < -1.5`
- **Behavioral Correlates:**
  1. **Sympathetic Arousal Spike ($Z_{F_0} \ge +1.5$):** Autonomic arousal increases laryngeal muscle tone, raising mean $F_0$. When paired with positive text valence and Duchenne smiling (AU06+AU12), it reflects **enthusiasm/engagement**. When paired with negative valence, AU04, or AU24, it reflects **acute stress, urgency, or agitation**.
  2. **Pitch Flattening / Monotone Constriction ($Z_{F_0,\text{std}} \le -1.5$):** A marked drop in pitch variability (monotone delivery) during an emotionally charged or affirmative statement correlates with **vocal control/suppression**, fatigue, or low affect.

---

# Section 2: Voice Quality Micro-Perturbations (Jitter, Shimmer, HNR)

## Local Jitter (`jitter_local`) & Local Shimmer (`shimmer_local_db`)
- **Physiological Mechanism:** Micro-instabilities in vocal fold vibration period (Jitter) and glottal closure amplitude (Shimmer) caused by autonomic fluctuations in laryngeal neuromuscular control and mucosal hydration.
- **Signal Triggers:** `jitter_local`, `shimmer_local_db`, `jitter_z > 1.8`, `shimmer_z > 1.8`
- **Behavioral Correlates:**
  1. **Cognitive & Emotional Load ($Z_{\text{jitter}} \ge +1.8$):** Under high cognitive load or emotional regulation, fine motor control of the intrinsic laryngeal muscles experiences micro-tremors, elevating local jitter and shimmer relative to the speaker's calm baseline.
  2. **Incongruent Vocal Strain:** When a speaker delivers a calm or positive verbal assurance (`text_valence_score > 0.4`) while `jitter_z > 1.8` and `shimmer_z > 1.5`, this indicates paralinguistic strain beneath composed verbal phrasing.
- **Mandatory Confounders:** Low microphone SNR ($< 15\text{ dB}$), vocal fry at the end of declarative sentences, upper respiratory congestion, throat clearing, or room reverberation.

## Harmonic-to-Noise Ratio (`hnr_db`)
- **Physiological Mechanism:** Ratio of periodic harmonic energy from clean vocal fold adduction to turbulent glottal airflow noise.
- **Signal Triggers:** `hnr_db`, `hnr_z < -1.5`
- **Behavioral Correlates:**
  1. A drop in HNR ($Z_{\text{HNR}} \le -1.5$) paired with elevated pitch and jitter reflects **pressed or breathy phonation** associated with vocal tension or autonomic stress.

---

# Section 3: Temporal Prosody, Response Latency, & Disfluencies

## Speech Rate (`speech_rate_wpm`), Pause Ratio (`pause_ratio`), & Fillers (`disfluency_count`)
- **Signal Triggers:** `speech_rate_wpm`, `pause_ratio`, `disfluency_count`, `response_latency_ms`
- **Behavioral Correlates:**
  1. **Pre-Utterance Response Latency (> 800 ms above baseline):** An abnormally long silent gap before answering a direct question—especially when paired with AU04 (brow furrow), gaze aversion, and an rPPG heart-rate shift—indicates high cognitive appraisal, memory search, or careful word selection.
  2. **Speech Rate Acceleration ($Z_{\text{wpm}} \ge +1.6$):** Rapid speaking pace correlates with high arousal, eagerness, or an attempt to move quickly past an uncomfortable topic.
  3. **Hesitation & Disfluency Cluster (`pause_ratio > 0.35` + `disfluency_count >= 2`):** Elevated unfilled pauses and filled pauses (*"uh", "um"*) indicate real-time speech planning load, uncertainty, or lexical retrieval difficulty.