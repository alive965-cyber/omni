---
doc_id: "doc_trimodal_matrix_2026_v1"
source_title: "Trimodal Behavioral & Physiological Fusion Matrix: Cross-Modal Congruency and State Synthesis"
authors: ["OmniPulse Scientific Advisory Board", "Scherer, K.R.", "Picard, R.W.", "Ekman, P."]
publication_year: 2026
modality_tags: ["FUSION", "INCONGRUENCE", "RPPG", "AUDIO", "FACS", "KINESICS"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Canonical Trimodal State Patterns (Heart Rate + Voice + Video + Words)

## Pattern 1: Cross-Modal Incongruence / Suppressed Reservation (`CMCI < 0.50`)
- **Multimodal Signature:**
  - **Spoken Words (NLP):** Positive or compliant semantic statement (`text_valence_score > 0.35`, e.g., *"Everything is on track"*, *"I completely agree"*).
  - **Voice (openSMILE):** Elevated local jitter ($Z_{\text{jitter}} \ge +1.8$) OR flattened monotone pitch ($Z_{F_0,\text{std}} \le -1.5$) OR pre-utterance latency $> 800\text{ ms}$.
  - **Video (FACS & Kinesics):** AU24 Lip Presser spike ($Z_{\text{AU24}} \ge +1.8$), Non-Duchenne smile ($\text{AU12} \gt 0.4, \text{AU06} \lt 0.15$), gaze aversion, or neck/face self-touch adaptor (`self_touch_ratio > 0.30`).
  - **Heart Rate (rPPG):** Sympathetic elevation ($Z_{\text{HR}} \ge +1.6$, $\text{SQI} \ge 0.45$).
- **Scientific Report Synthesis:**
  > "While the participant verbally expressed agreement/confidence, concurrent acoustic, visual, and cardiovascular telemetry exhibited significant divergence from personal baseline (CMCI < 0.50). Specifically, elevated vocal micro-tremor (jitter), post-utterance lip compression (AU24), pacifying self-touch, and sympathetic heart-rate acceleration indicate **active emotional regulation, unstated reservation, or high cognitive dissonance** regarding this topic."
- **Mandatory Alternative Explanations:** Effortful recall of complex details, polite social compliance under fatigue, or physical discomfort.

## Pattern 2: High Cognitive Load & Working Memory Strain
- **Multimodal Signature:**
  - **Spoken Words (NLP):** Slower speech rate, increased pause ratio (`pause_ratio > 0.35`), filler disfluencies (*"uh", "um"*).
  - **Voice (openSMILE):** Moderate $F_0$ elevation or increased shimmer; longer response onset latency.
  - **Video (FACS & Kinesics):** Sustained **AU04 (Brow Lowerer)** ($Z_{\text{AU04}} \ge +1.6$), cognitive gaze shift away from camera during speech planning, initial blink suppression followed by a **post-utterance blink flurry** ($Z_{\text{blink}} \ge +1.8$), and drop in illustrative hand gestures.
  - **Heart Rate (rPPG):** Moderate elevation ($+4\text{ to }+9\text{ BPM}$ above baseline).
- **Scientific Report Synthesis:**
  > "This segment reflects **high cognitive load and intensive working-memory processing**. Brow lowering (AU04), gaze aversion during speech formulation, reduced hand gesturing, and a rebound blink flurry are classic executive-control indicators as the speaker structures a complex response."

## Pattern 3: Acute Sympathetic Stress / Defensive Agitation
- **Multimodal Signature:**
  - **Spoken Words (NLP):** Negative or defensive semantic valence (`text_valence_score < -0.30`) or rapid urgent phrasing (`speech_rate_z > +1.5`).
  - **Voice (openSMILE):** High mean pitch ($Z_{F_0} \ge +1.8$), elevated loudness, elevated jitter ($Z_{\text{jitter}} \ge +1.8$), and reduced HNR ($Z_{\text{HNR}} \le -1.5$, tense phonation).
  - **Video (FACS & Kinesics):** AU04 (Brow Lowerer) + AU20/AU24 (Lip tension), elevated blink rate, repetitive hand fidgeting (`FIDGET_ADAPTOR`) or backward postural lean.
  - **Heart Rate (rPPG):** Pronounced sympathetic spike ($Z_{\text{HR}} \ge +2.2$).
- **Scientific Report Synthesis:**
  > "Co-occurring sympathetic heart-rate acceleration, elevated vocal pitch and jitter, tense phonation (low HNR), and kinesic restlessness indicate **acute autonomic arousal and defensive tension** triggered by the discussion topic."

## Pattern 4: Authentic Engagement & Concordant Confidence (`CMCI > 0.80`)
- **Multimodal Signature:**
  - **Spoken Words (NLP):** Fluent delivery, low pause ratio (`pause_ratio < 0.20`), positive or constructive valence.
  - **Voice (openSMILE):** Normative or moderately dynamic pitch contour ($F_0$ variability within or slightly above baseline), low jitter ($Z_{\text{jitter}} \lt +1.0$), high HNR (resonant voice).
  - **Video (FACS & Kinesics):** Direct mutual gaze (`gaze_aversion_ratio < 0.25`), Duchenne smile markers ($\text{AU06} + \text{AU12}$) when positive, fluid **illustrative hand gestures** synchronized with speech rhythm, slight forward postural lean.
  - **Heart Rate (rPPG):** Stable within personal baseline ($\pm 1.2\text{ MAD}$) or mild metabolic elevation proportional to lively speech.
- **Scientific Report Synthesis:**
  > "High cross-modal congruency (CMCI > 0.80) across vocal resonance, illustrative gesturing, direct gaze, and stable autonomic telemetry indicates **authentic engagement, psychological comfort, and high topic fluency**."

## Pattern 5: Cognitive Fatigue / Low-Arousal Withdrawal
- **Multimodal Signature:**
  - **Voice (openSMILE):** Decreased loudness, dropped mean $F_0$ and $F_0$ variability, slowed speech rate.
  - **Video (FACS & Kinesics):** Prolonged eye-blink closure duration (`blink_duration_ms > 300`), slumped shoulder posture, minimal facial reactivity, hand-supporting-head posture.
  - **Heart Rate (rPPG):** At or below baseline median ($Z_{\text{HR}} \le -0.5$).
- **Scientific Report Synthesis:**
  > "Progressive reductions in vocal energy, prolonged eyelid closure durations, postural slouching, and lower autonomic arousal indicate **mental fatigue, boredom, or attentional disengagement**."

---

# Section 2: Mandatory Reporting Rules for the LLM Synthesizer
1. **Require Minimum 2 Modalities for High-Confidence Claims:** Never flag a high-confidence behavioral shift based on a single isolated metric (e.g., a lone blink spike or a lone hand touch). Require convergence across at least **two independent modalities** (e.g., Acoustic + Visual, or Visual + rPPG).
2. **Baseline-Relative Quantification:** Always cite values relative to the participant's 00:00–01:30 calibration baseline using Robust $Z$-scores (MAD units) alongside absolute physical units (BPM, Hz/semitones, %, blinks/min).
3. **Strict Prohibition on Deception Labeling:** Behavioral incongruence measures **internal conflict, cognitive effort, or emotional regulation**—it is never deterministic proof of lying or deception.