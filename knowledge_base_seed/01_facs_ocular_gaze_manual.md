---
doc_id: "doc_facs_ocular_2026_v1"
source_title: "Facial Action Coding System (FACS), Ocular Blink Dynamics, and Gaze Telemetry Reference Manual"
authors: ["Ekman, P.", "Friesen, W.V.", "Hager, J.C.", "Soukupova, T.", "Stern, J.A."]
publication_year: 2024
modality_tags: ["FACS", "VISUAL", "OCULAR", "GAZE"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Upper Face Action Units (Brow & Forehead Dynamics)

## AU01 (Inner Brow Raiser) & AU02 (Outer Brow Raiser)
- **Anatomical Mechanism:** Frontalis, pars medialis (AU01) and pars lateralis (AU02).
- **Signal Triggers:** `au01_inner_brow`, `au02_outer_brow`, `au01_z > 1.5`, `au02_z > 1.5`
- **Behavioral Correlates:**
  1. When AU01 + AU02 fire bilaterally alongside jaw drop (AU25/26) for less than 1.5 seconds, they serve as a conversational emphasis or surprise marker.
  2. Sustained isolated AU01 (or AU01 + AU04 yielding oblique eyebrows) correlates with negative valence, concern, or sadness.
- **Mandatory Confounders:** Speakers with high prosodic expressiveness naturally raise eyebrows on stressed syllables.

## AU04 (Brow Lowerer / Corrugator Supercilii)
- **Anatomical Mechanism:** Corrugator supercilii and depressor supercilii drawing the eyebrows down and medially.
- **Signal Triggers:** `au04_brow_lowerer`, `au04_z > 1.5`
- **Behavioral Correlates:**
  1. **Cognitive Load & Perplexity:** Sustained AU04 elevation (Z >= 1.5) during listening or prior to answering reflects working-memory effort or processing difficulty.
  2. **Negative Valence / Frustration:** When paired with AU24 (Lip Presser) or elevated vocal loudness/shimmer, AU04 reflects controlled tension or appraisal resistance.
- **Mandatory Confounders:** Bright ambient screen glare, reading small text on a monitor, or myopia squinting.

# Section 2: Lower Face Action Units, Smiles, & Response Inhibition

## AU06 (Cheek Raiser) + AU12 (Lip Corner Puller) — Smile Differentiation
- **Anatomical Mechanism:** Zygomaticus major (AU12) pulls lip corners superolaterally; Orbicularis oculi, pars orbitalis (AU06) raises cheek and narrows eye aperture.
- **Signal Triggers:** `au12_lip_corner_puller`, `au06_cheek_raiser`, `facial_asymmetry_index`
- **Behavioral Correlates:**
  1. **Concordant Positive Affect (Duchenne Marker):** AU12 > 0.45 accompanied by AU06 > 0.30 and symmetric onset/offset correlates with genuine positive engagement.
  2. **Social / Politeness / Masking Smile (Non-Duchenne):** AU12 > 0.45 with low orbicularis oculi activity (AU06 < 0.15) or rapid snap-off offset functions as social appeasement or polite masking.

## AU20 (Lip Stretcher) & AU24 (Lip Presser)
- **Anatomical Mechanism:** Risorius draws lips laterally (AU20); Orbicularis oris presses upper and lower lips tightly together (AU24).
- **Signal Triggers:** `au24_lip_presser`, `au20_lip_stretcher`, `au24_z > 1.8`
- **Behavioral Correlates:**
  1. **Active Response Inhibition:** A spike in AU24 (Z >= 1.8) immediately before speaking, during a silent pause, or during post-utterance leakage reflects active verbal restraint or self-censorship.
  2. **Autonomic Tension:** AU24 paired with elevated vocal jitter or a rise in rPPG heart rate indicates heightened psychophysiological regulation.
- **Mandatory Confounders:** Bilabial consonant articulation (`/p/`, `/b/`, `/m/`), swallowing, or dry lips.

# Section 3: Eye-Blink Telemetry (EAR) & Gaze Dynamics

## Spontaneous Blink Rate (`blink_rate_per_min`) & Blink Flurry
- **Physiological Mechanism:** Central dopaminergic activity and pre-motor trigeminal-oculomotor gating modulated by working-memory load.
- **Signal Triggers:** `blink_rate_per_min`, `blink_rate_z > 1.5`, `blink_duration_ms_mean`
- **Behavioral Correlates:**
  1. **Cognitive Load Suppression (Freeze Phase):** During intense visual attention or high working-memory calculation, blink rate transiently drops below baseline.
  2. **Post-Load Blink Burst (Rebound Flurry):** Immediately following resolution of a demanding question, a rapid blink flurry (Z >= +1.8) releases oculomotor inhibition.
  3. **Prolonged Closure (`blink_duration_ms_mean > 300`):** Correlates with mental fatigue or drowsiness.
- **Mandatory Confounders:** Dry ambient air, contact lens irritation, air conditioning drafts, or screen refresh flicker.

## Gaze Aversion (`gaze_aversion_ratio`) & Saccadic Activity
- **Signal Triggers:** `gaze_aversion_ratio`, `saccade_velocity_deg_s`
- **Behavioral Correlates:**
  1. **Cognitive Floor/Ceiling Gaze Shift:** Breaking mutual gaze while formulating an answer reduces visual input to free up working memory.
  2. **Defensive / Discomfort Avoidance:** Gaze aversion occurring while listening to a challenging prompt alongside AU24 or self-touch indicates interpersonal discomfort.
- **Scientific Anti-Pseudoscience Rule:** NEVER equate gaze direction (e.g., looking up-right vs. up-left) with constructed lies.
