---
doc_id: "doc_facs_ocular_2026_v1"
source_title: "Facial Action Coding System (FACS), Ocular Blink Dynamics, and Gaze Telemetry Reference Manual"
authors: ["Ekman, P.", "Friesen, W.V.", "Hager, J.C.", "Soukupova, T.", "Stern, J.A."]
publication_year: 2024
modality_tags: ["FACS", "VISUAL", "OCULAR", "GAZE"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Upper Face Action Units (Brow & Forehead Dynamics)

## AU01 (Inner Brow Raiser) & AU02 (Outer Brow Raiser)
- **Anatomical Mechanism:** Frontalis, pars medialis (AU01) and pars lateralis (AU02).
- **Signal Triggers:** `au01_inner_brow`, `au02_outer_brow`, `au01_z > 1.5`, `au02_z > 1.5`
- **Behavioral Correlates:**
  1. When AU01 + AU02 fire bilaterally alongside jaw drop (AU25/26) for less than 1.5 seconds, they serve as a conversational **baton/illustrator** (emphasizing a spoken word) or an **orienting surprise response**.
  2. Sustained isolated **AU01** (or AU01 + AU04 yielding oblique eyebrows) correlates with negative valence, concern, or sadness/apprehension.
- **Mandatory Confounders:** Speakers with high prosodic expressiveness naturally raise eyebrows on stressed syllables ($F_0$ peaks). Do not interpret AU01+AU02 as apprehension if aligned with prosodic emphasis peaks.

## AU04 (Brow Lowerer / Corrugator Supercilii)
- **Anatomical Mechanism:** Corrugator supercilii and depressor supercilii drawing the eyebrows down and medially, producing vertical glabellar furrows.
- **Signal Triggers:** `au04_brow_lowerer`, `au04_z > 1.5`
- **Behavioral Correlates:**
  1. **Cognitive Load & Perplexity:** Sustained AU04 elevation ($Z \ge 1.5$) during listening or prior to answering complex questions correlates strongly with mental effort, working memory retrieval, or difficulty processing a prompt.
  2. **Negative Valence / Frustration:** When paired with AU24 (Lip Presser) or elevated vocal loudness/shimmer, AU04 reflects resistance, irritation, or unpleasant affect.
- **Mandatory Confounders:** Bright ambient screen glare, reading small text on a monitor, myopia (squinting without glasses), or concentration on technical problem-solving.

---

# Section 2: Lower Face Action Units, Smiles, & Response Inhibition

## AU06 (Cheek Raiser) + AU12 (Lip Corner Puller) — Smile Differentiation
- **Anatomical Mechanism:** Zygomaticus major (AU12) pulls lip corners superolaterally; Orbicularis oculi, pars orbitalis (AU06) raises the infraorbital triangle and wrinkles lateral eye corners.
- **Signal Triggers:** `au12_lip_corner_puller`, `au06_cheek_raiser`, `facial_asymmetry_index`
- **Behavioral Correlates:**
  1. **Concordant Positive Affect (Duchenne Marker):** $\text{AU12} \gt 0.45$ accompanied by $\text{AU06} \gt 0.30$ and symmetrical onset/offset ($\text{FAI} \lt 0.15$) correlates with genuine amusement, relief, or rapport.
  2. **Social / Politeness / Masking Smile (Non-Duchenne):** $\text{AU12} \gt 0.45$ with low orbicularis oculi activation ($\text{AU06} \lt 0.15$) or high unilateral asymmetry ($\text{FAI} \gt 0.25$) acts as a socialregulator, conversational appeasement, or emotional masking when accompanied by elevated vocal jitter or heart rate.

## AU20 (Lip Stretcher) & AU24 (Lip Presser)
- **Anatomical Mechanism:** Risorius draws lips laterally (AU20); Orbicularis oris presses upper and lower lips tightly together, narrowing the vermilion border (AU24).
- **Signal Triggers:** `au24_lip_presser`, `au20_lip_stretcher`, `au24_z > 1.8`
- **Behavioral Correlates:**
  1. **Active Response Inhibition:** A spike in **AU24** ($Z \ge 1.8$) immediately before speaking, during a silent pause, or immediately after finishing a brief statement correlates with withholding further verbal commentary, self-censorship, or regulatory tension.
  2. **Autonomic Tension:** AU24 paired with elevated vocal jitter or a rise in rPPG heart rate indicates heightened psychological tension during topic evaluation.
- **Mandatory Confounders:** Bilabial consonant articulation (`/p/`, `/b/`, `/m/`), swallowing, dry lips, or habitual resting lip compression. Only flag AU24 during non-speech intervals or when sustained $> 400\text{ ms}$.

---

# Section 3: Eye-Blink Telemetry (EAR) & Gaze Dynamics

## Spontaneous Blink Rate (`blink_rate_per_min`) & Blink Flurry
- **Physiological Mechanism:** Central dopaminergic activity and pre-motor trigeminal-oculomotor gating modulated by attentional load in the prefrontal cortex.
- **Signal Triggers:** `blink_rate_per_min`, `blink_rate_z > 1.5`, `blink_rate_z < -1.5`, `blink_duration_ms`
- **Behavioral Correlates:**
  1. **Cognitive Load Suppression (Freeze Phase):** During intense visual attention or high working-memory calculation, blink rate transiently drops below baseline ($Z \le -1.2$).
  2. **Post-Load Blink Burst (Rebound Flurry):** Immediately following the resolution of a difficult question or high cognitive demand, blink rate spikes sharply above baseline ($Z \ge +1.8$, often $30\text{–}50\text{ blinks/min}$).
  3. **Sustained Elevated Blinking:** Persistent blink elevation ($Z \ge +1.8$) alongside gaze aversion and vocal disfluencies correlates with heightened arousal, nervousness, or cognitive interference.
  4. **Prolonged Closure (`blink_duration_ms > 300`):** Correlates with mental fatigue, drowsiness, or momentary cognitive withdrawal.
- **Mandatory Confounders:** Dry ambient air, contact lens irritation, air conditioning drafts, screen refresh flicker, or looking upward/downward (which geometrically alters 2D EAR unless pitch-compensated).

## Gaze Aversion (`gaze_aversion_ratio`) & Saccadic Activity
- **Signal Triggers:** `gaze_aversion_ratio`, `saccade_velocity_deg_s`, `gaze_aversion_z > 1.5`
- **Behavioral Correlates:**
  1. **Cognitive Floor/Ceiling Gaze Shift:** Breaking mutual gaze while formulating an answer (`gaze_aversion_ratio > 0.40` during speech onset) is a normal **cognitive load management strategy**—reducing visual input from the interlocutor's face to free up working memory.
  2. **Defensive / Discomfort Avoidance:** Gaze aversion that occurs **while listening** to a challenging topic or immediately after delivering a short, clipped statement—especially when paired with AU24 (Lip Press) and neck/face self-touch—correlates with topic discomfort or reluctance.
- **Scientific Anti-Pseudoscience Rule:** NEVER equate gaze direction (e.g., looking up-right vs. up-left) with "constructing lies" vs. "recalling truth." NLP eye-accessing cues have been empirically refuted in peer-reviewed literature (Wiseman et al., 2012).