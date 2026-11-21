---
doc_id: "doc_situational_skeptic_atlas_2026_v1"
source_title: "Situational Behavioral Priors (Interview, Pitch, Viva) and The Pragmatic Skeptic's Differential Confounder Atlas"
authors: ["OmniPulse Methodology Board", "Harrigan, J.A.", "Scherer, K.R."]
publication_year: 2026
modality_tags: ["FUSION", "CULTURAL", "FACS", "AUDIO", "RPPG", "KINESICS"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Situational Context Priors

## Archetype A: Technical Software / Engineering Placement Interview
- **Normative Baseline Adjustment:** Solving algorithmic, architectural, or debugging questions requires heavy visuospatial and phonological working memory.
- **Context Rule:** Do NOT classify isolated `AU04 (Brow Lowerer)`, `gaze_aversion_ratio`, or mid-sentence `pause_ratio` spikes as emotional discomfort during technical explanations. Only flag emotional or interpersonal strain when accompanied by **post-utterance `AU24` lip seals**, **pronoun distancing (`"I"` drop)**, **self-touch adaptors**, or **vocal jitter/shimmer spikes ($Z \ge 2.0$)**.

## Archetype B: Academic Viva / Oral Defense & High-Stakes Presentation
- **Normative Baseline Adjustment:** Tonic autonomic arousal (`hr_bpm_pos`, `f0_semitone_mean`) is naturally elevated throughout the first 3 minutes due to social-evaluative threat (cortisol/sympathetic baseline shift).
- **Context Rule:** Evaluate performance resilience via **phasic recovery speed**—how quickly vocal HNR and speech fluency stabilize after a challenging examiner follow-up question.

---

# Section 2: Differential Confounder Atlas (For the Pragmatic Skeptic Agent)

## Acoustic & Phonetic Confounders
- **Glottalization / End-of-Sentence Vocal Fry:** Speakers frequently drop subglottal air pressure at the end of declarative sentences, causing creaky voice (vocal fry) that artificially spikes `jitter_local_mean` and `shimmer_local_db_mean` while dropping `hnr_db_mean`. **Skeptic Check:** If jitter/shimmer spikes occur *only* on the final 500ms of declarative clauses with falling $F_0$, classify as benign phrase-final glottalization.
- **Prosodic Contrastive Stress:** Emphasizing a key technical word naturally spikes `f0_semitone_mean`, `loudness_mean`, and `au01_inner_brow + au02_outer_brow`.

## Visual, Ocular, & Ergonomic Confounders
- **Monitor Illumination & Dark/Light Mode Switching:** When a user switches windows on their laptop screen from dark mode to a bright white browser tab, pupil constriction, squint (`AU06 + AU04`), and skin RGB luminance shift simultaneously—corrupting both `rPPG` and `AU04` for 3–5 seconds.
- **Swallowing & Dry-Mouth Regulation:** Normal deglutition (swallowing saliva) occurs every $45\text{–}90\text{ seconds}$ during continuous speech and produces a transient $600\text{ ms}$ `AU24` lip press + laryngeal movement.