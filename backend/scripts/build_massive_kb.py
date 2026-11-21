"""Complete Production 3-Tier Knowledge Base Builder for OmniPulse MBS (backend/scripts/build_massive_kb.py).

Combines (PDF Specification Pages 22-30 & 52-58):
  1. Tier 1 (`TIER_1_CANONICAL_RULE`): All 9 Canonical Markdown Seed Manuals (01 through 09).
     Automatically bootstraps all 9 Markdown seed files into `knowledge_base_seed/` if missing!
  2. Tier 2 (`TIER_2_COMBINATORIAL_MATRIX`): 648-Permutation Combinatorial Multimodal Decision Matrix
     (6 FACS States x 4 Voice States x 3 rPPG Heart States x 3 Kinesic States x 3 Linguistic Contexts)
     with pre-built Signal Advocate Interpretations and Pragmatic Skeptic Counter-Hypotheses.
  3. Tier 3 (`TIER_3_EMPIRICAL_STUDY`): Auto-Tagged EuropePMC Open-Access Peer-Reviewed Studies (1,500+ chunks)
     with HTML cleaning, 2-part abstract splitting, and automatic `modality_tags` + `signal_triggers` detection.

Upserts all 3 tiers into both SQLite (`repository`) and Qdrant (`knowledge_retrieval`) using local ONNX FastEmbed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import knowledge_retrieval, repository  # noqa: E402
from app.services.knowledge import chunk_document, integrity_status, parse_yaml_frontmatter  # noqa: E402

LOGGER = logging.getLogger("omnipulse.kb_builder")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ============================================================================
# TIER 1: 9 CANONICAL SEED MANUALS BOOTSTRAP (PDF Pages 22-29)
# ============================================================================
SEED_MANUALS: dict[str, str] = {
    "01_facs_ocular_gaze_manual.md": """---
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
""",
    "02_acoustic_prosodic_egemaps_manual.md": """---
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
""",
    "03_kinesics_posture_adaptors_manual.md": """---
doc_id: "doc_kinesics_adaptors_2026_v1"
source_title: "Kinesic Self-Adaptors, Hand Motion Energy, and Postural Telemetry Guide"
authors: ["Ekman, P.", "Friesen, W.V.", "Navarro, J.", "Freedman, N.", "Harrigan, J.A."]
publication_year: 2024
modality_tags: ["KINESICS", "VISUAL", "POSTURE", "GESTURES"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Self-Adaptors (Pacifying Touch Behaviors)

## Face, Neck, and Suprasternale Touch (`self_touch_ratio`)
- **Neurobiological Mechanism:** Tactile stimulation of the face (trigeminal nerve, CN V) and anterior neck/carotid sinus activates parasympathetic vagal pathways to down-regulate arousal.
- **Signal Triggers:** `self_touch_ratio`, `hand_kinetic_energy`
- **Behavioral Correlates:**
  1. **Emotional & Cognitive Pacifying:** Sudden onset of hand-to-neck, suprasternal notch, or chin/nose contact (>300ms) during or immediately after a challenging prompt signals self-soothing regulation.
- **Mandatory Confounders:** Itching, adjusting eyeglasses or headset microphone, or resting the chin on a hand during prolonged listening.

# Section 2: Hand Kinetic Energy — Illustrators vs. Fidgeting

## Hand Motion Disambiguation (`hand_kinetic_energy`, `gesture_type`)
- **Signal Triggers:** `hand_kinetic_energy`, `postural_lean_z`
- **Behavioral Correlates:**
  1. **Illustrative Gestures (`gesture_type == "ILLUSTRATIVE"`):** High hand kinetic energy synchronized with active speech reflects communicative engagement and fluency.
  2. **Illustrator Drop / Freeze (Z_hand_ke <= -1.5 during speech):** Sudden cessation of habitual hand gestures during a complex answer reflects cognitive resource reallocation.
  3. **Repetitive Hand Fidgeting (`gesture_type == "FIDGET_ADAPTOR"`):** High-frequency, small-radius hand-to-hand rubbing correlates with sensorimotor regulation or nervous tension.
""",
    "04_rppg_cardiovascular_autonomic_manual.md": """---
doc_id: "doc_rppg_autonomic_2026_v1"
source_title: "Remote Photoplethysmography (rPPG POS/CHROM) and Autonomic Cardiovascular Psychophysiology Manual"
authors: ["Wang, W.", "de Haan, G.", "Berntson, G.G.", "Cacioppo, J.T.", "Porges, S.W."]
publication_year: 2025
modality_tags: ["RPPG", "PHYSIOLOGY", "HEART_RATE", "AUTONOMIC"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Autonomic Nervous System (ANS) Interpretation of rPPG Heart Rate

## Sympathetic Acceleration (Phasic Tachycardia: `hr_bpm_z >= +1.8`)
- **Physiological Mechanism:** Withdrawal of vagal parasympathetic brake (nucleus ambiguus) followed by sympathetic beta-adrenergic stimulation of the sinoatrial node.
- **Signal Triggers:** `hr_bpm`, `hr_bpm_pos`, `rppg_sqi`
- **Behavioral Correlates:**
  1. **Defensive Arousal / Acute Stress:** A sustained rise of >= +8 BPM (or Z_HR >= +1.8) with high phasic delta reflects acute autonomic mobilization.
  2. **High Cognitive Effort:** Complex mental calculation or defending a high-stakes technical decision elevates heart rate alongside AU04 brow lowering.

## Mandatory rPPG Signal Quality & Confounder Constraints
- **Signal Triggers:** `hr_bpm`, `hr_bpm_pos`, `rppg_sqi`
- **Strict Interpretation Rules:**
  1. **High Confidence (`rppg_sqi >= 0.55` and `|POS - CHROM| <= 5.0 BPM`):** Cardiovascular trend can be weighted alongside vocal and visual channels.
  2. **Unreliable / Masked (`rppg_sqi < 0.38`):** Completely exclude heart rate from behavioral interpretation.
- **Physiological Confounders:** Continuous rapid speaking alters intrathoracic pressure and induces Respiratory Sinus Arrhythmia (RSA) heart-rate elevation.
""",
    "05_trimodal_fusion_incongruence_matrix.md": """---
doc_id: "doc_trimodal_matrix_2026_v1"
source_title: "Trimodal Behavioral & Physiological Fusion Matrix: Cross-Modal Congruency and State Synthesis"
authors: ["OmniPulse Scientific Advisory Board", "Scherer, K.R.", "Picard, R.W.", "Ekman, P."]
publication_year: 2026
modality_tags: ["FUSION", "INCONGRUENCE", "RPPG", "AUDIO", "FACS", "KINESICS"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Canonical Trimodal State Patterns (Heart Rate + Voice + Video + Words)

## Pattern 1: Cross-Modal Incongruence / Suppressed Reservation (`CMCI < 0.50`)
- **Signal Triggers:** `cross_modal_congruency_index`, `text_valence_score`, `jitter_local`, `au24_lip_presser`, `hr_bpm_pos`
- **Multimodal Signature:** Positive or compliant semantic statement (`text_valence_score > 0.35`) paired with elevated vocal jitter (Z >= +1.8), AU24 lip presser spike (Z >= +1.8) or Non-Duchenne smile, and sympathetic heart-rate elevation (Z >= +1.6).
- **Scientific Report Synthesis:** While the participant verbally expressed agreement or comfort, concurrent acoustic, visual, and cardiovascular indicators reflect internal reservation, cognitive effort, or active emotional regulation.
- **Mandatory Alternative Explanations:** Effortful recall of complex schedule details, polite social compliance under fatigue, or L2 phrasing effort.
""",
    "06_psycholinguistics_pronoun_discourse_manual.md": """---
doc_id: "doc_psycholinguistics_liwc_2026_v1"
source_title: "Psycholinguistic Discourse Markers, Pronoun Distancing, and Verbal Defense Mechanisms (LIWC Framework)"
authors: ["Pennebaker, J.W.", "Newman, M.L.", "Tausczik, Y.R.", "Hancock, J.T."]
publication_year: 2025
modality_tags: ["LINGUISTIC", "NLP", "DISCOURSE", "INCONGRUENCE"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Psychological Distancing via Pronoun Shifts & Epistemic Hedging

## First-Person Singular Drop ("I / Me / My" Withdrawal) & Qualifier Inflation
- **Signal Triggers:** `first_person_pronoun_ratio`, `text_valence_score`, `disfluency_count`, `au24_lip_presser`
- **Psycholinguistic Mechanism:** Self-reference words ("I", "me", "my") reflect personal ownership. A sudden shift to collective/passive phrasing ("we", "the team", "it was handled") paired with epistemic hedges ("basically", "sort of", "to be honest") and AU24 lip compression or gaze aversion indicates psychological distancing from a sensitive topic.
- **Mandatory Confounders:** Collaborative engineering norms where speakers naturally credit team-wide workflows.
""",
    "07_bilingual_l2_cultural_neurodivergent_calibration.md": """---
doc_id: "doc_cultural_l2_neuro_2026_v1"
source_title: "Cross-Cultural Kinesics, Second-Language (L2) Cognitive Load, and Neurodivergent Calibration Manual"
authors: ["Matsumoto, D.", "Kroll, J.F.", "Bavelas, J.B.", "Cohn, J.F."]
publication_year: 2026
modality_tags: ["CULTURAL", "L2_LANGUAGE", "NEURODIVERGENCE", "CALIBRATION", "FACS", "AUDIO"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Second-Language (L2) Speaking, South Asian Head Wobble, & Neurodivergent Stimming

## L2 Lexical Retrieval Load vs. Emotional Stress & Cultural Kinesics
- **Signal Triggers:** `pause_ratio`, `disfluency_count`, `au04_brow_lowerer`, `gaze_aversion_ratio`, `hand_kinetic_energy`
- **Critical Disambiguation Rules:**
  1. **Pure L2 / Hinglish Lexical Search:** Characterized by mid-sentence pauses before technical nouns/verbs, upward/lateral gaze shift, and AU04 brow furrow with stable heart rate and absence of AU24 lip compression.
  2. **South Asian Head Roll Oscillation:** Smooth lateral head tilt oscillation accompanied by concordant prosody functions as an affirmative backchannel in Indian communicative norms, not equivocation.
  3. **Neurodivergent Motor Regulation:** Because OmniPulse normalizes against the participant's own baseline MAD, continuous baseline fidgeting or gaze aversion in ADHD/autistic speakers is protected from false flagging.
""",
    "08_temporal_dynamics_vagal_recovery_manual.md": """---
doc_id: "doc_temporal_vagal_recovery_2026_v1"
source_title: "Temporal Micro-Dynamics, Facial Onset/Offset Kinetics, and Autonomic Vagal Recovery Trajectories"
authors: ["Yan, W.J.", "Porges, S.W.", "Thayer, J.F.", "Cohn, J.F.", "Levenson, R.W."]
publication_year: 2025
modality_tags: ["TEMPORAL", "RPPG", "FACS", "AUTONOMIC", "RECOVERY"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: Facial Expression Kinetics (Smile Snap-Off) & Autonomic Vagal Recovery

## Abrupt Smile Drop (`smile_offset_velocity > 1.80`) & Vagal Rebound vs. Rumination
- **Signal Triggers:** `au12_lip_corner_puller`, `au06_cheek_raiser`, `hr_bpm`, `hr_bpm_pos`
- **Behavioral Correlates:**
  1. **Abrupt Smile Drop ("Snap-Off Offset"):** When a posed smile (AU12 > 0.45, AU06 < 0.15) drops to baseline in < 250 ms (`smile_offset_velocity > 1.80 s^-1`), it reflects cortical switching rather than subcortical positive affect decay.
  2. **Autonomic Carryover / Rumination:** When heart rate (`hr_bpm_z_floor >= 1.6`) remains elevated > 30s after a topic change (`delta_z_phasic ≈ 0`), it reflects ongoing cognitive rumination from a prior turn rather than reaction to the new prompt.
""",
    "09_compound_facs_dyads_triads_encyclopedia.md": """---
doc_id: "doc_compound_facs_triads_2026_v1"
source_title: "Encyclopedia of Compound FACS Action Unit Dyads and Triads in Multimodal Affective Computing"
authors: ["Ekman, P.", "Friesen, W.V.", "Du, S.", "Martinez, A.M."]
publication_year: 2025
modality_tags: ["FACS", "VISUAL", "COMPOUND_EMOTION"]
evidence_tier: "TIER_1_CANONICAL_RULE"
---
# Section 1: High-Frequency Compound FACS Combinations

## Compound Dyads & Triads (`AU04 + AU24`, `AU01 + AU04 + AU15`, `AU12 + AU15 + AU24`)
- **Signal Triggers:** `au01_inner_brow`, `au02_outer_brow`, `au04_brow_lowerer`, `au12_lip_corner_puller`, `au15_lip_corner_depressor`, `au24_lip_presser`, `facial_asymmetry_index`
- **Behavioral Correlates:**
  1. **Dyad `AU04 + AU24` (Brow Lowerer + Lip Presser):** Controlled frustration, determination, or active disagreement while listening.
  2. **Dyad `AU12 (Asymmetric) + AU04` (`facial_asymmetry_index > 0.25`):** Skeptical evaluation, irony, or ambivalence.
  3. **Triad `AU12 + AU15 + AU24` (Ambivalent / Suppressed Smile-Frown):** Politeness masking over underlying concern or disappointment.
""",
}


def ensure_seed_manuals_exist(seed_dir: Path) -> list[Path]:
    """Ensure all 9 canonical Markdown seed files from the PDF exist in `seed_dir`."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in SEED_MANUALS.items():
        target = seed_dir / filename
        if not target.exists():
            target.write_text(content.strip() + "\n", encoding="utf-8")
    return sorted(seed_dir.glob("*.md"))


# ============================================================================
# TIER 2: 648-PERMUTATION COMBINATORIAL MULTIMODAL DECISION MATRIX (PDF Pages 53-55)
# ============================================================================
FACS_STATES = [
    {
        "code": "AU04_BROW_LOWER",
        "triggers": ["au04_brow_lowerer"],
        "desc": "Sustained glabellar brow lowering (AU04 Z >= 1.6)",
        "advocate": "indicates elevated cognitive load, working-memory effort, or negative appraisal of the prompt",
        "skeptic": "may reflect visual squinting at screen elements or concentration on technical phrasing",
    },
    {
        "code": "AU24_LIP_PRESS_INHIBITION",
        "triggers": ["au24_lip_presser"],
        "desc": "Tight orbicularis oris lip compression (AU24 Z >= 1.8) during pause or post-utterance",
        "advocate": "signifies active response inhibition, self-censorship, or withholding of additional verbal commentary",
        "skeptic": "can occur during bilabial consonant preparation (/p/, /b/, /m/) or swallowing/dry-lip wetting",
    },
    {
        "code": "NON_DUCHENNE_MASKED_SMILE",
        "triggers": ["au12_lip_corner_puller", "au06_cheek_raiser"],
        "desc": "Zygomaticus lip corner pull (AU12 > 0.45) without orbicularis oculi cheek raise (AU06 < 0.15) or with rapid snap-off offset",
        "advocate": "demonstrates deliberate cortical social masking or polite appeasement concealing underlying tension",
        "skeptic": "frequently functions as a culturally normative conversational backchannel or polite greeting",
    },
    {
        "code": "DUCHENNE_CONCORDANT_SMILE",
        "triggers": ["au12_lip_corner_puller", "au06_cheek_raiser"],
        "desc": "Symmetrical AU12 + AU06 activation with smooth ballistic onset and gradual offset",
        "advocate": "reflects authentic positive affect, rapport, or relief following topic resolution",
        "skeptic": "can occasionally accompany self-deprecating humor or nervous laughter under social scrutiny",
    },
    {
        "code": "BLINK_FLURRY_AND_GAZE_AVERT",
        "triggers": ["blink_rate_per_min", "gaze_aversion_ratio"],
        "desc": "Elevated spontaneous blink flurry (Z >= 1.8) paired with lateral/downward gaze aversion",
        "advocate": "correlates with post-appraisal cognitive load release or visual avoidance of interpersonal scrutiny",
        "skeptic": "breaking mutual gaze during speech planning is a standard working-memory optimization mechanism",
    },
    {
        "code": "ASYMMETRIC_CONTEMPT_OR_DOUBT",
        "triggers": ["facial_asymmetry_index", "au12_lip_corner_puller", "au15_lip_corner_depressor"],
        "desc": "Unilateral lip corner tightening or mixed AU12+AU15 ambivalence (FAI > 0.25)",
        "advocate": "signals skeptical evaluation, irony, or affective ambivalence toward the statement",
        "skeptic": "some speakers exhibit habitual anatomical motor asymmetry during rapid speech articulation",
    },
]

VOICE_STATES = [
    {
        "code": "VOCAL_JITTER_SHIMMER_SPIKE",
        "triggers": ["jitter_local", "jitter_local_mean", "shimmer_local_db", "shimmer_local_db_mean", "hnr_db"],
        "desc": "Elevated laryngeal micro-tremor (Jitter Z >= 1.8, Shimmer Z >= 1.5, reduced HNR)",
        "advocate": "reflects vagal/sympathetic perturbation of intrinsic laryngeal muscle tone under acute emotional or cognitive strain",
        "skeptic": "may be influenced by end-of-phrase vocal fry, throat dryness, or low microphone SNR",
    },
    {
        "code": "FLAT_MONOTONE_SUPPRESSION",
        "triggers": ["f0_semitone_std", "loudness_mean"],
        "desc": "Compressed pitch variability (F0_std Z <= -1.5) and restricted vocal intensity",
        "advocate": "indicates deliberate vocal control, emotional dampening, or guarded delivery",
        "skeptic": "characteristic of analytical lecturing, fatigue, or calm low-arousal baseline speech",
    },
    {
        "code": "HIGH_PITCH_AND_LATENCY_HESITATION",
        "triggers": ["f0_semitone_mean", "pause_ratio", "disfluency_count", "response_latency_ms"],
        "desc": "Elevated mean F0 pitch (Z >= 1.6) preceded by >900ms response latency and filler disfluencies",
        "advocate": "signals high autonomic arousal coupled with real-time speech planning conflict",
        "skeptic": "in bilingual/L2 speakers, mid-clause pauses and fillers often reflect lexical translation retrieval rather than emotional stress",
    },
    {
        "code": "RESONANT_FLUENT_PROSODY",
        "triggers": ["hnr_db", "speech_rate_wpm"],
        "desc": "Stable HNR, baseline jitter, and fluent speech rate without abnormal pause clusters",
        "advocate": "confirms high vocal comfort, domain fluency, and absence of laryngeal tension",
        "skeptic": "rehearsed or scripted answers can maintain fluent prosody despite internal reservation",
    },
]

RPPG_STATES = [
    {
        "code": "SYMPATHETIC_HR_ACCELERATION",
        "triggers": ["hr_bpm", "hr_bpm_pos"],
        "desc": "Phasic heart-rate acceleration (Z_floor >= 1.8 and Delta_Z_phasic >= 1.4, SQI >= 0.45)",
        "advocate": "provides objective cardiovascular evidence of sympathetic beta-adrenergic mobilization",
        "skeptic": "active rapid speaking increases intrathoracic pressure and metabolic demand (RSA effect)",
    },
    {
        "code": "AUTONOMIC_CARRYOVER_RUMINATION",
        "triggers": ["hr_bpm", "hr_bpm_pos"],
        "desc": "Sustained elevated HR (Z_floor >= 1.6) persisting >30s after a topic change (low phasic delta)",
        "advocate": "demonstrates delayed vagal rebound and ongoing cognitive rumination from a prior trigger",
        "skeptic": "ambient room temperature or cumulative speaking fatigue can gradually elevate tonic HR",
    },
    {
        "code": "HOMEOSTATIC_VAGAL_STABILITY",
        "triggers": ["hr_bpm", "hr_bpm_pos", "rppg_sqi"],
        "desc": "Heart rate stable within personal resting floor (|Z_floor| < 1.2, SQI >= 0.45)",
        "advocate": "indicates absence of acute sympathetic threat appraisal; non-verbal shifts are likely cognitive rather than autonomic",
        "skeptic": "brief cognitive events (<2s) may resolve before the 3-to-5 second hemodynamic rPPG peak appears",
    },
]

KINESIC_STATES = [
    {
        "code": "PACIFYING_NECK_FACE_ADAPTOR",
        "triggers": ["self_touch_ratio"],
        "desc": "Sustained hand-to-neck, suprasternal notch, or chin/lip pacifying touch (>300ms)",
        "advocate": "activates trigeminal/vagal self-soothing pathways to down-regulate acute anxiety or discomfort",
        "skeptic": "must be verified via video gate to rule out eyeglass adjustment, hair fixing, or skin itching",
    },
    {
        "code": "ILLUSTRATOR_FREEZE_OR_FIDGET",
        "triggers": ["hand_kinetic_energy"],
        "desc": "Sudden cessation of illustrative hand gestures (freeze) OR repetitive small-radius hand fidgeting",
        "advocate": "reflects attentional resource reallocation under high cognitive load or psychomotor agitation",
        "skeptic": "neurodivergent or high-kinesic individuals use continuous hand motion for focus regulation",
    },
    {
        "code": "ENGAGED_ILLUSTRATIVE_POSTURE",
        "triggers": ["hand_kinetic_energy", "postural_lean_z"],
        "desc": "Speech-synchronized illustrative gestures with stable or slightly forward postural lean",
        "advocate": "supports communicative openness, high topic engagement, and motor-verbal synchrony",
        "skeptic": "culturally expressive speakers maintain illustrative gesturing even during challenging turns",
    },
]

LINGUISTIC_CONTEXTS = [
    {
        "code": "POSITIVE_VERBAL_CLAIM_WITH_HEDGING_OR_PRONOUN_DROP",
        "triggers": ["text_valence_score", "first_person_pronoun_ratio"],
        "desc": "Positive surface claim ('Everything is fine / We handled it') accompanied by 'I' pronoun drop or epistemic hedging",
        "congruency_note": "High risk of Cross-Modal Incongruence (CMCI < 0.50) when paired with tense visual/vocal/autonomic markers.",
    },
    {
        "code": "COMPLEX_TECHNICAL_OR_L2_EXPLANATION",
        "triggers": ["pause_ratio", "disfluency_count"],
        "desc": "Detailed analytical explanation, episodic recall, or second-language (L2/Hinglish) lexical formulation",
        "congruency_note": "High cognitive effort indicators (AU04, gaze shift, pauses) are expected and concordant with complex problem-solving.",
    },
    {
        "code": "DIRECT_CONFRONTATION_OR_BOUNDARY_STATEMENT",
        "triggers": ["text_valence_score"],
        "desc": "Explicit expression of disagreement, concern, or resource constraints (negative semantic valence)",
        "congruency_note": "Elevated arousal in voice/face/HR is congruent with the direct verbal message (authentic distress/assertion).",
    },
]


def generate_tier2_combinatorial_chunks() -> list[dict[str, Any]]:
    """Generates 6 x 4 x 3 x 3 x 3 = 648 structured multimodal decision matrix chunks (PDF Page 55)."""
    matrix_chunks: list[dict[str, Any]] = []
    combinations = itertools.product(
        FACS_STATES, VOICE_STATES, RPPG_STATES, KINESIC_STATES, LINGUISTIC_CONTEXTS
    )
    for idx, (facs, voice, rppg, kin, ling) in enumerate(combinations, 1):
        combo_id = f"doc_matrix_perm_{idx:04d}"
        title = f"Multimodal Pattern #{idx:04d}: {facs['code']} + {voice['code']} + {rppg['code']} + {kin['code']}"
        triggers = sorted(set(facs["triggers"] + voice["triggers"] + rppg["triggers"] + kin["triggers"] + ling["triggers"]))
        section_header = f"{facs['code']} | {voice['code']} | {rppg['code']}"
        content = (
            f"MULTIMODAL COMBINATION SIGNATURE ({combo_id}):\n"
            f"- Visual FACS & Ocular: {facs['desc']}\n"
            f"- Acoustic Voice (openSMILE): {voice['desc']}\n"
            f"- Cardiovascular rPPG: {rppg['desc']}\n"
            f"- Kinesics & Posture: {kin['desc']}\n"
            f"- Linguistic & Discourse Context: {ling['desc']}\n\n"
            f"SIGNAL ADVOCATE INTERPRETATION:\n"
            f"Convergence across modalities: Visually, the subject {facs['advocate']}; acoustically, the voice {voice['advocate']}; "
            f"physiologically, the rPPG trend {rppg['advocate']}; kinesically, the body {kin['advocate']}. {ling['congruency_note']}\n"
            f"PRAGMATIC SKEPTIC COUNTER-HYPOTHESIS & CONFOUNDERS:\n"
            f"Before flagging psychological discomfort, evaluate benign alternatives: visually, {facs['skeptic']}; "
            f"acoustically, {voice['skeptic']}; physiologically, {rppg['skeptic']}; kinesically, {kin['skeptic']}."
        )
        matrix_chunks.append({
            "chunk_id": f"chunk_{combo_id}",
            "doc_id": combo_id,
            "document_id": "doc_tier2_combinatorial_matrix_v2",
            "parent_id": f"parent_{combo_id}",
            "parent_text": content,
            "source_title": title,
            "authors": ["OmniPulse Combinatorial Synthesis Engine"],
            "publication_year": 2026,
            "modality_tags": ["FUSION", "FACS", "AUDIO", "RPPG", "KINESICS", "LINGUISTIC"],
            "signal_triggers": triggers,
            "evidence_tier": "TIER_2_COMBINATORIAL_MATRIX",
            "section": section_header,
            "section_header": section_header,
            "content_text": content,
        })
    return matrix_chunks


def generate_tier2_combinatorial_matrix() -> tuple[str, int]:
    """Compatibility helper returning markdown text + count (648)."""
    chunks = generate_tier2_combinatorial_chunks()
    md_sections = [f"## {c['source_title']}\n\n{c['content_text']}" for c in chunks]
    return "# Tier 2 Combinatorial Multimodal Decision Matrix (648 Permutations)\n\n" + "\n\n".join(md_sections), len(chunks)


# ============================================================================
# TIER 3: EUROPE PMC AUTO-TAGGING SCIENTIFIC SCRAPER (PDF Pages 56-57)
# ============================================================================
PUBMED_QUERIES = [
    ("openSMILE eGeMAPS vocal jitter shimmer cognitive load stress", ["AUDIO", "ACOUSTIC"]),
    ("Facial Action Coding System AU24 AU04 response inhibition emotion regulation", ["FACS", "VISUAL"]),
    ("remote photoplethysmography POS CHROM heart rate variability mental stress", ["RPPG", "PHYSIOLOGY"]),
    ("spontaneous eye blink rate aspect ratio cognitive load working memory", ["OCULAR", "FACS", "VISUAL"]),
    ("nonverbal self-adaptors pacifying touch anxiety hand movements", ["KINESICS", "VISUAL"]),
    ("Duchenne smile AU06 AU12 social masking facial asymmetry", ["FACS", "VISUAL", "INCONGRUENCE"]),
    ("gaze aversion cognitive load floor gaze visual avoidance", ["GAZE", "OCULAR", "VISUAL"]),
    ("psycholinguistic pronoun drop LIWC psychological distancing", ["LINGUISTIC", "DISCOURSE"]),
    ("vagal tone heart rate recovery emotional carryover rumination", ["RPPG", "AUTONOMIC", "RECOVERY"]),
    ("bilingual second language L2 disfluency pause ratio cognitive load", ["CULTURAL", "L2_LANGUAGE", "AUDIO"]),
    ("multimodal speech emotion recognition acoustic linguistic incongruence sarcasm", ["FUSION", "INCONGRUENCE", "AUDIO"]),
    ("respiratory sinus arrhythmia speech breathing heart rate confounder", ["RPPG", "AUDIO"]),
    ("vocal fundamental frequency F0 pitch variability autonomic arousal", ["AUDIO", "PROSODY"]),
    ("harmonic to noise ratio HNR vocal fold tension psychological stress", ["AUDIO", "ACOUSTIC"]),
    ("postural lean head nod backchannel conversational engagement", ["KINESICS", "POSTURE"]),
    ("interpersonal vocal entrainment acoustic convergence rapport conversation", ["AUDIO", "DISCOURSE"]),
    ("facial micro-expression onset offset duration deliberate vs spontaneous", ["FACS", "TEMPORAL"]),
    ("neurodivergent ADHD autism gaze aversion fidgeting cognitive regulation", ["CULTURAL", "NEURODIVERGENCE", "KINESICS"]),
]

HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_html_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(HTML_TAG_RE.sub(" ", value or ""))).strip()


def extract_signal_triggers(text: str) -> list[str]:
    return knowledge_retrieval._extract_signal_triggers(text)


def auto_detect_modality_tags(text: str, base_tags: list[str] | None = None) -> list[str]:
    t_lower = text.lower()
    tags = set(base_tags or [])
    if any(k in t_lower for k in ("facial", "action unit", "smile", "brow", "au0", "au1", "au2")):
        tags.update(["FACS", "VISUAL"])
    if any(k in t_lower for k in ("vocal", "acoustic", "pitch", "f0", "jitter", "shimmer", "prosod", "speech rate")):
        tags.add("AUDIO")
    if any(k in t_lower for k in ("heart rate", "rppg", "cardiovascular", "autonomic", "vagal", "hrv")):
        tags.add("RPPG")
    if any(k in t_lower for k in ("gesture", "posture", "hand", "self-touch", "adaptor", "head nod")):
        tags.add("KINESICS")
    if any(k in t_lower for k in ("pronoun", "linguistic", "lexical", "discourse", "liwc")):
        tags.add("LINGUISTIC")
    return sorted(tags or {"GENERAL"})


def fetch_tier3_empirical_papers(max_per_query: int = 45) -> list[dict[str, Any]]:
    seen_pmc_ids: set[str] = set()
    empirical_chunks: list[dict[str, Any]] = []

    for idx, (query, base_tags) in enumerate(PUBMED_QUERIES, 1):
        LOGGER.info("[Tier 3 - %d/%d] Scraping EuropePMC for: %r", idx, len(PUBMED_QUERIES), query)
        encoded_q = urllib.parse.quote(f"{query} AND (OPEN_ACCESS:y)")
        url = (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            f"?query={encoded_q}&format=json&pageSize={max_per_query}&resultType=core"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OmniPulse-MBS-RAG-Ingestor/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for item in data.get("resultList", {}).get("result", []):
                pmc_id = item.get("pmcid") or item.get("id")
                if not pmc_id or pmc_id in seen_pmc_ids:
                    continue
                seen_pmc_ids.add(str(pmc_id))

                raw_abstract = item.get("abstractText", "")
                clean_abstract = clean_html_text(raw_abstract)
                if len(clean_abstract) < 220:
                    continue

                title = clean_html_text(item.get("title", "Peer-Reviewed Study"))
                authors = [clean_html_text(a.get("fullName", "")) for a in (item.get("authorList") or {}).get("author", [])[:4] if a.get("fullName")]
                pub_year = int(item.get("pubYear", 2024)) if str(item.get("pubYear", "")).isdigit() else 2024
                journal = clean_html_text(((item.get("journalInfo") or {}).get("journal") or {}).get("title", "Peer-Reviewed Journal"))

                sentences = [s.strip() for s in clean_abstract.split(". ") if len(s.strip()) > 25]
                mid = max(1, len(sentences) // 2)
                sub_parts = [
                    ("Empirical Context & Mechanism", ". ".join(sentences[:mid]) + "."),
                    ("Quantitative Findings & Confounders", ". ".join(sentences[mid:]) + "."),
                ]

                for part_label, part_text in sub_parts:
                    if len(part_text) < 100:
                        continue
                    full_text = f"STUDY: {title} ({journal}, {pub_year})\nSECTION: {part_label}\n{part_text}"
                    doc_id = f"pmc_{pmc_id}_{part_label[:4].lower()}"
                    empirical_chunks.append({
                        "chunk_id": f"chunk_{doc_id}",
                        "doc_id": doc_id,
                        "document_id": f"doc_pmc_{pmc_id}",
                        "parent_id": f"parent_{doc_id}",
                        "parent_text": full_text,
                        "source_title": f"{title} ({journal})",
                        "authors": authors or ["Published Study"],
                        "publication_year": pub_year,
                        "modality_tags": auto_detect_modality_tags(full_text, base_tags),
                        "signal_triggers": extract_signal_triggers(full_text),
                        "evidence_tier": "TIER_3_EMPIRICAL_STUDY",
                        "section": f"{title[:65]}... -> {part_label}",
                        "section_header": f"{title[:65]}... -> {part_label}",
                        "content_text": full_text,
                    })
            time.sleep(0.25)
        except Exception as exc:
            LOGGER.warning("Query %r skipped due to network error: %s", query, exc)

    return empirical_chunks


def load_tier1_markdown_seeds(seed_dir: Path) -> list[dict[str, Any]]:
    seed_paths = ensure_seed_manuals_exist(seed_dir)
    chunks: list[dict[str, Any]] = []
    for md_path in seed_paths:
        raw = md_path.read_text(encoding="utf-8")
        meta, body = parse_yaml_frontmatter(raw)
        doc_id = str(meta.get("doc_id") or md_path.stem)
        source_title = str(meta.get("source_title") or "OmniPulse Core Manual")
        authors = meta.get("authors") or ["OmniPulse Scientific Board"]
        pub_year = int(meta["publication_year"]) if str(meta.get("publication_year", "")).isdigit() else 2026
        mod_tags = [str(t).upper() for t in (meta.get("modality_tags") or ["GENERAL"])]

        for section in body.split("\n## "):
            section = section.strip()
            if not section or section.startswith("# Section"):
                continue
            lines = section.splitlines()
            header = lines[0].replace("#", "").strip()
            content = "\n".join(lines[1:]).strip()
            if len(content) < 40:
                continue
            full_content = f"[CANONICAL MANUAL: {header}]\n{content}"
            c_hash = hashlib.md5(f"{doc_id}:{header}".encode()).hexdigest()[:12]
            chunks.append({
                "chunk_id": f"chunk_{c_hash}",
                "doc_id": doc_id,
                "document_id": doc_id,
                "parent_id": f"parent_{c_hash}",
                "parent_text": full_content,
                "source_title": source_title,
                "authors": authors,
                "publication_year": pub_year,
                "modality_tags": mod_tags,
                "signal_triggers": extract_signal_triggers(full_content),
                "evidence_tier": "TIER_1_CANONICAL_RULE",
                "section": header,
                "section_header": header,
                "content_text": full_content,
            })
    return chunks


def build_and_ingest_complete_kb(
    *,
    seed_dir: Path = ROOT / "knowledge_base_seed",
    max_per_query: int = 45,
    include_europepmc: bool = True,
    dry_run: bool = False,
    refresh_existing: bool = False,
    backup_json_path: Path | None = None,
) -> dict[str, int]:
    if not dry_run:
        repository.init_db()

    tier1_chunks = load_tier1_markdown_seeds(seed_dir)
    tier2_chunks = generate_tier2_combinatorial_chunks()
    tier3_chunks = (fetch_tier3_empirical_papers(max_per_query=max_per_query) + fetch_arxiv_affective_computing_papers(max_per_query=min(35, max_per_query))) if include_europepmc else []
    if not tier3_chunks:
        # Include built-in empirical studies when offline so Tier 3 is never empty
        tier3_chunks = [
            c for c in knowledge_retrieval._BUILTIN_STRATIFIED_CHUNKS if c["evidence_tier"] == "TIER_3_EMPIRICAL_STUDY"
        ]

    all_chunks = tier1_chunks + tier2_chunks + tier3_chunks
    totals = {
        "documents": 0,
        "chunks": len(all_chunks),
        "skipped": 0,
        "tier1_chunks": len(tier1_chunks),
        "tier2_chunks": len(tier2_chunks),
        "tier3_chunks": len(tier3_chunks),
    }

    if backup_json_path and not dry_run:
        backup_json_path.parent.mkdir(parents=True, exist_ok=True)
        backup_json_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    if dry_run:
        return totals

    # Group chunks by document_id for SQLite + Qdrant ingestion
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ch in all_chunks:
        doc_id = str(ch.get("document_id") or ch.get("doc_id"))
        grouped.setdefault(doc_id, []).append(ch)

    for doc_id, doc_chunks in grouped.items():
        first = doc_chunks[0]
        raw_concat = "\n\n".join(c["content_text"] for c in doc_chunks).encode("utf-8")
        sha256_hash = hashlib.sha256(raw_concat).hexdigest()
        doc_record = {
            "id": doc_id,
            "file_name": f"{doc_id}.md",
            "source_title": first.get("source_title", doc_id),
            "authors": first.get("authors", ["OmniPulse Scientific Board"]),
            "publication_year": first.get("publication_year", 2026),
            "modality_tags": sorted(set(tag for c in doc_chunks for tag in (c.get("modality_tags") or ["GENERAL"]))),
            "evidence_tier": first.get("evidence_tier", "TIER_1_CANONICAL_RULE"),
            "status": "INDEXED",
            "sha256_hash": sha256_hash,
            "rejection_reason": None,
        }
        existing = repository.get_document(doc_id)
        if not existing:
            try:
                repository.insert_document(doc_record, doc_chunks)
                totals["documents"] += 1
            except Exception:
                pass
        if knowledge_retrieval.configuration_ready() and (refresh_existing or not (existing and existing.get("dense_indexed"))):
            try:
                knowledge_retrieval.index_chunks(doc_record, doc_chunks)
                repository.mark_document_dense_indexed(doc_id)
            except Exception as exc:
                repository.mark_document_dense_error(doc_id, f"{type(exc).__name__}: {exc}")

    return totals



import xml.etree.ElementTree as ET


def fetch_pmc_fulltext_sections(pmc_id: str, title: str, journal: str, pub_year: int, base_tags: list[str]) -> list[dict[str, Any]]:
    """
    Fetches the full JATS XML of an open-access PMC article (e.g., PMC8941203)
    and extracts high-value paragraphs from <sec> sections (Results, Discussion, Findings)
    containing quantitative thresholds (r=, p<, Z-score, Hz, BPM, AU).
    """
    if not str(pmc_id).startswith("PMC"):
        return []

    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_id}/fullTextXML"
    chunks: list[dict[str, Any]] = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OmniPulse-MBS-FullText-Ingestor/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_bytes = resp.read()

        root = ET.fromstring(xml_bytes)
        body = root.find(".//body")
        if body is None:
            return []

        for sec in body.findall(".//sec"):
            sec_title_el = sec.find("title")
            sec_title = "".join(sec_title_el.itertext()).strip() if sec_title_el is not None else "Section"

            if not any(k in sec_title.lower() for k in ["result", "discuss", "finding", "analysis", "experiment", "feature", "correlat", "limitation"]):
                continue

            paragraphs = ["".join(p_el.itertext()).strip() for p_el in sec.findall("p")]
            combined_text = " ".join(p_txt for p_txt in paragraphs if len(p_txt) > 80)

            for idx in range(0, len(combined_text), 850):
                window_text = combined_text[idx : idx + 950].strip()
                if len(window_text) < 200:
                    continue

                full_chunk_text = f"FULL-TEXT STUDY ({pmc_id} - {journal}, {pub_year}): {title}\nSECTION: {sec_title}\n{window_text}"
                doc_id = f"pmc_ft_{pmc_id}_{len(chunks):02d}"
                chunks.append({
                    "chunk_id": f"chunk_{doc_id}",
                    "doc_id": doc_id,
                    "document_id": f"doc_pmc_ft_{pmc_id}",
                    "parent_id": f"parent_{doc_id}",
                    "parent_text": full_chunk_text,
                    "source_title": f"{title} ({journal})",
                    "authors": ["Peer-Reviewed Full-Text Corpus"],
                    "publication_year": pub_year,
                    "modality_tags": auto_detect_modality_tags(full_chunk_text, base_tags),
                    "signal_triggers": extract_signal_triggers(full_chunk_text),
                    "evidence_tier": "TIER_3_EMPIRICAL_STUDY",
                    "section": f"{sec_title[:50]} (Chunk {len(chunks)+1})",
                    "section_header": f"{sec_title[:50]} (Chunk {len(chunks)+1})",
                    "content_text": full_chunk_text,
                })
                if len(chunks) >= 8:
                    return chunks
    except Exception:
        pass
    return chunks


def fetch_arxiv_affective_computing_papers(max_per_query: int = 35) -> list[dict[str, Any]]:
    """
    Scrapes arXiv API for recent Multimodal Emotion Recognition, rPPG, and OpenSMILE papers.
    """
    arxiv_queries = [
        ("all:openSMILE+AND+all:emotion", ["AUDIO", "FUSION"]),
        ("all:rPPG+AND+all:heart+rate", ["RPPG", "PHYSIOLOGY"]),
        ("all:Facial+Action+Coding+System+AND+all:stress", ["FACS", "VISUAL"]),
        ("all:multimodal+incongruence+AND+all:speech", ["FUSION", "INCONGRUENCE", "AUDIO"]),
        ("all:eye+blink+AND+all:cognitive+load", ["OCULAR", "VISUAL"]),
    ]
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    arxiv_chunks: list[dict[str, Any]] = []

    for q_str, base_tags in arxiv_queries:
        url = f"http://export.arxiv.org/api/query?search_query={q_str}&start=0&max_results={max_per_query}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                root = ET.fromstring(resp.read())

            for entry in root.findall("atom:entry", ns):
                arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1].replace(".", "_")
                title = " ".join(entry.find("atom:title", ns).text.split())
                summary = " ".join(entry.find("atom:summary", ns).text.split())
                pub_year = int(entry.find("atom:published", ns).text[:4])
                authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)[:4]]

                if len(summary) < 180:
                    continue

                full_text = f"ARXIV PREPRINT ({pub_year}): {title}\nABSTRACT & METHODOLOGY:\n{summary}"
                doc_id = f"arxiv_{arxiv_id}"
                arxiv_chunks.append({
                    "chunk_id": f"chunk_{doc_id}",
                    "doc_id": doc_id,
                    "document_id": f"doc_{doc_id}",
                    "parent_id": f"parent_{doc_id}",
                    "parent_text": full_text,
                    "source_title": f"{title} (arXiv)",
                    "authors": authors,
                    "publication_year": pub_year,
                    "modality_tags": auto_detect_modality_tags(full_text, base_tags),
                    "signal_triggers": extract_signal_triggers(full_text),
                    "evidence_tier": "TIER_3_EMPIRICAL_STUDY",
                    "section": f"{title[:65]}...",
                    "section_header": f"{title[:65]}...",
                    "content_text": full_text,
                })
        except Exception as e:
            print(f"  [ArXiv Warning] {e}")

    print(f"[Tier 3 - ArXiv] Loaded {len(arxiv_chunks)} multimodal AI & signal processing chunks.")
    return arxiv_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, default=ROOT / "knowledge_base_seed")
    parser.add_argument("--max-per-query", type=int, default=45)
    parser.add_argument("--skip-europepmc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--backup-json", type=Path, default=BACKEND_DIR / "data" / "massive_kb_backup.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        build_and_ingest_complete_kb(
            seed_dir=args.seed_dir.resolve(),
            max_per_query=args.max_per_query,
            include_europepmc=not args.skip_europepmc,
            dry_run=args.dry_run,
            refresh_existing=args.refresh_existing,
            backup_json_path=args.backup_json.resolve(),
        )
    finally:
        knowledge_retrieval.close()


if __name__ == "__main__":
    main()
