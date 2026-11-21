---
doc_id: "doc_rppg_autonomic_2026_v1"
source_title: "Remote Photoplethysmography (rPPG POS/CHROM) and Autonomic Cardiovascular Psychophysiology Manual"
authors: ["Wang, W.", "de Haan, G.", "Berntson, G.G.", "Cacioppo, J.T.", "Porges, S.W."]
publication_year: 2025
modality_tags: ["RPPG", "PHYSIOLOGY", "HEART_RATE", "AUTONOMIC"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Autonomic Nervous System (ANS) Interpretation of rPPG Heart Rate

## Sympathetic Acceleration (Phasic Tachycardia: `hr_bpm_z >= +1.8`)
- **Physiological Mechanism:** Withdrawal of vagal parasympathetic brake (nucleus ambiguus) followed by sympathetic beta-adrenergic stimulation of the sinoatrial (SA) node. Has a hemodynamic latency of **2.0 to 5.0 seconds** after a psychological trigger.
- **Signal Triggers:** `hr_bpm_pos`, `hr_bpm_z >= 1.8`, `rppg_sqi >= 0.45`
- **Behavioral Correlates:**
  1. **Defensive Arousal / Acute Stress:** A sustained rise of $\ge +8\text{ BPM}$ (or $Z_{\text{HR}} \ge +1.8$) over a 6–15 second window accompanied by vocal jitter elevation ($Z_{\text{jitter}} \ge +1.5$), AU24 lip compression, or self-touch adaptors indicates sympathetic mobilization in response to perceived challenge, social evaluation, or emotional strain.
  2. **High Cognitive Effort:** Complex mental arithmetic, difficult memory retrieval, or defending a controversial position elevates HR by $5\text{ to }12\text{ BPM}$ due to central command metabolic demand.
  3. **Positive Excitement:** High HR paired with Duchenne smiling (AU06+AU12), high vocal loudness, and positive verbal valence reflects enthusiastic arousal rather than distress.

## Orienting Response (Transient Parasympathetic Deceleration: `hr_bpm_z <= -1.5`)
- **Physiological Mechanism:** Sensory intake reflex mediated by rapid vagal efferent discharge to the SA node upon encountering a novel, unexpected, or highly salient stimulus.
- **Signal Triggers:** `hr_bpm_z <= -1.5`, `blink_rate_z < -1.0`
- **Behavioral Correlates:**
  1. **Attentional Orienting / Freeze:** A brief drop in heart rate ($3\text{ to }6\text{ BPM}$ below baseline) paired with reduced blinking and postural stillness while listening to a critical question reflects intense sensory intake and focused appraisal before the sympathetic response begins.

---

# Section 2: Mandatory rPPG Signal Quality & Confounder Constraints

## Signal Quality Index (`rppg_sqi`) & Algorithm Agreement (`POS` vs `CHROM`)
- **Signal Triggers:** `rppg_sqi`, `rppg_status`, `hr_bpm_pos`, `hr_bpm_chrom`
- **Strict Interpretation Rules:**
  1. **High Confidence (`rppg_sqi >= 0.55` and `|POS - CHROM| <= 5.0 BPM`):** Cardiovascular trend can be weighted fully alongside acoustic and visual signals.
  2. **Moderate Confidence (`0.38 <= rppg_sqi < 0.55`):** Report HR trend as an approximate indicator only; require corroboration from openSMILE acoustic or MediaPipe visual channels before flagging an autonomic event.
  3. **Unreliable / Masked (`rppg_sqi < 0.38` or `rppg_status != "VALID"`):** Completely exclude heart rate from behavioral reasoning for that timestamp window. Explicitly state that optical pulse tracking was paused due to head motion or illumination variance.
- **Physiological Confounders:** Continuous rapid speaking alters intrathoracic pressure and induces **Respiratory Sinus Arrhythmia (RSA)**, naturally elevating HR by $3\text{ to }6\text{ BPM}$ compared to silent listening. Deep sighs, coughing, or laughing also produce transient HR spikes unrelated to psychological stress.