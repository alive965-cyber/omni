---
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
