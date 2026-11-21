---
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
