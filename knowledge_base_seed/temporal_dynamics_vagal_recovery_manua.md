---
doc_id: "doc_temporal_vagal_recovery_2026_v1"
source_title: "Temporal Micro-Dynamics, Facial Onset/Offset Kinetics, and Autonomic Vagal Recovery Trajectories"
authors: ["Yan, W.J.", "Porges, S.W.", "Thayer, J.F.", "Cohn, J.F.", "Levenson, R.W."]
publication_year: 2025
modality_tags: ["TEMPORAL", "RPPG", "FACS", "AUTONOMIC", "RECOVERY"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Facial Expression Kinetics (Onset, Apex, Offset Timing)

## Ballistic vs. Regulated Expression Timing
- **Anatomical Mechanism:** Spontaneous subcortical emotional expressions (extrapyramidal motor pathway) exhibit smooth, symmetrical ballistic onset ($0.3\text{ to }0.7\text{ s}$) and gradual logarithmic offset. Deliberate cortical expressions (pyramidal motor pathway) show abrupt, stepped onset or sudden "snap-off" offset.
- **Signal Triggers:** Window-to-window first derivative $\Delta \text{AU12} / \Delta t$, $\Delta \text{AU04} / \Delta t$.
- **Behavioral Correlates:**
  1. **Abrupt Smile Drop ("Snap-Off Offset"):** When a smile ($\text{AU12} \gt 0.50$) drops to $< 0.10$ within $< 250\text{ ms}$ as soon as the speaker finishes talking or looks away, it indicates a **deliberately posed social mask** rather than lingering positive affect.
  2. **Brief Leakage (< 500 ms):** A rapid transient spike in **AU04 (Brow Lowerer)**, **AU15 (Lip Corner Depressor)**, or **AU20 (Lip Stretcher)** lasting $150\text{ to }500\text{ ms}$ before being overridden by a neutral or smiling expression (`AU12`) indicates **rapid regulatory suppression of an initial negative appraisal**.

---

# Section 2: Autonomic Recovery & Emotional Carryover Across 10 Minutes

## Vagal Rebound vs. Sustained Autonomic Carryover (Rumination)
- **Physiological Mechanism:** After a stressful or cognitively demanding topic ends, a healthy parasympathetic vagal brake returns heart rate (`hr_bpm_pos`) and vocal jitter toward baseline within **15 to 40 seconds**.
- **Signal Triggers:** Post-event trajectory over windows $t+1$ through $t+20$ ($40\text{ seconds}$ after topic shift).
- **Behavioral Correlates:**
  1. **Fast Vagal Recovery (High Autonomic Resilience):** HR drops back within $\pm 1.0\text{ MAD}$ of baseline and vocal HNR normalizes within $20\text{–}30\text{ seconds}$ of moving to a neutral topic. Indicates the speaker compartmentalized the challenge effectively.
  2. **Autonomic Carryover / Rumination Effect:** If the conversation moves to a completely benign topic at `04:30`, yet `hr_bpm_z > 1.8`, `jitter_z > 1.5`, and `blink_rate_z > 1.5` remain elevated until `06:00`, the speaker is experiencing **cognitive rumination and unresolved autonomic distress** carried over from the `04:00` trigger topic.
  3. **Anticipatory Relief Effect:** A sudden drop in heart rate ($Z_{\text{HR}}$ falling by $> 1.0$), accompanied by a deep exhalation/sigh (`loudness` low-frequency breath burst), shoulder drop, and genuine Duchenne relaxation (`AU06+AU12`) right as a sensitive topic concludes, signals **psychological relief** that scrutiny on that topic has ended.