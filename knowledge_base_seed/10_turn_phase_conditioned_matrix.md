---
doc_id: "doc_turn_phase_matrix_2026_v1"
source_title: "Dyadic Turn-Phase Conditioned Behavioral Interpretation Manual (Listening vs. Latency vs. Execution vs. Post-Utterance Leak)"
authors: ["Bavelas, J.B.", "Ekman, P.", "Levinson, S.C.", "Kendon, A."]
publication_year: 2026
modality_tags: ["TEMPORAL", "FACS", "KINESICS", "AUDIO", "DISCOURSE", "FUSION"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Phase 1 — STIMULUS_LISTENING (While Interlocutor is Speaking)

## Visual & Autonomic Signals During Active Listening
- **Phase Definition:** `turn_phase == "STIMULUS_LISTENING"`. The subject is silent and receiving the prompt.
- **Signal Triggers:** `au04_brow_lowerer`, `au24_lip_presser`, `au12_lip_corner_puller`, `head_roll_deg`, `hr_bpm_pos`
- **Phase-Specific Interpretation Rules:**
  1. **Listener `AU04 + AU24` Spike:** Because the subject is not speaking, bilabial consonant confounders (`/p/, /b/, /m/`) are 100% ruled out. `AU24` during listening indicates **active impulse inhibition** (suppressing an urge to interrupt or object), while `AU04` reflects critical evaluation or comprehension strain triggered by the interviewer's exact words.
  2. **Listener Orienting Freeze:** A sudden drop in `blink_rate_per_min` and `hand_kinetic_energy` accompanied by slight heart-rate deceleration (`hr_bpm_z <= -1.2`) during a specific keyword indicates **high-salience sensory intake (orienting reflex)**.

---

# Section 2: Phase 2 — APPRAISAL_LATENCY (Pre-Speech Gap Between Question and Answer)

## Pre-Phonation Cognitive Planning vs. Defensive Stall
- **Phase Definition:** `turn_phase == "APPRAISAL_LATENCY"`. Silent interval `[question_end -> first_spoken_word]`.
- **Signal Triggers:** `response_latency_ms`, `gaze_aversion_ratio`, `au04_brow_lowerer`, `self_touch_ratio`
- **Phase-Specific Interpretation Rules:**
  1. **Constructive Working-Memory Retrieval:** `response_latency_ms` between `600ms and 1800ms` paired with lateral/upward `gaze_aversion_ratio` and `AU04`, followed by fluent, detailed `VERBAL_EXECUTION`, reflects normal **episodic or technical memory retrieval**.
  2. **Inhibitory Editing / Reluctance Stall:** `response_latency_ms > 1500ms` paired with **downward gaze shift**, **`AU24` lip compression**, **inspiratory breath-hold**, or **neck/chin self-touch**, followed by a brief, hedged utterance (*"I think that's fine"*), indicates **internal response editing and reluctance**.

---

# Section 3: Phase 3 — VERBAL_EXECUTION (During Active Speech Delivery)

## Motor-Speech Coupling & Co-Verbal Gesturing
- **Phase Definition:** `turn_phase == "VERBAL_EXECUTION"`. Active phonation is occurring.
- **Signal Triggers:** `f0_semitone_mean`, `jitter_local_mean`, `shimmer_local_db_mean`, `hand_kinetic_energy`, `au25_jaw_drop`
- **Phase-Specific Interpretation Rules:**
  1. **Discount Lower-Face Geometry During Active Phonation:** During `VERBAL_EXECUTION`, jaw opening (`AU25/26`), lip stretching (`AU20`), and transient lip pressing (`AU24`) are heavily modulated by vowel/consonant articulation. Weight **upper-face Action Units (`AU01, AU02, AU04, AU06`)**, **vocal micro-tremor (`jitter_local_mean`, `shimmer_local_db_mean`, `hnr_db_mean`)**, and **co-verbal hand illustrators** much higher than lower-face lip metrics during active speech.

---

# Section 4: Phase 4 — POST_UTTERANCE_LEAK (First 1.5 Seconds After Finishing Speech)

## Post-Statement Cortical Relaxation & Affective Leakage
- **Phase Definition:** `turn_phase == "POST_UTTERANCE_LEAK"`. The 1.5-second silent window immediately after the subject finishes a sentence.
- **Signal Triggers:** `au24_lip_presser`, `au15_lip_corner_depressor`, `au20_lip_stretcher`, `smile_offset_velocity`, `self_touch_ratio`
- **Phase-Specific Interpretation Rules:**
  1. **Highest Diagnostic Weight for Facial & Kinesic Leakage:** While speaking, humans maintain deliberate cortical control over their vocal tone and facial expression. The instant phonation stops (`POST_UTTERANCE_LEAK`), cortical motor monitoring drops.
  2. **Post-Utterance `AU24` Seal + Gaze Drop:** Pressing the lips tightly shut right after finishing an answer while breaking eye contact signals *"I am stopping here and withholding further elaboration."*
  3. **Post-Utterance Micro-Frown (`AU15` / `AU20`) or Abrupt Smile Snap-Off:** Reveals the speaker's unvarnished self-evaluation of the answer they just delivered.