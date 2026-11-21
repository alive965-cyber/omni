---
doc_id: "doc_respiration_oculomotor_2026_v1"
source_title: "Thoracic Respiration Kinematics, Post-Stress Sigh Reflex, and Oculomotor Screen-Reading Forensics"
authors: ["Vlemincx, E.", "Grassmann, M.", "Rayner, K.", "Holmqvist, K."]
publication_year: 2026
modality_tags: ["PHYSIOLOGY", "RPPG", "OCULAR", "GAZE", "KINESICS", "FORENSICS"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Optical Thoracic & Shoulder Respiration Telemetry

## Breath-Holding (Apnea Freeze) & Rapid Shallow Thoracic Breathing
- **Physiological Mechanism:** Optical tracking of clavicular and acromial (shoulder `#11, #12`) vertical oscillation ($0.12\text{–}0.45\text{ Hz}$) captures respiratory rate without skin-color dependency.
- **Signal Triggers:** `respiration_rate_breaths_per_min`, `postural_lean_z`, `hr_bpm_pos`
- **Behavioral Correlates:**
  1. **Preparatory Breath-Hold (Inspiratory Apnea):** A temporary cessation of shoulder respiratory oscillation for $3\text{ to }6\text{ seconds}$ during `STIMULUS_LISTENING` or `APPRAISAL_LATENCY` reflects an **autonomic freeze / attentional lock** under acute scrutiny.
  2. **Tachypnea (Elevated Breathing Rate $> 20\text{ breaths/min}$):** Shallow, rapid upper-chest/shoulder breathing during low physical exertion correlates with sympathetic arousal and performance anxiety.

## The Psychophysiological Sigh Reflex (`detected_sigh_timestamps_sec`)
- **Physiological Mechanism:** A deep biphasic inspiration or high-amplitude exhalation ($> 2.2\times$ tidal volume prominence) mediated by the pre-Bötzinger complex in the medulla, acting as a mechanical and vagal reset of alveolar and autonomic tension.
- **Behavioral Correlates:**
  1. **Cognitive Load & Frustration Reset:** A prominent shoulder-drop sigh immediately before or after a complex topic signals **mental fatigue, frustration, or emotional regulation effort**.
  2. **Post-Topic Relief Sigh:** A deep exhalation sigh occurring right as the interviewer transitions away from a sensitive question indicates **anticipatory relief** and parasympathetic rebound.

---

# Section 2: Oculomotor Forensics — Spontaneous Thinking vs. Screen Reading

## Sawtooth Horizontal Raster Saccades (`STRUCTURED_HORIZONTAL_SCREEN_READING`)
- **Oculomotor Mechanism:** Reading text on a monitor produces a highly stereotyped oculomotor signature (Rayner, 1998): progressive rightward fixations ($200\text{–}250\text{ ms}$) separated by small rightward saccades ($1^\circ\text{–}3^\circ$), terminated by a rapid, large-amplitude leftward **return sweep** ($\Delta G_x \ll 0$) with minimal vertical dispersion ($\sigma(G_y) \ll \sigma(G_x)$).
- **Signal Triggers:** `saccade_velocity_deg_s`, `gaze_aversion_ratio`, `f0_semitone_std`, `blink_rate_per_min`
- **Forensic Disambiguation Rule:**
  1. **Spontaneous Cognitive Gaze Aversion:** Gaze shifts toward an unpopulated region of space (ceiling, floor, or side wall) and holds a relatively stable fixation or slow drift while the speaker accesses episodic/working memory. Pitch variability (`f0_semitone_std`) remains natural.
  2. **External Script / AI Teleprompter Reading:** Repeated horizontal sawtooth saccades + return sweeps during `VERBAL_EXECUTION`, accompanied by **suppressed blink rate** (`blink_rate_z <= -1.2`), **flattened prosodic pitch contour** (`f0_semitone_std_z <= -1.2`), and absence of natural semantic pauses.