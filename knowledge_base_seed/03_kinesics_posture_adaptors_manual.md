---
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
