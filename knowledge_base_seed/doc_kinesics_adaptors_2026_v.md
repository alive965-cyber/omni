---
doc_id: "doc_kinesics_adaptors_2026_v1"
source_title: "Kinesic Self-Adaptors, Hand Motion Energy, and Postural Telemetry Guide"
authors: ["Ekman, P.", "Friesen, W.V.", "Navarro, J.", "Freedman, N.", "Harrigan, J.A."]
publication_year: 2024
modality_tags: ["KINESICS", "VISUAL", "POSTURE", "GESTURES"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Self-Adaptors (Pacifying Touch Behaviors)

## Face, Neck, and Suprasternale Touch (`self_touch_ratio`)
- **Neurobiological Mechanism:** Tactile stimulation of the face (trigeminal nerve, CN V) and anterior neck/carotid sinus region activates parasympathetic vagal pathways that help down-regulate acute sympathetic arousal and cognitive discomfort.
- **Signal Triggers:** `self_touch_ratio > 0.25`, `gesture_type == "SELF_TOUCH"`
- **Behavioral Correlates:**
  1. **Emotional & Cognitive Pacifying:** A sudden onset of hand-to-neck, hand-to-chin, or hand-to-nose contact during or immediately after a specific topic or question correlates strongly with self-soothing under heightened anxiety, uncertainty, or cognitive load.
  2. **Mouth/Lip Covering Adaptor:** Bringing fingers to the lips or chin while listening or immediately after speaking frequently co-occurs with response inhibition (similar to AU24).
- **Mandatory Confounders:** Itching, adjusting glasses or headset microphone, resting the chin on a hand during prolonged listening (postural support), or habitual baseline grooming. Always verify whether `self_touch_ratio` was low during the 90-second baseline before flagging as a stress adaptor.

---

# Section 2: Hand Kinetic Energy — Illustrators vs. Fidgeting

## Hand Motion Disambiguation (`hand_kinetic_energy`, `gesture_type`)
- **Signal Triggers:** `hand_kinetic_energy`, `hand_ke_z > 1.5`, `gesture_type`
- **Behavioral Correlates:**
  1. **Illustrative Gestures (`gesture_type == "ILLUSTRATIVE"`):** High hand kinetic energy synchronized with active speech (`speech_rate_wpm > 100`) and open spatial trajectories reflects high speaker engagement, fluency, and expressive confidence.
  2. **Illustrator Drop / Freeze ($Z_{\text{hand\_ke}} \le -1.5$ during speech):** When an expressive speaker suddenly stops moving their hands while answering a specific question, this reflects **cognitive overload** (attentional resources diverted from motor gesturing to verbal formulation) or guardedness.
  3. **Repetitive Hand Fidgeting (`gesture_type == "FIDGET_ADAPTOR"`):** High-frequency, small-radius hand-to-hand rubbing, finger tapping, or wrist restlessness during silence or low speech fluency reflects psychomotor agitation or nervous energy.

---

# Section 3: Postural Lean & Trunk Dynamics

## Anterior-Posterior Lean (`postural_lean_z`)
- **Signal Triggers:** `postural_lean_z`
- **Behavioral Correlates:**
  1. **Forward Lean (Approach Orientation):** Leaning toward the camera/interlocutor accompanied by direct gaze and elevated $F_0$ variability indicates active interest, engagement, or assertive emphasis.
  2. **Backward Withdrawal / Distancing:** A rapid backward shift away from the camera immediately following a question or statement, especially when paired with crossed arms or lip compression (AU24), correlates with psychological distancing, defensiveness, or withdrawal.