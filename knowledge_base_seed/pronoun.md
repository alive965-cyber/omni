---
doc_id: "doc_psycholinguistics_liwc_2026_v1"
source_title: "Psycholinguistic Discourse Markers, Pronoun Distancing, and Verbal Defense Mechanisms (LIWC Framework)"
authors: ["Pennebaker, J.W.", "Newman, M.L.", "Tausczik, Y.R.", "Hancock, J.T."]
publication_year: 2025
modality_tags: ["LINGUISTIC", "NLP", "DISCOURSE", "INCONGRUENCE"]
evidence_tier: "PEER_REVIEWED_GOLD"
---

# Section 1: Psychological Distancing via Pronoun Shifts

## First-Person Singular Drop ("I / Me / My" Withdrawal)
- **Psycholinguistic Mechanism:** Self-reference words (*"I", "me", "my"*) reflect psychological ownership and personal accountability. When a speaker feels uncomfortable, defensive, or detached from an event or outcome, unconscious linguistic distancing occurs.
- **Signal Triggers:** `first_person_pronoun_ratio` drops $> 50\%$ below speaker's 90-second baseline; shift to passive voice (*"The deadline was missed"* vs. *"I missed the deadline"*) or collective generalization (*"We/You/One"*).
- **Multimodal Fusion Rule:**
  1. When a sudden drop in `"I"` pronouns co-occurs with **AU24 (Lip Presser)**, **gaze aversion**, or **elevated vocal jitter ($Z \ge 1.5$)**, it indicates **psychological distancing from responsibility or discomfort with personal attribution**.
  2. Do not flag if the question explicitly asks about team-wide or industry-wide processes.

---

# Section 2: Epistemic Hedging, Qualifiers, & Non-Contracted Emphasis

## Epistemic Hedging & Qualifier Inflation
- **Linguistic Markers:** Sudden cluster of qualifiers and hedges: *"to be completely honest"*, *"basically"*, *"technically speaking"*, *"as far as I recall"*, *"sort of"*, *"probably"*.
- **Behavioral Correlates:**
  1. **Commitment Avoidance:** High density of epistemic hedges paired with rising terminal pitch ($F_0$ uptalk at declarative sentence ends) and reduced vocal loudness ($Z_{\text{loudness}} \le -1.2$) indicates **low epistemic confidence** or reluctance to commit to a definitive claim.
  2. **Bolstering Phrases (*"To tell you the truth"*, *"Honestly"*):** When accompanied by a **self-touch adaptor** or **post-utterance lip press (AU24)**, bolstering phrases reflect heightened awareness of listener skepticism and an effort to manage impression.

## Tense Shifting & Verbatim Question Repetition
- **Linguistic Markers:**
  1. **Past-to-Present Tense Slip:** Describing a past decision in the present tense (or vice versa) signals active mental simulation or unresolved emotional salience.
  2. **Question Echoing:** Repeating the interviewer's question verbatim before answering, paired with `response_latency_ms > 1000` and `AU04` brow lowering, serves as a **cognitive stalling buffer** to buy working-memory processing time.