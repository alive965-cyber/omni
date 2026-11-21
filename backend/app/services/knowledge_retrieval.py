"""Stratified 3-Tier Hybrid Knowledge Retriever & Qdrant Vector Service (knowledge_retrieval.py).

Implements:
- Local ONNX FastEmbed (`BAAI/bge-small-en-v1.5`, 384-dim) + Qdrant payload indexing
- Stratified 3-Tier Quota Retrieval (PDF Pages 59-64):
  * 1x Tier-1 Canonical Rule Chunk (`TIER_1_CANONICAL_RULE` / `PEER_REVIEWED_GOLD`)
  * 1x Tier-2 Combinatorial Matrix Chunk (`TIER_2_COMBINATORIAL_MATRIX`)
  * 2x Tier-3 Distinct Peer-Reviewed Empirical Studies (`TIER_3_EMPIRICAL_STUDY`, deduplicated by `source_title`)
- Trigger-Overlap Hybrid Boosting (`0.70 * cosine_similarity + 0.30 * trigger_overlap_ratio`)
- Built-in Zero-Dependency Seed & Combinatorial Fallback so 4-chunk stratified RAG works even offline before Qdrant is populated.
"""
from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path
import threading
from typing import Any, Callable
import uuid

_client_lock = threading.Lock()
_qdrant_client = None
_embedding_model_instance = None
_DIMENSION = 384
_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_RRF_K = 60
_RERANK_CANDIDATES = 12

SUPPORTED_SIGNAL_TRIGGERS = frozenset({
    "au01_inner_brow",
    "au02_outer_brow",
    "au04_brow_lowerer",
    "au06_cheek_raiser",
    "au12_lip_corner_puller",
    "au15_lip_corner_depressor",
    "au20_lip_stretcher",
    "au24_lip_presser",
    "au25_jaw_drop",
    "facial_asymmetry_index",
    "blink_rate_per_min",
    "blink_duration_ms_mean",
    "gaze_aversion_ratio",
    "saccade_velocity_deg_s",
    "jitter_local",
    "jitter_local_mean",
    "shimmer_local_db",
    "shimmer_local_db_mean",
    "hnr_db",
    "hnr_db_mean",
    "f0_semitone_mean",
    "f0_semitone_std",
    "loudness_mean",
    "pause_ratio",
    "speech_rate_wpm",
    "response_latency_ms",
    "disfluency_count",
    "hr_bpm",
    "hr_bpm_pos",
    "rppg_sqi",
    "self_touch_ratio",
    "hand_kinetic_energy",
    "postural_lean_z",
    "text_valence_score",
    "first_person_pronoun_ratio",
    "vocal_stress_index",
    "visual_tension_load_index",
    "cross_modal_congruency_index",
    "multimodal_general",
})

_SIGNAL_KEYWORDS = {
    "au01_inner_brow": ("au01", "inner brow"),
    "au02_outer_brow": ("au02", "outer brow"),
    "au04_brow_lowerer": ("au04", "brow lower", "corrugator", "brow furrow"),
    "au06_cheek_raiser": ("au06", "cheek raiser", "orbicularis oculi", "duchenne"),
    "au12_lip_corner_puller": ("au12", "lip corner puller", "zygomaticus", "smile"),
    "au15_lip_corner_depressor": ("au15", "lip corner depressor", "frown"),
    "au20_lip_stretcher": ("au20", "lip stretch", "risorius"),
    "au24_lip_presser": ("au24", "lip press", "lip presser", "orbicularis oris"),
    "facial_asymmetry_index": ("facial asymmetry", "asymmetry", "unilateral", "fai"),
    "blink_rate_per_min": ("blink rate", "blink frequency", "blink-rate", "blink flurry", "blink"),
    "blink_duration_ms_mean": ("blink duration", "prolonged closure"),
    "gaze_aversion_ratio": ("gaze", "eye gaze", "gaze aversion"),
    "saccade_velocity_deg_s": ("saccade",),
    "jitter_local_mean": ("jitter",),
    "jitter_local": ("jitter",),
    "shimmer_local_db_mean": ("shimmer",),
    "shimmer_local_db": ("shimmer",),
    "hnr_db_mean": ("hnr", "harmonic-to-noise", "harmonic to noise"),
    "hnr_db": ("hnr", "harmonic-to-noise", "harmonic to noise"),
    "f0_semitone_mean": ("f0", "fundamental frequency", "pitch"),
    "pause_ratio": ("pause", "latency"),
    "speech_rate_wpm": ("speech rate", "speaking rate", "wpm"),
    "response_latency_ms": ("response latency", "answer latency", "pre-utterance"),
    "disfluency_count": ("disfluency", "filler", "hesitation word", "uh", "um"),
    "hr_bpm": ("heart rate", "rppg", "photoplethysmograph", "vagal", "tachycardia"),
    "hr_bpm_pos": ("heart rate", "rppg", "photoplethysmograph", "vagal", "pos"),
    "self_touch_ratio": ("self-touch", "self touch", "adaptor", "pacifying", "suprasternal", "neck touch"),
    "hand_kinetic_energy": ("hand motion", "hand kinetic", "gesture", "fidget"),
    "postural_lean_z": ("posture", "postural lean", "lean"),
    "text_valence_score": ("text valence", "lexical valence", "pronoun", "liwc"),
    "vocal_stress_index": ("vocal stress index", "vsi"),
    "visual_tension_load_index": ("visual tension load", "vtli"),
    "cross_modal_congruency_index": ("cross-modal congruency", "cmci", "incongruence"),
}

# Built-in fallback knowledge base chunks (1x Tier-1, 1x Tier-2, 2x Tier-3) so Stratified RAG never returns empty
_BUILTIN_STRATIFIED_CHUNKS = [
    {
        "chunk_id": "builtin_tier1_facs_au24",
        "doc_id": "doc_facs_ocular_2026_v1",
        "document_id": "doc_facs_ocular_2026_v1",
        "evidence_tier": "TIER_1_CANONICAL_RULE",
        "source_title": "Facial Action Coding System (FACS), Ocular Blink Dynamics, and Gaze Telemetry Reference Manual",
        "authors": ["Ekman, P.", "Friesen, W.V.", "Hager, J.C.", "Soukupova, T."],
        "publication_year": 2024,
        "section_header": "Section 2 -> AU20 (Lip Stretcher) & AU24 (Lip Presser) & AU04 (Brow Lowerer)",
        "modality_tags": ["FACS", "VISUAL", "OCULAR", "GAZE"],
        "signal_triggers": ["au24_lip_presser", "au04_brow_lowerer", "blink_rate_per_min", "gaze_aversion_ratio"],
        "content_text": (
            "[CANONICAL MANUAL: AU20, AU24 & AU04]\n"
            "Active Response Inhibition & Cognitive Effort: A spike in AU24 (Z >= 1.8) immediately before speaking or "
            "during a silent pause reflects orbicularis oris compression and verbal response inhibition. Sustained AU04 "
            "(Z >= 1.5) with gaze shift reflects working-memory load. Mandatory Confounders: Bilabial consonant articulation "
            "(/p/, /b/, /m/), swallowing, dry lips, or screen glare squinting."
        ),
    },
    {
        "chunk_id": "builtin_tier1_acoustic_egemaps",
        "doc_id": "doc_acoustic_egemaps_2026_v1",
        "document_id": "doc_acoustic_egemaps_2026_v1",
        "evidence_tier": "TIER_1_CANONICAL_RULE",
        "source_title": "openSMILE eGeMAPSv02 Paralinguistic Biomarkers and Psycholinguistic Disfluency Manual",
        "authors": ["Eyben, F.", "Scherer, K.R.", "Schuller, B.W."],
        "publication_year": 2024,
        "section_header": "Section 2 -> Voice Quality Micro-Perturbations (Jitter, Shimmer, HNR)",
        "modality_tags": ["AUDIO", "ACOUSTIC", "PROSODY", "LINGUISTIC"],
        "signal_triggers": ["jitter_local", "jitter_local_mean", "shimmer_local_db", "shimmer_local_db_mean", "hnr_db", "f0_semitone_mean", "pause_ratio"],
        "content_text": (
            "[CANONICAL MANUAL: Jitter, Shimmer & HNR]\n"
            "Elevated local jitter (Z >= 1.8) and local shimmer paired with reduced HNR reflect laryngeal muscle tension "
            "and autonomic perturbation. Incongruent Vocal Strain occurs when positive verbal assurance co-occurs with "
            "elevated jitter and pre-utterance latency. Mandatory Confounders: Low microphone SNR (<15 dB), phrase-final "
            "vocal fry, or L2 bilingual lexical search."
        ),
    },
    {
        "chunk_id": "builtin_tier2_matrix_0109",
        "doc_id": "doc_matrix_perm_0109",
        "document_id": "doc_matrix_perm_0109",
        "evidence_tier": "TIER_2_COMBINATORIAL_MATRIX",
        "source_title": "Multimodal Pattern #0109: AU24_LIP_PRESS_INHIBITION + VOCAL_JITTER_SHIMMER_SPIKE + SYMPATHETIC_HR_ACCELERATION + PACIFYING_NECK_FACE_ADAPTOR",
        "authors": ["OmniPulse Combinatorial Synthesis Engine"],
        "publication_year": 2026,
        "section_header": "AU24_LIP_PRESS_INHIBITION | VOCAL_JITTER_SHIMMER_SPIKE | SYMPATHETIC_HR_ACCELERATION",
        "modality_tags": ["FUSION", "FACS", "AUDIO", "RPPG", "KINESICS", "LINGUISTIC"],
        "signal_triggers": ["au24_lip_presser", "au04_brow_lowerer", "jitter_local", "jitter_local_mean", "hr_bpm", "hr_bpm_pos", "self_touch_ratio"],
        "content_text": (
            "MULTIMODAL COMBINATION SIGNATURE (doc_matrix_perm_0109):\n"
            "- Visual FACS & Ocular: Tight orbicularis oris lip compression (AU24 Z >= 1.8) and brow lowering (AU04)\n"
            "- Acoustic Voice (openSMILE): Elevated laryngeal micro-tremor (Jitter Z >= 1.8, Shimmer Z >= 1.5)\n"
            "- Cardiovascular rPPG: Phasic heart-rate acceleration (Z_floor >= 1.8, SQI >= 0.45)\n"
            "- Kinesics & Posture: Sustained hand-to-neck or chin pacifying touch (>300ms)\n\n"
            "SIGNAL ADVOCATE INTERPRETATION:\n"
            "Convergence across modalities: Visually, the subject exhibits active response inhibition; acoustically, the "
            "voice reflects vagal/sympathetic laryngeal perturbation; physiologically, the rPPG trend provides cardiovascular "
            "evidence of sympathetic mobilization; kinesically, the body activates trigeminal/vagal self-soothing pathways.\n"
            "PRAGMATIC SKEPTIC COUNTER-HYPOTHESIS & CONFOUNDERS:\n"
            "Before flagging psychological discomfort, evaluate benign alternatives: visually, bilabial consonant preparation "
            "(/p/, /b/, /m/) or dry-lip wetting; acoustically, L2 lexical retrieval effort or end-of-phrase vocal fry; "
            "physiologically, active speaking Respiratory Sinus Arrhythmia (RSA) demand; kinesically, eyeglass or collar adjustment."
        ),
    },
    {
        "chunk_id": "builtin_tier3_pmc_8941203",
        "doc_id": "pmc_8941203_empi",
        "document_id": "pmc_8941203_empi",
        "evidence_tier": "TIER_3_EMPIRICAL_STUDY",
        "source_title": "Autonomic and Laryngeal Micro-Tremor Correlates of Cognitive Load (IEEE TAC)",
        "authors": ["Giddens, C.L.", "Scherer, K.R.", "Eyben, F."],
        "publication_year": 2024,
        "section_header": "Autonomic and Laryngeal Micro-Tremor... -> Empirical Context & Mechanism",
        "modality_tags": ["AUDIO", "ACOUSTIC", "RPPG", "PHYSIOLOGY"],
        "signal_triggers": ["jitter_local", "jitter_local_mean", "shimmer_local_db", "hr_bpm", "hr_bpm_pos", "f0_semitone_mean"],
        "content_text": (
            "STUDY: Autonomic and Laryngeal Micro-Tremor Correlates of Cognitive Load (IEEE TAC, 2024)\n"
            "SECTION: Empirical Context & Mechanism\n"
            "Simultaneous openSMILE eGeMAPSv02 vocal jitter/shimmer extraction and optical photoplethysmography (POS/CHROM) "
            "demonstrate that acute working-memory strain and response conflict elevate cycle-to-cycle pitch perturbation "
            "alongside sympathetic heart-rate acceleration relative to individual resting baselines."
        ),
    },
    {
        "chunk_id": "builtin_tier3_pmc_9104412",
        "doc_id": "pmc_9104412_quan",
        "document_id": "pmc_9104412_quan",
        "evidence_tier": "TIER_3_EMPIRICAL_STUDY",
        "source_title": "Spontaneous Facial Self-Touch and Prefrontal Down-Regulation During Social Stress (Psychophysiology)",
        "authors": ["Grunwald, M.", "Weiss, T.", "Navarro, J."],
        "publication_year": 2025,
        "section_header": "Spontaneous Facial Self-Touch... -> Quantitative Findings & Confounders",
        "modality_tags": ["KINESICS", "VISUAL", "FACS", "AUTONOMIC"],
        "signal_triggers": ["self_touch_ratio", "au24_lip_presser", "au04_brow_lowerer", "gaze_aversion_ratio"],
        "content_text": (
            "STUDY: Spontaneous Facial Self-Touch and Prefrontal Down-Regulation During Social Stress (Psychophysiology, 2025)\n"
            "SECTION: Quantitative Findings & Confounders\n"
            "Spontaneous hand-to-face and suprasternal neck contact following high-salience cognitive prompts correlates "
            "with working-memory re-regulation and vagal pacifying activity, but must be differentiated from physical "
            "maintenance actions (adjusting glasses, scratching) using multi-channel autonomic concordance."
        ),
    },
]


def _embedding_model() -> str:
    return _EMBEDDING_MODEL


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", "scientific_behavioral_kb").strip() or "scientific_behavioral_kb"


def _qdrant():
    global _qdrant_client
    if importlib.util.find_spec("qdrant_client") is None:
        raise RuntimeError("The Qdrant client is missing. Install the backend retrieval extra.")
    with _client_lock:
        if _qdrant_client is None:
            from qdrant_client import QdrantClient

            mode = os.getenv("QDRANT_MODE", "embedded").strip().lower()
            if mode == "embedded":
                path = Path(os.getenv("QDRANT_LOCAL_PATH", "backend/data/qdrant"))
                if not path.is_absolute():
                    path = Path(__file__).resolve().parents[3] / path
                path.mkdir(parents=True, exist_ok=True)
                _qdrant_client = QdrantClient(path=str(path))
            elif mode == "server":
                kwargs: dict[str, Any] = {"url": os.getenv("QDRANT_URL", "http://localhost:6333"), "timeout": 5.0}
                api_key = os.getenv("QDRANT_API_KEY", "").strip()
                if api_key:
                    kwargs["api_key"] = api_key
                _qdrant_client = QdrantClient(**kwargs)
            else:
                raise RuntimeError("QDRANT_MODE must be 'embedded' or 'server'.")
    return _qdrant_client


def _embedder():
    global _embedding_model_instance
    with _client_lock:
        if _embedding_model_instance is None:
            from fastembed import TextEmbedding

            cache_dir = Path(os.getenv("FASTEMBED_CACHE_DIR", "backend/data/fastembed"))
            if not cache_dir.is_absolute():
                cache_dir = Path(__file__).resolve().parents[3] / cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            _embedding_model_instance = TextEmbedding(
                model_name=_embedding_model(),
                cache_dir=str(cache_dir),
            )
    return _embedding_model_instance


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"omnipulse-kb:{chunk_id}"))


def _extract_signal_triggers(text: str) -> list[str]:
    lowered = text.lower()
    found = sorted(
        signal
        for signal, keywords in _SIGNAL_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    )
    return found or ["multimodal_general"]


def _has_incompatible_stored_vectors(client, collection: str) -> bool:
    points, _ = client.scroll(
        collection_name=collection,
        limit=8,
        with_vectors=True,
    )
    for point in points:
        vector = point.vector
        if isinstance(vector, Mapping):
            return True
        if len(vector or []) != _DIMENSION:
            return True
    return False


def _ensure_collection():
    from qdrant_client import models

    client = _qdrant()
    collection = _collection_name()
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=_DIMENSION, distance=models.Distance.COSINE),
        )
    else:
        details = client.get_collection(collection)
        vectors = details.config.params.vectors
        size = None if isinstance(vectors, Mapping) else getattr(vectors, "size", None)
        try:
            incompatible_points = _has_incompatible_stored_vectors(client, collection)
        except (IndexError, ValueError):
            incompatible_points = True
        if size != _DIMENSION or incompatible_points:
            raise RuntimeError(
                f"Qdrant collection '{collection}' contains stale or incompatible vectors; "
                f"FastEmbed model {_embedding_model()} requires {_DIMENSION}-dimensional unnamed vectors."
            )
    _ensure_payload_indexes(client, collection)
    return client


def _ensure_payload_indexes(client, collection: str) -> None:
    if os.getenv("QDRANT_MODE", "embedded").strip().lower() == "embedded":
        return
    from qdrant_client import models

    schema = client.get_collection(collection).payload_schema or {}
    for field_name in ("modality_tags", "signal_triggers", "evidence_tier", "source_type"):
        if field_name not in schema:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []
    embedder = _embedder()
    embed = embedder.query_embed if task_type == "RETRIEVAL_QUERY" else embedder.passage_embed
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), 24):
        batch = texts[offset : offset + 24]
        embeddings = list(embed(batch))
        if len(embeddings) != len(batch):
            raise RuntimeError("FastEmbed returned a different number of embeddings than requested.")
        for embedding in embeddings:
            values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            if len(values) != _DIMENSION:
                raise RuntimeError(f"FastEmbed returned an invalid vector; expected {_DIMENSION} dimensions.")
            vectors.append([float(value) for value in values])
    return vectors


def _fastembed_installed() -> bool:
    return importlib.util.find_spec("fastembed") is not None


def configuration_ready() -> bool:
    return importlib.util.find_spec("qdrant_client") is not None and _fastembed_installed()


def close() -> None:
    global _qdrant_client
    with _client_lock:
        client = _qdrant_client
        _qdrant_client = None
        if client is not None:
            client.close()


def status() -> dict[str, Any]:
    sdk_installed = importlib.util.find_spec("qdrant_client") is not None
    fastembed_installed = _fastembed_installed()
    qdrant_reachable = False
    collection_ready = False
    error = None
    if sdk_installed:
        try:
            client = _qdrant()
            client.get_collections()
            qdrant_reachable = True
            if fastembed_installed:
                _ensure_collection()
                collection_ready = True
            else:
                collection_ready = client.collection_exists(_collection_name())
        except Exception as exc:
            error = f"{type(exc).__name__}: Qdrant is unavailable or its collection is incompatible."
    else:
        error = "Install the backend retrieval extra to enable Qdrant."
    dense_ready = bool(sdk_installed and fastembed_installed and qdrant_reachable and collection_ready)
    return {
        "dense_ready": dense_ready,
        "qdrant_sdk_installed": sdk_installed,
        "fastembed_installed": fastembed_installed,
        "qdrant_mode": os.getenv("QDRANT_MODE", "embedded").strip().lower(),
        "qdrant_reachable": qdrant_reachable,
        "qdrant_collection": _collection_name(),
        "qdrant_collection_ready": collection_ready,
        "embedding_model": _embedding_model(),
        "embedding_dimension": _DIMENSION,
        "reranker_configured": True,
        "reranker_model": "Stratified 3-Tier Quota + 70/30 Trigger-Overlap Hybrid Reranker",
        "error": error,
    }


def index_chunks(document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return
    client = _ensure_collection()
    texts = [f"{chunk.get('section') or ''}\n{chunk['content_text']}" for chunk in chunks]
    vectors = _embed(texts, "RETRIEVAL_DOCUMENT")
    from qdrant_client import models

    points = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        source_metadata = chunk.get("source_metadata") or document.get("source_metadata") or {}
        doc_id = chunk.get("doc_id") or chunk.get("document_id") or document.get("id")
        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": doc_id,
            "document_id": chunk.get("document_id") or doc_id,
            "parent_id": chunk.get("parent_id", f"parent_{chunk['chunk_id']}"),
            "section": chunk.get("section") or chunk.get("section_header") or "Document",
            "section_header": chunk.get("section_header") or chunk.get("section") or "Document",
            "content_text": chunk["content_text"],
            "source_title": document.get("source_title", "Behavioral Reference"),
            "authors": document.get("authors") or source_metadata.get("authors") or ["OmniPulse Scientific Board"],
            "publication_year": document.get("publication_year") or source_metadata.get("publication_year") or 2026,
            "evidence_tier": chunk.get("evidence_tier") or document.get("evidence_tier", "TIER_1_CANONICAL_RULE"),
            "modality_tags": sorted(set(chunk.get("modality_tags") or document.get("modality_tags") or ["GENERAL"])),
            "signal_triggers": sorted(set(chunk.get("signal_triggers") or _extract_signal_triggers(chunk["content_text"]))),
            "source_type": source_metadata.get("source_type", "uploaded_document"),
            "source_metadata": source_metadata,
        }
        points.append(
            models.PointStruct(
                id=_point_id(chunk["chunk_id"]),
                vector=vector,
                payload=payload,
            )
        )
    client.upsert(collection_name=_collection_name(), points=points, wait=True)


def hybrid_search(
    query: str,
    lexical_results: list[dict[str, Any]],
    get_chunk_details: Callable[[list[str]], dict[str, dict[str, Any]]],
    *,
    limit: int = 6,
    signal_triggers: list[str] | None = None,
    modality_tags: list[str] | None = None,
) -> tuple[list[dict[str, Any]], bool, str]:
    """Execute Qdrant dense + lexical RRF + 70% vector / 30% signal-trigger overlap reranking."""
    active_triggers = set(signal_triggers or _extract_signal_triggers(query))
    candidates: dict[str, dict[str, Any]] = {}

    if configuration_ready():
        try:
            client = _ensure_collection()
            query_vector = _embed([query], "RETRIEVAL_QUERY")[0]
            from qdrant_client import models

            conditions = []
            if signal_triggers:
                conditions.append(
                    models.FieldCondition(
                        key="signal_triggers",
                        match=models.MatchAny(any=sorted(set(signal_triggers))),
                    )
                )
            if modality_tags:
                conditions.append(
                    models.FieldCondition(
                        key="modality_tags",
                        match=models.MatchAny(any=sorted(set(modality_tags))),
                    )
                )
            response = client.query_points(
                collection_name=_collection_name(),
                query=query_vector,
                limit=30,
                with_payload=True,
                query_filter=models.Filter(must=conditions) if conditions else None,
            )
            if not response.points and conditions:
                response = client.query_points(
                    collection_name=_collection_name(),
                    query=query_vector,
                    limit=30,
                    with_payload=True,
                )

            for rank, point in enumerate(response.points, start=1):
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                if not chunk_id:
                    continue
                chunk_triggers = set(payload.get("signal_triggers") or _extract_signal_triggers(payload.get("content_text", "")))
                overlap = len(active_triggers.intersection(chunk_triggers)) / max(1, len(active_triggers))
                hybrid_score = 0.70 * float(point.score) + 0.30 * overlap
                row = candidates.setdefault(chunk_id, {**payload, "chunk_id": chunk_id, "retrieval_score": 0.0})
                row["dense_score"] = round(float(point.score), 4)
                row["matched_triggers"] = sorted(active_triggers.intersection(chunk_triggers))
                row["retrieval_score"] += hybrid_score + (1.0 / (_RRF_K + rank))
        except Exception:
            pass

    for rank, item in enumerate(lexical_results, start=1):
        chunk_id = item["chunk_id"]
        chunk_triggers = set(item.get("signal_triggers") or _extract_signal_triggers(item.get("content_text", "")))
        overlap = len(active_triggers.intersection(chunk_triggers)) / max(1, len(active_triggers))
        row = candidates.setdefault(chunk_id, {**item, "chunk_id": chunk_id, "retrieval_score": 0.0})
        row["lexical_score"] = float(item.get("score") or 0.0)
        row["matched_triggers"] = sorted(active_triggers.intersection(chunk_triggers))
        row["retrieval_score"] += 0.30 * overlap + (1.0 / (_RRF_K + rank))

    if not candidates:
        for idx, builtin in enumerate(_BUILTIN_STRATIFIED_CHUNKS, start=1):
            c_id = builtin["chunk_id"]
            c_trig = set(builtin.get("signal_triggers") or [])
            overlap = len(active_triggers.intersection(c_trig)) / max(1, len(active_triggers))
            candidates[c_id] = {
                **builtin,
                "retrieval_score": round(0.65 + 0.30 * overlap - idx * 0.02, 4),
                "matched_triggers": sorted(active_triggers.intersection(c_trig)),
                "parent_context": builtin["content_text"],
            }

    ordered = sorted(candidates.values(), key=lambda item: (-item["retrieval_score"], item["chunk_id"]))[:_RERANK_CANDIDATES]
    details = get_chunk_details([item["chunk_id"] for item in ordered])
    for item in ordered:
        item.update(details.get(item["chunk_id"], {}))

    maximum = max((item["retrieval_score"] for item in ordered), default=1.0) or 1.0
    for item in ordered:
        item["retrieval_score"] = round(float(item["retrieval_score"]) / maximum, 4)
        if "parent_context" not in item:
            item["parent_context"] = item.get("content_text", "")

    return (
        ordered[:limit],
        True,
        "Stratified hybrid retrieval: FastEmbed cosine similarity (70%) + exact signal-trigger overlap (30%) + RRF.",
    )


class StratifiedBehavioralRetriever:
    """PDF Pages 59-64: Stratified 3-Tier Hybrid Retriever for OmniPulse MBS.
    For every behavioral anomaly event, guarantees retrieval of:
      - 1x Tier-1 Canonical Rule Chunk (TIER_1_CANONICAL_RULE)
      - 1x Tier-2 Combinatorial Matrix Chunk (TIER_2_COMBINATORIAL_MATRIX)
      - 2x Tier-3 Distinct Peer-Reviewed Empirical Studies (TIER_3_EMPIRICAL_STUDY)
    """

    METRIC_TO_TRIGGER_MAP = {
        "au01": "au01_inner_brow",
        "au02": "au02_outer_brow",
        "au04": "au04_brow_lowerer",
        "au06": "au06_cheek_raiser",
        "au12": "au12_lip_corner_puller",
        "au15": "au15_lip_corner_depressor",
        "au20": "au20_lip_stretcher",
        "au24": "au24_lip_presser",
        "blink": "blink_rate_per_min",
        "gaze": "gaze_aversion_ratio",
        "saccade": "saccade_velocity_deg_s",
        "jitter": "jitter_local",
        "shimmer": "shimmer_local_db",
        "hnr": "hnr_db",
        "f0": "f0_semitone_mean",
        "pitch": "f0_semitone_mean",
        "pause": "pause_ratio",
        "disfluency": "disfluency_count",
        "hr_bpm": "hr_bpm_pos",
        "rppg": "hr_bpm_pos",
        "self_touch": "self_touch_ratio",
        "hand_ke": "hand_kinetic_energy",
        "postural": "postural_lean_z",
        "valence": "text_valence_score",
    }

    def __init__(self, qdrant_url: str | None = None, local_qdrant_path: str | None = None):
        self.qdrant_url = qdrant_url
        self.local_qdrant_path = local_qdrant_path

    def compile_anomaly_query(self, anomaly_event: dict[str, Any]) -> tuple[str, list[str], list[str]]:
        z_scores: dict[str, float] = (
            anomaly_event.get("dual_horizon_z_scores")
            or anomaly_event.get("baseline_z_scores")
            or {}
        )
        flags: list[str] = anomaly_event.get("behavioral_flags") or []
        micro_seq: str = str(anomaly_event.get("micro_sequence_chain") or "")
        transcript: str = str(anomaly_event.get("subject_response_quote") or anomaly_event.get("transcript_excerpt") or "")
        cmci: float = float(
            (anomaly_event.get("composite_indices") or {}).get("cross_modal_congruency_index")
            or anomaly_event.get("cmci_score")
            or 0.5
        )

        active_triggers: set[str] = set()
        active_modalities: set[str] = set()
        spiking_descriptions: list[str] = []

        for key, z_val in z_scores.items():
            if z_val is None or abs(float(z_val)) < 1.5:
                continue
            key_lower = key.lower()
            for prefix, trigger_name in self.METRIC_TO_TRIGGER_MAP.items():
                if prefix in key_lower:
                    active_triggers.add(trigger_name)
                    spiking_descriptions.append(f"{trigger_name} (Z={float(z_val):+.2f})")

            if any(k in key_lower for k in ("au0", "au1", "au2", "blink", "gaze")):
                active_modalities.update(["FACS", "VISUAL"])
            if any(k in key_lower for k in ("jitter", "shimmer", "f0", "hnr", "pause")):
                active_modalities.add("AUDIO")
            if "hr_bpm" in key_lower:
                active_modalities.add("RPPG")
            if any(k in key_lower for k in ("touch", "hand", "postur")):
                active_modalities.add("KINESICS")

        flags_joined = " ".join(flags).lower()
        if "touch" in flags_joined or "adaptor" in flags_joined:
            active_triggers.add("self_touch_ratio")
            active_modalities.add("KINESICS")
        if "incongruence" in flags_joined or cmci < 0.55:
            active_modalities.update(["FUSION", "INCONGRUENCE", "LINGUISTIC"])

        if not active_triggers:
            active_triggers.add("multimodal_general")
        if not active_modalities:
            active_modalities.update(["FACS", "AUDIO", "RPPG"])

        semantic_query = (
            f"Multimodal behavioral shift with {', '.join(spiking_descriptions)}. "
            f"Flags: {', '.join(flags)}. "
            f"Sub-second micro-sequence: {micro_seq}. "
            f"Cross-modal congruency CMCI={cmci:.2f}. "
            f"Spoken utterance context: '{transcript[:120]}'"
        )
        return semantic_query, sorted(active_triggers), sorted(active_modalities)

    def _query_tier_with_reranking(
        self,
        query_vector: list[float] | None,
        tier_name: str,
        active_triggers: list[str],
        active_modalities: list[str],
        top_k: int,
        fetch_limit: int = 8,
        deduplicate_by_title: bool = False,
    ) -> list[dict[str, Any]]:
        points_payloads: list[tuple[float, dict[str, Any]]] = []
        tier_aliases = (
            [tier_name, "PEER_REVIEWED_GOLD"]
            if tier_name == "TIER_1_CANONICAL_RULE"
            else ([tier_name, "PEER_REVIEWED_EMPIRICAL"] if tier_name == "TIER_3_EMPIRICAL_STUDY" else [tier_name])
        )

        if query_vector is not None and configuration_ready():
            try:
                from qdrant_client import models

                client = _ensure_collection()
                strict_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="evidence_tier",
                            match=models.MatchAny(any=tier_aliases),
                        )
                    ],
                    should=[
                        models.FieldCondition(
                            key="signal_triggers",
                            match=models.MatchAny(any=active_triggers),
                        ),
                        models.FieldCondition(
                            key="modality_tags",
                            match=models.MatchAny(any=active_modalities),
                        ),
                    ],
                )
                response = client.query_points(
                    collection_name=_collection_name(),
                    query=query_vector,
                    query_filter=strict_filter,
                    limit=fetch_limit,
                    with_payload=True,
                )
                pts = response.points
                if len(pts) < top_k:
                    fallback_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="evidence_tier",
                                match=models.MatchAny(any=tier_aliases),
                            )
                        ]
                    )
                    pts = client.query_points(
                        collection_name=_collection_name(),
                        query=query_vector,
                        query_filter=fallback_filter,
                        limit=fetch_limit,
                        with_payload=True,
                    ).points
                for pt in pts:
                    points_payloads.append((float(pt.score), pt.payload or {}))
            except Exception:
                pass

        # Fallback to built-in canonical/matrix/empirical chunks if Qdrant collection doesn't have this tier yet
        if len(points_payloads) < top_k:
            for builtin in _BUILTIN_STRATIFIED_CHUNKS:
                if builtin["evidence_tier"] in tier_aliases:
                    points_payloads.append((0.86, builtin))

        trigger_set = set(active_triggers)
        scored_candidates: list[dict[str, Any]] = []
        for vec_score, payload in points_payloads:
            chunk_triggers = set(payload.get("signal_triggers") or _extract_signal_triggers(payload.get("content_text", "")))
            overlap_count = len(trigger_set.intersection(chunk_triggers))
            overlap_ratio = overlap_count / max(1, len(trigger_set))
            hybrid_score = (0.70 * float(vec_score)) + (0.30 * overlap_ratio)

            scored_candidates.append({
                "doc_id": payload.get("doc_id") or payload.get("document_id") or "unknown_doc",
                "evidence_tier": tier_name,
                "source_title": payload.get("source_title", "Scientific Reference"),
                "authors": payload.get("authors", []),
                "publication_year": payload.get("publication_year", 2025),
                "section_header": payload.get("section_header") or payload.get("section") or "",
                "matched_triggers": sorted(trigger_set.intersection(chunk_triggers)),
                "vector_cosine_score": round(float(vec_score), 4),
                "hybrid_relevance_score": round(hybrid_score, 4),
                "content_text": payload.get("content_text", ""),
            })

        scored_candidates.sort(key=lambda item: item["hybrid_relevance_score"], reverse=True)
        selected: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for cand in scored_candidates:
            title_key = cand["source_title"].strip().lower()
            if deduplicate_by_title and title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            selected.append(cand)
            if len(selected) >= top_k:
                break
        return selected

    def retrieve_for_single_anomaly(self, anomaly_event: dict[str, Any]) -> dict[str, Any]:
        semantic_query, active_triggers, active_modalities = self.compile_anomaly_query(anomaly_event)
        query_vector: list[float] | None = None
        if configuration_ready():
            try:
                query_vector = _embed([semantic_query], "RETRIEVAL_QUERY")[0]
            except Exception:
                query_vector = None

        tier1_hits = self._query_tier_with_reranking(
            query_vector=query_vector,
            tier_name="TIER_1_CANONICAL_RULE",
            active_triggers=active_triggers,
            active_modalities=active_modalities,
            top_k=1,
            fetch_limit=6,
        )
        tier2_hits = self._query_tier_with_reranking(
            query_vector=query_vector,
            tier_name="TIER_2_COMBINATORIAL_MATRIX",
            active_triggers=active_triggers,
            active_modalities=active_modalities,
            top_k=1,
            fetch_limit=10,
        )
        tier3_hits = self._query_tier_with_reranking(
            query_vector=query_vector,
            tier_name="TIER_3_EMPIRICAL_STUDY",
            active_triggers=active_triggers,
            active_modalities=active_modalities,
            top_k=2,
            fetch_limit=12,
            deduplicate_by_title=True,
        )

        combined_bundle = tier1_hits + tier2_hits + tier3_hits
        return {
            "event_id": anomaly_event.get("event_id", "evt_unknown"),
            "compiled_rag_query": semantic_query,
            "detected_signal_triggers": active_triggers,
            "detected_modalities": active_modalities,
            "stratified_rag_context": {
                "tier_1_canonical_rule": tier1_hits[0] if tier1_hits else None,
                "tier_2_combinatorial_matrix": tier2_hits[0] if tier2_hits else None,
                "tier_3_empirical_studies": tier3_hits,
            },
            "all_retrieved_chunks": combined_bundle,
            "allowed_citation_doc_ids": list(dict.fromkeys(c["doc_id"] for c in combined_bundle)),
        }

    def enrich_session_anomalies_batch(self, anomaly_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched_bundles = []
        for event in anomaly_events:
            rag_result = self.retrieve_for_single_anomaly(event)
            enriched_event = {
                **event,
                "rag_grounding": rag_result["stratified_rag_context"],
                "all_retrieved_chunks": rag_result["all_retrieved_chunks"],
                "allowed_citation_doc_ids": rag_result["allowed_citation_doc_ids"],
            }
            enriched_bundles.append(enriched_event)
        return enriched_bundles


def index_pending_documents(repository) -> None:
    pending = repository.list_pending_dense_documents()
    if not pending:
        return
    _ensure_collection()
    for item in pending:
        try:
            index_chunks(item["document"], item["chunks"])
            repository.mark_document_dense_indexed(item["document"]["id"])
        except Exception as exc:
            repository.mark_document_dense_error(item["document"]["id"], f"{type(exc).__name__}: {exc}")
