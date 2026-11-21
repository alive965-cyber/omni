"""Session Lifecycle & RAG Document Ingestion Finite State Machines (PDF Section 7.1 & 9.4)."""
from enum import StrEnum


class SessionState(StrEnum):
    CREATED = "CREATED"
    CALIBRATING = "CALIBRATING"
    CALIBRATION_RETRY_REQUIRED = "CALIBRATION_RETRY_REQUIRED"
    RECORDING_LIVE = "RECORDING_LIVE"
    PROCESSING_ALIGNMENT = "PROCESSING_ALIGNMENT"
    SYNTHESIZING_RAG_REPORT = "SYNTHESIZING_RAG_REPORT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


TRANSITIONS = {
    SessionState.CREATED: {SessionState.CALIBRATING},
    SessionState.CALIBRATING: {
        SessionState.RECORDING_LIVE,
        SessionState.CALIBRATION_RETRY_REQUIRED,
        SessionState.PROCESSING_ALIGNMENT,
    },
    SessionState.CALIBRATION_RETRY_REQUIRED: {
        SessionState.CALIBRATING,
        SessionState.PROCESSING_ALIGNMENT,
    },
    SessionState.RECORDING_LIVE: {SessionState.PROCESSING_ALIGNMENT},
    SessionState.PROCESSING_ALIGNMENT: {
        SessionState.SYNTHESIZING_RAG_REPORT,
        SessionState.FAILED,
    },
    SessionState.SYNTHESIZING_RAG_REPORT: {
        SessionState.COMPLETED,
        SessionState.FAILED,
    },
    SessionState.COMPLETED: set(),
    SessionState.FAILED: {SessionState.PROCESSING_ALIGNMENT},
}


def transition(current: str, requested: SessionState) -> str:
    state = SessionState(current)
    if requested not in TRANSITIONS[state]:
        raise ValueError(f"Invalid session transition: {state.value} → {requested.value}")
    return requested.value


class RAGIngestionState(StrEnum):
    UPLOADED = "UPLOADED"
    PARSING = "PARSING"
    VALIDATING = "VALIDATING"
    EMBEDDING = "EMBEDDING"
    INDEXED = "INDEXED"
    REJECTED_PSEUDOSCIENCE = "REJECTED_PSEUDOSCIENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


RAG_TRANSITIONS = {
    RAGIngestionState.UPLOADED: {RAGIngestionState.PARSING, RAGIngestionState.FAILED},
    RAGIngestionState.PARSING: {RAGIngestionState.VALIDATING, RAGIngestionState.FAILED},
    RAGIngestionState.VALIDATING: {
        RAGIngestionState.EMBEDDING,
        RAGIngestionState.REJECTED_PSEUDOSCIENCE,
        RAGIngestionState.REVIEW_REQUIRED,
        RAGIngestionState.FAILED,
    },
    RAGIngestionState.EMBEDDING: {RAGIngestionState.INDEXED, RAGIngestionState.FAILED},
    RAGIngestionState.INDEXED: {RAGIngestionState.EMBEDDING},
    RAGIngestionState.REJECTED_PSEUDOSCIENCE: set(),
    RAGIngestionState.REVIEW_REQUIRED: {RAGIngestionState.EMBEDDING, RAGIngestionState.REJECTED_PSEUDOSCIENCE},
    RAGIngestionState.FAILED: {RAGIngestionState.PARSING, RAGIngestionState.EMBEDDING},
}


def transition_rag(current: str, requested: RAGIngestionState) -> str:
    state = RAGIngestionState(current)
    if requested not in RAG_TRANSITIONS[state]:
        raise ValueError(f"Invalid RAG ingestion transition: {state.value} → {requested.value}")
    return requested.value
