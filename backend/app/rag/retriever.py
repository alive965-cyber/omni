"""Stratified 3-Tier Hybrid Retriever module (backend/app/rag/retriever.py).
Re-exports StratifiedBehavioralRetriever from knowledge_retrieval for full compatibility
with both `app.rag.retriever` and `app.services.knowledge_retrieval` import paths.
"""
from __future__ import annotations

from app.services.knowledge_retrieval import (
     _EMBEDDING_MODEL as EMBEDDING_MODEL,
    _collection_name,
    StratifiedBehavioralRetriever,
)

COLLECTION_NAME = _collection_name()

__all__ = ["COLLECTION_NAME", "EMBEDDING_MODEL", "StratifiedBehavioralRetriever"]
