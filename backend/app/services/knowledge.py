"""RAG Document Validation, Layout-Aware Parsing, Anti-Pseudoscience Gate,
and Hierarchical Parent-Child Chunking (PDF Specification Section 5.2).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

ALLOWED_EVIDENCE_TIERS = frozenset({
    "TIER_1_CANONICAL_RULE",
    "TIER_2_COMBINATORIAL_MATRIX",
    "TIER_3_EMPIRICAL_STUDY",
    "PEER_REVIEWED_GOLD",
    "PEER_REVIEWED_EMPIRICAL",
    "UNVERIFIED",
})

ALLOWED_MODALITY_TAGS = frozenset({
    "FACS", "VISUAL", "OCULAR", "GAZE", "AUDIO", "ACOUSTIC", "PROSODY",
    "RPPG", "PHYSIOLOGY", "HEART_RATE", "AUTONOMIC", "RECOVERY",
    "KINESICS", "POSTURE", "GESTURES", "LINGUISTIC", "NLP", "DISCOURSE",
    "FUSION", "INCONGRUENCE", "CULTURAL", "L2_LANGUAGE", "NEURODIVERGENCE",
    "CALIBRATION", "TEMPORAL", "COMPOUND_EMOTION", "GENERAL",
})

# Banned deterministic pseudoscience claims (ignoring cautionary rules that explicitly say "never equate" or "do not")
BANNED_CLAIM_PATTERNS = (
    r"(?<!never\s)(?<!not\s)looking\s+(up|down|left|right)(?:[- ][a-z]+)?\s+(?:always|means|proves)[\s\S]{0,60}(lying|deception|visual\s+construction)",
    r"(?<!never\s)(?<!not\s)microexpression[s]?\s+(prove|detect|reveal)\s+(a\s+)?lie",
    r"(?<!never\s)(?<!not\s)diagnos(e|es|ed|ing)\s+(a\s+)?(psychiatric|mental)\s+(condition|illness)",
)


def parse_yaml_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Extract optional YAML frontmatter (`--- ... ---`) from Markdown files without requiring PyYAML."""
    if not raw_text.startswith("---"):
        return {}, raw_text
    end_idx = raw_text.find("\n---", 3)
    if end_idx < 0:
        return {}, raw_text
    fm_block = raw_text[3:end_idx].strip()
    body = raw_text[end_idx + 4 :].lstrip("\r\n")

    meta: dict[str, Any] = {}
    try:
        import yaml

        loaded = yaml.safe_load(fm_block)
        if isinstance(loaded, dict):
            return loaded, body
    except Exception:
        pass

    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = v.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [item.strip().strip("'\"") for item in val[1:-1].split(",") if item.strip()]
            meta[key] = items
        elif val.isdigit():
            meta[key] = int(val)
        else:
            meta[key] = val.strip("'\"")
    return meta, body


def validate_and_extract(filename: str, content: bytes) -> str:
    if not filename or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Document is missing a filename or exceeds the 50 MB upload limit.")
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".txt"}:
        return content.decode("utf-8", errors="strict")
    if suffix == ".json":
        text = content.decode("utf-8", errors="strict")
        json.loads(text)
        return text
    if suffix == ".pdf":
        try:
            import fitz

            doc = fitz.open(stream=content, filetype="pdf")
            return "\n\n".join(page.get_text() for page in doc)
        except ImportError:
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(content))
                return "\n\n".join((page.extract_text() or "") for page in reader.pages)
            except ImportError as exc:
                raise RuntimeError("PDF parsing is optional; install PyMuPDF or pypdf.") from exc
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX parsing is optional; install python-docx.") from exc
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError("Supported formats are .md, .txt, .json, .pdf, and .docx.")


def integrity_status(text: str) -> tuple[str, str | None]:
    """Step 3 (PDF Sec 5.2): Scientific Integrity & Anti-Pseudoscience Gate."""
    for line in text.splitlines():
        lower_line = line.lower()
        if any(guard in lower_line for guard in ("never equate", "do not equate", "anti-pseudoscience", "banned", "myth")):
            continue
        for pattern in BANNED_CLAIM_PATTERNS:
            if re.search(pattern, line, flags=re.IGNORECASE):
                return (
                    "REJECTED_PSEUDOSCIENCE",
                    "The document contains deterministic or unsupported behavioral claims and was held from retrieval.",
                )
    if len(text.strip()) < 80:
        return "REVIEW_REQUIRED", "This document is too short for automated evidence screening; review it before indexing."
    return "INDEXED", None


def chunk_document(
    document_id: str,
    text: str,
    child_words: int = 300,
    overlap_words: int = 50,
    parent_words: int = 1200,
) -> list[dict[str, Any]]:
    """Step 4 (PDF Sec 5.2): Hierarchical Parent-Child & Header-Aware Chunking.
    Parent Chunks: 1,200 tokens (full section context returned to LLM).
    Child Chunks: 300 tokens with 50-token overlap (embedded for search).
    """
    _, clean_body = parse_yaml_frontmatter(text)
    sections: list[tuple[str, str]] = []
    current_h1 = ""
    current_heading = "Document"
    current_lines: list[str] = []

    for line in clean_body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_h1 = stripped.lstrip("# ").strip()
            current_heading = current_h1 or "Document"
        elif stripped.startswith("##"):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            sub = stripped.lstrip("# ").strip()
            current_heading = f"{current_h1} -> {sub}" if current_h1 else (sub or "Document")
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    chunks: list[dict[str, Any]] = []
    for heading, section_text in sections:
        words = section_text.split()
        if not words:
            continue
        for parent_start in range(0, len(words), parent_words):
            parent_content = " ".join(words[parent_start : parent_start + parent_words])
            parent_id = f"parent_{hashlib.md5(f'{document_id}:{heading}:{parent_start}'.encode()).hexdigest()[:12]}"
            parent_tokens = parent_content.split()
            step = max(1, child_words - overlap_words)
            for child_start in range(0, len(parent_tokens), step):
                child = parent_tokens[child_start : child_start + child_words]
                if not child:
                    continue
                chunk_id = f"chunk_{hashlib.md5(f'{document_id}:{heading}:{parent_start}:{child_start}'.encode()).hexdigest()[:12]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "parent_id": parent_id,
                    "parent_text": f"{heading}\n{parent_content}",
                    "section": heading,
                    "section_header": heading,
                    "content_text": f"{heading}\n{' '.join(child)}",
                })
                if child_start + child_words >= len(parent_tokens):
                    break
            if parent_start + parent_words >= len(words):
                break
    return chunks


def prepare_document(
    filename: str,
    content: bytes,
    tags: list[str],
    evidence_tier: str = "UNVERIFIED",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_text = validate_and_extract(filename, content)
    fm_meta, body_text = parse_yaml_frontmatter(raw_text)
    status, reason = integrity_status(body_text or raw_text)
    digest = hashlib.sha256(content).hexdigest()
    doc_id = str(fm_meta.get("doc_id") or f"doc_{digest[:16]}")
    source_title = str(fm_meta.get("source_title") or Path(filename).stem.replace("_", " "))
    merged_tags = sorted(set((tags or []) + [str(t).upper() for t in (fm_meta.get("modality_tags") or []) if str(t).strip()])) or ["GENERAL"]

    resolved_tier = evidence_tier
    if resolved_tier == "UNVERIFIED" and fm_meta.get("evidence_tier") in ALLOWED_EVIDENCE_TIERS:
        resolved_tier = str(fm_meta["evidence_tier"])

    chunks = chunk_document(doc_id, raw_text) if status == "INDEXED" else []
    document = {
        "id": doc_id,
        "file_name": filename,
        "source_title": source_title,
        "authors": fm_meta.get("authors") or ["OmniPulse Scientific Corpus"],
        "publication_year": int(fm_meta["publication_year"]) if str(fm_meta.get("publication_year", "")).isdigit() else 2026,
        "modality_tags": merged_tags,
        "evidence_tier": resolved_tier,
        "status": status,
        "sha256_hash": digest,
        "rejection_reason": reason,
    }
    return document, chunks
