"""SQLite Persistence Layer for OmniPulse MBS v1.0 + v2.0 Cognitive & Dyadic Telemetry (repository.py)."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import sqlite3
import tempfile
from threading import Lock
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
except ImportError:
    pass

_lock = Lock()


def _resolve_db_path() -> Path:
    db_url = os.getenv("DATABASE_URL", "sqlite:///./data/omnipulse.db")
    if not db_url.startswith("sqlite:///"):
        raise RuntimeError(
            "This API build uses SQLite for zero-config local execution. "
            "PostgreSQL/TimescaleDB migrations are provided in backend/migrations/versions/."
        )
    path = Path(db_url.replace("sqlite:///", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_db_path = _resolve_db_path()


@contextmanager
def connect():
    global _db_path
    try:
        conn = sqlite3.connect(_db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
    except sqlite3.OperationalError:
        # Fallback to /tmp if current working directory is on a FUSE/network mount that blocks fcntl locks
        _db_path = Path(tempfile.gettempdir()) / "omnipulse_fallback.db"
        conn = sqlite3.connect(_db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    global _db_path
    with _lock:
        try:
            with connect() as db:
                _create_schema(db)
        except sqlite3.OperationalError:
            _db_path = Path(tempfile.gettempdir()) / "omnipulse_fallback.db"
            with connect() as db:
                _create_schema(db)


def _create_schema(db: sqlite3.Connection):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      subject_identifier TEXT NOT NULL,
      session_title TEXT NOT NULL,
      status TEXT NOT NULL,
      consent_verified INTEGER NOT NULL,
      ambient_lux_score REAL,
      mic_snr_db REAL,
      calibration_duration_sec REAL NOT NULL DEFAULT 90,
      capture_token TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      completed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS telemetry_windows (
      session_id TEXT NOT NULL,
      window_index INTEGER NOT NULL,
      start_sec REAL NOT NULL,
      end_sec REAL NOT NULL,
      payload TEXT NOT NULL,
      PRIMARY KEY(session_id, window_index),
      FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS session_baselines (
      session_id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS session_report_narratives (
      session_id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS dyadic_turns (
      session_id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS discovered_personal_motifs (
      session_id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS session_events (
      session_id TEXT PRIMARY KEY,
      payload TEXT NOT NULL,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      source_title TEXT NOT NULL,
      file_name TEXT NOT NULL,
      modality_tags TEXT NOT NULL,
      evidence_tier TEXT NOT NULL,
      status TEXT NOT NULL,
      rejection_reason TEXT,
      chunk_count INTEGER NOT NULL DEFAULT 0,
      sha256_hash TEXT NOT NULL UNIQUE,
      dense_indexed INTEGER NOT NULL DEFAULT 0,
      dense_index_error TEXT,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS kb_parents (
      parent_id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL,
      section TEXT,
      content_text TEXT NOT NULL,
      FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS kb_chunks (
      chunk_id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL,
      parent_id TEXT NOT NULL,
      section TEXT,
      content_text TEXT NOT NULL,
      signal_triggers TEXT DEFAULT '[]',
      FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
      FOREIGN KEY(parent_id) REFERENCES kb_parents(parent_id) ON DELETE CASCADE
    );
    """)
    session_columns = {row["name"] for row in db.execute("PRAGMA table_info(sessions)").fetchall()}
    if "capture_token" not in session_columns:
        db.execute("ALTER TABLE sessions ADD COLUMN capture_token TEXT")
    document_columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)").fetchall()}
    if "dense_indexed" not in document_columns:
        db.execute("ALTER TABLE documents ADD COLUMN dense_indexed INTEGER NOT NULL DEFAULT 0")
    if "dense_index_error" not in document_columns:
        db.execute("ALTER TABLE documents ADD COLUMN dense_index_error TEXT")
    chunk_columns = {row["name"] for row in db.execute("PRAGMA table_info(kb_chunks)").fetchall()}
    if "signal_triggers" not in chunk_columns:
        db.execute("ALTER TABLE kb_chunks ADD COLUMN signal_triggers TEXT DEFAULT '[]'")


def row_dict(row):
    return dict(row) if row else None


def create_session(session_id: str, data: dict[str, Any]):
    capture_token = secrets.token_urlsafe(32)
    with _lock, connect() as db:
        db.execute(
            "INSERT INTO sessions(id,subject_identifier,session_title,status,consent_verified,ambient_lux_score,mic_snr_db,capture_token) VALUES(?,?,?,?,?,?,?,?)",
            (
                session_id,
                data["subject_identifier"],
                data["session_title"],
                "CREATED",
                int(data["consent_verified"]),
                data.get("ambient_lux_score"),
                data.get("mic_snr_db"),
                capture_token,
            ),
        )
        return row_dict(db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone())


def get_session(session_id: str):
    with connect() as db:
        return row_dict(
            db.execute(
                "SELECT id,subject_identifier,session_title,status,consent_verified,ambient_lux_score,mic_snr_db,calibration_duration_sec,created_at,completed_at FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        )


def get_capture_token(session_id: str):
    with connect() as db:
        row = db.execute("SELECT capture_token FROM sessions WHERE id=?", (session_id,)).fetchone()
        return row["capture_token"] if row else None


def list_sessions(limit: int = 50):
    with connect() as db:
        return [
            row_dict(row)
            for row in db.execute(
                "SELECT id,subject_identifier,session_title,status,consent_verified,ambient_lux_score,mic_snr_db,calibration_duration_sec,created_at,completed_at FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        ]


def update_session(session_id: str, **fields):
    if not fields:
        return get_session(session_id)
    columns = ",".join(f"{key}=?" for key in fields)
    with _lock, connect() as db:
        db.execute(f"UPDATE sessions SET {columns} WHERE id=?", (*fields.values(), session_id))
        return row_dict(
            db.execute(
                "SELECT id,subject_identifier,session_title,status,consent_verified,ambient_lux_score,mic_snr_db,calibration_duration_sec,created_at,completed_at FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        )


def put_window(session_id: str, window: dict[str, Any]):
    with _lock, connect() as db:
        existing = db.execute(
            "SELECT payload FROM telemetry_windows WHERE session_id=? AND window_index=?",
            (session_id, window["window_index"]),
        ).fetchone()
        if existing:
            previous = json.loads(existing["payload"])
            window["metrics"] = {**(previous.get("metrics") or {}), **(window.get("metrics") or {})}
            window["confidence"] = {**(previous.get("confidence") or {}), **(window.get("confidence") or {})}
            if not window.get("transcript_segment"):
                window["transcript_segment"] = previous.get("transcript_segment")
        db.execute(
            "INSERT OR REPLACE INTO telemetry_windows(session_id,window_index,start_sec,end_sec,payload) VALUES(?,?,?,?,?)",
            (session_id, window["window_index"], window["start_sec"], window["end_sec"], json.dumps(window)),
        )


def put_windows_bulk(session_id: str, windows: list[dict[str, Any]]):
    with _lock, connect() as db:
        for window in windows:
            db.execute(
                "INSERT OR REPLACE INTO telemetry_windows(session_id,window_index,start_sec,end_sec,payload) VALUES(?,?,?,?,?)",
                (session_id, window["window_index"], window["start_sec"], window["end_sec"], json.dumps(window)),
            )


def get_windows(session_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        return [
            json.loads(row["payload"])
            for row in db.execute(
                "SELECT payload FROM telemetry_windows WHERE session_id=? ORDER BY window_index",
                (session_id,),
            ).fetchall()
        ]


def put_baselines(session_id: str, payload: dict[str, Any]):
    with _lock, connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO session_baselines(session_id,payload) VALUES(?,?)",
            (session_id, json.dumps(payload, allow_nan=False)),
        )


def get_baselines(session_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT payload FROM session_baselines WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["payload"]) if row else None


def put_report_narrative(session_id: str, payload: dict[str, Any]):
    with _lock, connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO session_report_narratives(session_id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
            (session_id, json.dumps(payload, allow_nan=False)),
        )


def get_report_narrative(session_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT payload FROM session_report_narratives WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["payload"]) if row else None


def put_dyadic_turns(session_id: str, turns: list[dict[str, Any]]):
    with _lock, connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO dyadic_turns(session_id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
            (session_id, json.dumps(turns, allow_nan=False)),
        )


def get_dyadic_turns(session_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        row = db.execute("SELECT payload FROM dyadic_turns WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["payload"]) if row else []


def put_personal_motifs(session_id: str, motifs: list[dict[str, Any]]):
    with _lock, connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO discovered_personal_motifs(session_id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
            (session_id, json.dumps(motifs, allow_nan=False)),
        )


def get_personal_motifs(session_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        row = db.execute("SELECT payload FROM discovered_personal_motifs WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["payload"]) if row else []


def put_session_events(session_id: str, events: list[dict[str, Any]]):
    with _lock, connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO session_events(session_id,payload,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
            (session_id, json.dumps(events, allow_nan=False)),
        )


def get_session_events(session_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        row = db.execute("SELECT payload FROM session_events WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["payload"]) if row else []


def insert_document(document: dict[str, Any], chunks: list[dict[str, Any]]):
    with _lock, connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO documents(id,source_title,file_name,modality_tags,evidence_tier,status,rejection_reason,chunk_count,sha256_hash) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                document["id"],
                document["source_title"],
                document["file_name"],
                json.dumps(document["modality_tags"]),
                document["evidence_tier"],
                document["status"],
                document.get("rejection_reason"),
                len(chunks),
                document["sha256_hash"],
            ),
        )
        parents = {}
        for chunk in chunks:
            parents.setdefault(
                chunk["parent_id"],
                (chunk["document_id"], chunk.get("section", "Document"), chunk.get("parent_text", chunk["content_text"])),
            )
        db.executemany(
            "INSERT OR REPLACE INTO kb_parents(parent_id,document_id,section,content_text) VALUES(?,?,?,?)",
            [(parent_id, *values) for parent_id, values in parents.items()],
        )
        db.executemany(
            "INSERT OR REPLACE INTO kb_chunks(chunk_id,document_id,parent_id,section,content_text,signal_triggers) VALUES(?,?,?,?,?,?)",
            [
                (
                    c["chunk_id"],
                    c["document_id"],
                    c["parent_id"],
                    c.get("section", "Document"),
                    c["content_text"],
                    json.dumps(c.get("signal_triggers") or []),
                )
                for c in chunks
            ],
        )
    return get_document(document["id"])


def get_document(document_id: str):
    with connect() as db:
        row = row_dict(db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone())
        if row:
            row["modality_tags"] = json.loads(row["modality_tags"])
            row["dense_indexed"] = bool(row.get("dense_indexed"))
        return row


def list_documents():
    with connect() as db:
        rows = [row_dict(row) for row in db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()]
        for row in rows:
            row["modality_tags"] = json.loads(row["modality_tags"])
            row["dense_indexed"] = bool(row.get("dense_indexed"))
        return rows


def mark_document_dense_indexed(document_id: str):
    with _lock, connect() as db:
        db.execute("UPDATE documents SET dense_indexed=1,dense_index_error=NULL WHERE id=?", (document_id,))


def mark_document_dense_error(document_id: str, error: str):
    with _lock, connect() as db:
        db.execute("UPDATE documents SET dense_indexed=0,dense_index_error=? WHERE id=?", (error[:500], document_id))


def count_pending_dense_documents() -> int:
    with connect() as db:
        row = db.execute("SELECT COUNT(*) AS count FROM documents WHERE status='INDEXED' AND dense_indexed=0").fetchone()
        return int(row["count"])


def get_document_chunks(document_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT chunk_id,document_id,parent_id,section,content_text,signal_triggers FROM kb_chunks WHERE document_id=? ORDER BY chunk_id",
            (document_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = row_dict(row)
        try:
            item["signal_triggers"] = json.loads(item.get("signal_triggers") or "[]")
        except Exception:
            item["signal_triggers"] = []
        out.append(item)
    return out


def list_pending_dense_documents():
    with connect() as db:
        rows = db.execute("SELECT id FROM documents WHERE status='INDEXED' AND dense_indexed=0 ORDER BY created_at").fetchall()
    pending = []
    for row in rows:
        document_id = row["id"]
        document = get_document(document_id)
        pending.append({"document": document, "chunks": get_document_chunks(document_id)})
    return pending


def get_chunk_details(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    with connect() as db:
        rows = db.execute(
            f"SELECT c.chunk_id,c.document_id,c.parent_id,c.section,c.content_text,c.signal_triggers, "
            f"p.content_text AS parent_context,d.source_title,d.evidence_tier,d.modality_tags "
            f"FROM kb_chunks c JOIN kb_parents p ON p.parent_id=c.parent_id "
            f"JOIN documents d ON d.id=c.document_id WHERE c.chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    result = {}
    for row in rows:
        item = row_dict(row)
        try:
            item["signal_triggers"] = json.loads(item.get("signal_triggers") or "[]")
        except Exception:
            item["signal_triggers"] = []
        try:
            item["modality_tags"] = json.loads(item.get("modality_tags") or "[]")
        except Exception:
            item["modality_tags"] = []
        result[item["chunk_id"]] = item
    return result


def search_chunks(query: str, limit: int = 6):
    terms = {word.lower() for word in query.split() if len(word) > 1}
    with connect() as db:
        candidates = db.execute(
            "SELECT c.*,p.content_text AS parent_context,d.source_title,d.evidence_tier,d.modality_tags "
            "FROM kb_chunks c JOIN kb_parents p ON p.parent_id=c.parent_id "
            "JOIN documents d ON d.id=c.document_id WHERE d.status='INDEXED'"
        ).fetchall()
    scored = []
    for raw in candidates:
        row = row_dict(raw)
        try:
            row["signal_triggers"] = json.loads(row.get("signal_triggers") or "[]")
        except Exception:
            row["signal_triggers"] = []
        try:
            row["modality_tags"] = json.loads(row.get("modality_tags") or "[]")
        except Exception:
            row["modality_tags"] = []
        text = row["content_text"].lower()
        score = sum(text.count(term) for term in terms)
        if score:
            row["score"] = score / max(len(terms), 1)
            scored.append(row)
    return sorted(scored, key=lambda r: (-r["score"], r["chunk_id"]))[:limit]
