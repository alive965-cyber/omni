"use client";

import { DragEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Badge, Icon, PageHeading } from "@/components/shell";

type DocumentRow = {
  id: string;
  source_title: string;
  file_name: string;
  modality_tags: string[];
  evidence_tier: string;
  status: string;
  rejection_reason?: string | null;
  dense_indexed: boolean;
  dense_index_error?: string | null;
  chunk_count: number;
  created_at: string;
};

type KnowledgeStatus = {
  dense_ready: boolean;
  qdrant_sdk_installed: boolean;
  qdrant_reachable: boolean;
  qdrant_collection: string;
  qdrant_collection_ready: boolean;
  fastembed_installed: boolean;
  embedding_model: string;
  embedding_dimension: number;
  reranker_configured: boolean;
  reranker_model: string;
  pending_documents: number;
  setup_hint?: string | null;
};

type SearchResult = {
  chunk_id: string;
  document_id: string;
  doc_id?: string;
  source_title: string;
  evidence_tier: string;
  content_text: string;
  parent_context: string;
  retrieval_score: number;
  dense_score?: number;
  lexical_score?: number;
  matched_triggers?: string[];
};

type StratifiedChunk = {
  doc_id: string;
  evidence_tier: string;
  source_title: string;
  authors?: string[];
  publication_year?: number;
  section_header?: string;
  matched_triggers?: string[];
  hybrid_relevance_score?: number;
  content_text: string;
};

type StratifiedBundle = {
  tier_1_canonical_rule?: StratifiedChunk | null;
  tier_2_combinatorial_matrix?: StratifiedChunk | null;
  tier_3_empirical_studies?: StratifiedChunk[];
};

const MODALITY_TAGS = [
  "FACS",
  "VISUAL",
  "OCULAR",
  "GAZE",
  "AUDIO",
  "ACOUSTIC",
  "PROSODY",
  "RPPG",
  "PHYSIOLOGY",
  "KINESICS",
  "POSTURE",
  "LINGUISTIC",
  "DISCOURSE",
  "FUSION",
  "INCONGRUENCE",
  "CULTURAL",
  "L2_LANGUAGE",
  "NEURODIVERGENCE",
  "TEMPORAL",
  "GENERAL",
];

const EVIDENCE_TIERS = [
  { value: "TIER_1_CANONICAL_RULE", label: "Tier 1 · Canonical Manual / Gold Standard" },
  { value: "TIER_2_COMBINATORIAL_MATRIX", label: "Tier 2 · Combinatorial Multimodal Matrix" },
  { value: "TIER_3_EMPIRICAL_STUDY", label: "Tier 3 · Peer-Reviewed Empirical Study" },
  { value: "PEER_REVIEWED_GOLD", label: "Peer-Reviewed Gold Reference" },
  { value: "PEER_REVIEWED_EMPIRICAL", label: "Peer-Reviewed Empirical" },
  { value: "UNVERIFIED", label: "Unverified / Pending Human Review" },
];

const SIGNAL_TRIGGERS = [
  "au24_lip_presser",
  "au04_brow_lowerer",
  "au12_lip_corner_puller",
  "au06_cheek_raiser",
  "facial_asymmetry_index",
  "blink_rate_per_min",
  "gaze_aversion_ratio",
  "jitter_local",
  "shimmer_local_db",
  "hnr_db",
  "f0_semitone_mean",
  "pause_ratio",
  "disfluency_count",
  "hr_bpm_pos",
  "self_touch_ratio",
  "hand_kinetic_energy",
  "first_person_pronoun_ratio",
  "cross_modal_congruency_index",
];

const PIPELINE_STAGES = ["UPLOADED", "PARSING", "VALIDATING", "EMBEDDING", "INDEXED"];

export default function KnowledgeBasePage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>(["FACS", "AUDIO", "RPPG"]);
  const [evidenceTier, setEvidenceTier] = useState<string>("TIER_1_CANONICAL_RULE");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [ingestionStage, setIngestionStage] = useState<string>("");
  const [activeDocId, setActiveDocId] = useState<string>("");

  // Sandbox state
  const [query, setQuery] = useState(
    "Tight lip press AU24 spike with vocal jitter elevation and sympathetic heart rate acceleration during positive statement"
  );
  const [sandboxTriggers, setSandboxTriggers] = useState<string[]>([
    "au24_lip_presser",
    "jitter_local",
    "hr_bpm_pos",
  ]);
  const [sandboxModalities, setSandboxModalities] = useState<string[]>(["FACS", "AUDIO", "RPPG"]);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [stratifiedBundle, setStratifiedBundle] = useState<StratifiedBundle | null>(null);
  const [allowedIds, setAllowedIds] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [apiReady, setApiReady] = useState(false);
  const [kbStatus, setKbStatus] = useState<KnowledgeStatus | null>(null);

  const loadDocuments = useCallback(async () => {
    try {
      const [rows, status] = await Promise.all([
        api.get<DocumentRow[]>("/kb/documents"),
        api.get<KnowledgeStatus>("/kb/status"),
      ]);
      setDocuments(rows);
      setKbStatus(status);
      setApiReady(true);
    } catch {
      setApiReady(false);
      setKbStatus(null);
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
  }, [loadDocuments]);

  useEffect(() => {
    if (!apiReady) return;
    const shouldRefresh = !kbStatus || !kbStatus.dense_ready || kbStatus.pending_documents > 0;
    if (!shouldRefresh) return;
    const timer = window.setInterval(() => {
      void loadDocuments();
    }, kbStatus?.dense_ready ? 4000 : 10000);
    return () => window.clearInterval(timer);
  }, [apiReady, kbStatus?.dense_ready, kbStatus?.pending_documents, loadDocuments]);

  function chooseFile(incoming: File | null) {
    setError("");
    if (incoming) setFile(incoming);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    chooseFile(e.dataTransfer.files.item(0));
  }

  function streamIngestionEvents(docId: string) {
    setActiveDocId(docId);
    setIngestionStage("UPLOADED");
    try {
      const es = new EventSource(api.url(`/kb/documents/${docId}/events`));
      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.ingestion_status) {
            setIngestionStage(data.ingestion_status);
            if (
              ["INDEXED", "REJECTED_PSEUDOSCIENCE", "REVIEW_REQUIRED", "FAILED"].includes(data.ingestion_status)
            ) {
              es.close();
            }
          }
        } catch {
          es.close();
        }
      };
      es.onerror = () => {
        es.close();
      };
    } catch {
      // Fallback if EventSource is blocked
    }
  }

  async function submitUpload(e: FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a document (.md, .txt, .json, .pdf, .docx) to upload.");
      return;
    }
    if (!selectedTags.length) {
      setError("Choose at least one modality tag.");
      return;
    }
    setUploading(true);
    setError("");
    setNotice("");
    setIngestionStage("UPLOADED");

    const form = new FormData();
    form.append("file", file);
    form.append("modality_tags", JSON.stringify(selectedTags));
    form.append("evidence_tier", evidenceTier);

    try {
      const result = await api.upload<DocumentRow & { dense_indexed?: boolean; indexing_notice?: string }>(
        "/kb/documents",
        form
      );
      if (result.id) {
        streamIngestionEvents(result.id);
      }
      setNotice(
        result.indexing_notice ||
          `${result.status.replaceAll("_", " ")} · ${result.chunk_count} parent-child chunks · ${
            result.dense_indexed ? "Qdrant 384-d vectors indexed" : "Stored in SQLite (lexical + built-in fallback active)"
          }`
      );
      setFile(null);
      if (fileRef.current) fileRef.current.value = "";
      await loadDocuments();
    } catch (err) {
      setIngestionStage("FAILED");
      setError(err instanceof Error ? err.message : "Document upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function runQuery(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setError("");
    setNotice("");
    setResults([]);
    setStratifiedBundle(null);
    try {
      const response = await api.post<{
        results: SearchResult[];
        stratified_4_chunk_bundle?: StratifiedBundle;
        allowed_citation_doc_ids?: string[];
        notice: string;
      }>("/kb/query-sandbox", {
        query,
        signal_triggers: sandboxTriggers,
        modality_tags: sandboxModalities,
      });
      setResults(response.results || []);
      setStratifiedBundle(response.stratified_4_chunk_bundle || null);
      setAllowedIds(response.allowed_citation_doc_ids || []);
      setNotice(response.notice);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retrieval request failed.");
    }
  }

  const tier1Count = documents.filter((d) => d.evidence_tier === "TIER_1_CANONICAL_RULE").length;
  const tier2Count = documents.filter((d) => d.evidence_tier === "TIER_2_COMBINATORIAL_MATRIX").length;
  const tier3Count = documents.filter((d) => d.evidence_tier === "TIER_3_EMPIRICAL_STUDY").length;

  return (
    <>
      <PageHeading
        eyebrow="3-TIER STRATIFIED SCIENTIFIC RAG · BAAI/BGE-SMALL-EN-V1.5 (384-DIM)"
        title="Scientific Knowledge Base & Retrieval Sandbox"
        description="Ingest behavioral coding manuals, combinatorial decision matrices, and peer-reviewed empirical studies with automated Anti-Pseudoscience screening and Parent-Child (1,200 / 300-word) chunking."
        action={
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
            <Badge tone={apiReady ? "green" : "amber"}>
              {apiReady ? `${documents.length} DOCUMENTS` : "API NOT CONNECTED"}
            </Badge>
            <Badge tone={kbStatus?.dense_ready ? "green" : "blue"}>
              {kbStatus?.dense_ready ? "FASTEMBED + QDRANT READY" : "HYBRID + BUILT-IN CORPUS ACTIVE"}
            </Badge>
          </div>
        }
      />

      <div className="sample-banner">
        <Icon name="shield" size={14} />
        <span>
          <b>Anti-Pseudoscience Gate Active:</b> Deterministic lie-detection claims (e.g. &ldquo;looking up-right proves
          lying&rdquo; or &ldquo;micro-expressions prove a lie&rdquo;) are automatically quarantined as{" "}
          <code>REJECTED_PSEUDOSCIENCE</code> and excluded from retrieval.
        </span>
      </div>

      <div className="grid kb-top-grid">
        {/* Left Card: Document Uploader + SSE Stepper */}
        <section className="card">
          <div className="card-heading">
            <div>
              <h2 className="card-title">Ingest scientific document</h2>
              <div className="card-subtitle">Markdown (with YAML frontmatter), TXT, JSON, PDF, or DOCX · up to 50 MB</div>
            </div>
            <Badge tone="blue">1,200 / 300-WORD CHUNKER</Badge>
          </div>

          <form onSubmit={(e) => void submitUpload(e)}>
            <div
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onClick={() => fileRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") fileRef.current?.click();
              }}
            >
              <div className="drop-icon">
                <Icon name="upload" size={17} />
              </div>
              <h3>{file ? file.name : "Drop a scientific manual or paper here"}</h3>
              <p>{file ? `${(file.size / 1024).toFixed(1)} KB · ready for validation` : "or click to browse your device"}</p>
              <span className="file-types">.MD · .TXT · .JSON · .PDF · .DOCX</span>
              <input
                className="hidden-input"
                ref={fileRef}
                type="file"
                accept=".md,.txt,.json,.pdf,.docx"
                onChange={(e) => chooseFile(e.target.files?.item(0) || null)}
              />
            </div>

            <div className="input-wrap" style={{ marginTop: 14 }}>
              <label htmlFor="evidence-tier">Evidence tier classification</label>
              <select
                id="evidence-tier"
                className="select"
                value={evidenceTier}
                onChange={(e) => setEvidenceTier(e.target.value)}
              >
                {EVIDENCE_TIERS.map((tier) => (
                  <option key={tier.value} value={tier.value}>
                    {tier.label}
                  </option>
                ))}
              </select>
            </div>

            <div style={{ marginTop: 14 }}>
              <div className="card-subtitle" style={{ marginBottom: 8 }}>
                MODALITY TAGS (SELECT ALL THAT APPLY)
              </div>
              <div className="tag-row">
                {MODALITY_TAGS.map((tag) => (
                  <button
                    type="button"
                    className={`tag-choice ${selectedTags.includes(tag) ? "selected" : ""}`}
                    onClick={() =>
                      setSelectedTags((prev) =>
                        prev.includes(tag) ? prev.filter((x) => x !== tag) : [...prev, tag]
                      )
                    }
                    key={tag}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>

            {/* Live SSE Ingestion Pipeline Stepper (PDF Section 9.4) */}
            {ingestionStage && (
              <div className="notice-block" style={{ marginTop: 14 }}>
                <div className="micro-label" style={{ marginBottom: 8 }}>
                  LIVE SSE INGESTION PIPELINE {activeDocId ? `· ${activeDocId}` : ""}
                </div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                  {PIPELINE_STAGES.map((stage, idx) => {
                    const currentIdx = PIPELINE_STAGES.indexOf(ingestionStage);
                    const done = currentIdx >= idx || ingestionStage === "INDEXED";
                    const rejected = ingestionStage === "REJECTED_PSEUDOSCIENCE";
                    return (
                      <Badge
                        key={stage}
                        tone={rejected && stage === "VALIDATING" ? "rose" : done ? "green" : "neutral"}
                      >
                        {idx + 1}. {stage}
                      </Badge>
                    );
                  })}
                  {ingestionStage === "REJECTED_PSEUDOSCIENCE" && (
                    <Badge tone="rose">REJECTED PSEUDOSCIENCE</Badge>
                  )}
                </div>
              </div>
            )}

            <div className="notice-block">
              Passages are embedded locally with <b>{kbStatus?.embedding_model || "BAAI/bge-small-en-v1.5"}</b> (384-d)
              and indexed with automatic <code>signal_triggers</code> extraction.
            </div>

            {error && <div className="form-error">{error}</div>}
            {notice && <div className="form-success">{notice}</div>}

            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 14 }}>
              <button className="button button-primary" type="submit" disabled={uploading || !apiReady}>
                {uploading ? "Validating & indexing…" : "Upload & index document"}
                <Icon name="arrow" size={14} />
              </button>
            </div>
          </form>
        </section>

        {/* Right Card: Stratified 3-Tier Retrieval Sandbox */}
        <section className="card">
          <div className="card-heading">
            <div>
              <h2 className="card-title">Stratified 3-Tier Retrieval Sandbox</h2>
              <div className="card-subtitle">
                70% Cosine Similarity + 30% Exact Signal-Trigger Overlap + Stratified 4-Chunk Quota
              </div>
            </div>
            <Badge tone="green">1× T1 + 1× T2 + 2× T3</Badge>
          </div>

          <form onSubmit={(e) => void runQuery(e)}>
            <div className="input-wrap">
              <label htmlFor="query">Behavioral anomaly description / query</label>
              <textarea
                id="query"
                className="textarea"
                placeholder="e.g. AU24 lip press spike with elevated vocal jitter and sympathetic HR rise"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>

            <div style={{ marginTop: 10 }}>
              <div className="card-subtitle" style={{ marginBottom: 6 }}>
                ACTIVE SIGNAL TRIGGERS (30% HYBRID BOOST)
              </div>
              <div className="tag-row">
                {SIGNAL_TRIGGERS.map((trig) => (
                  <button
                    type="button"
                    key={trig}
                    className={`tag-choice ${sandboxTriggers.includes(trig) ? "selected" : ""}`}
                    onClick={() =>
                      setSandboxTriggers((prev) =>
                        prev.includes(trig) ? prev.filter((x) => x !== trig) : [...prev, trig]
                      )
                    }
                  >
                    {trig}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ marginTop: 10 }}>
              <div className="card-subtitle" style={{ marginBottom: 6 }}>
                MODALITY FILTER
              </div>
              <div className="tag-row">
                {["FACS", "AUDIO", "RPPG", "KINESICS", "LINGUISTIC", "FUSION", "CULTURAL"].map((mod) => (
                  <button
                    type="button"
                    key={mod}
                    className={`tag-choice ${sandboxModalities.includes(mod) ? "selected" : ""}`}
                    onClick={() =>
                      setSandboxModalities((prev) =>
                        prev.includes(mod) ? prev.filter((x) => x !== mod) : [...prev, mod]
                      )
                    }
                  >
                    {mod}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 12 }}>
              <span className="subtle-note">
                Returns guaranteed 4-chunk stratified bundle + hybrid RRF candidates
              </span>
              <button className="button button-primary" type="submit" disabled={!apiReady || !query.trim()}>
                Run Stratified Retrieval <Icon name="arrow" size={13} />
              </button>
            </div>
          </form>

          {stratifiedBundle && (
            <div style={{ marginTop: 14 }}>
              <div className="micro-label" style={{ marginBottom: 6 }}>
                STRATIFIED 4-CHUNK ANOMALY BUNDLE · ALLOWED CITATIONS: {allowedIds.join(", ")}
              </div>
              {stratifiedBundle.tier_1_canonical_rule && (
                <div className="sandbox-result" style={{ borderLeft: "3px solid #60d8c3" }}>
                  <div className="sandbox-result-head">
                    <b>[TIER 1 · CANONICAL RULE] {stratifiedBundle.tier_1_canonical_rule.source_title}</b>
                    <span className="score-chip">
                      SCORE {stratifiedBundle.tier_1_canonical_rule.hybrid_relevance_score?.toFixed(2) ?? "0.90"}
                    </span>
                  </div>
                  <p>{stratifiedBundle.tier_1_canonical_rule.content_text}</p>
                  <div className="card-subtitle" style={{ marginTop: 6 }}>
                    Doc ID: <code>{stratifiedBundle.tier_1_canonical_rule.doc_id}</code> · Matched:{" "}
                    {(stratifiedBundle.tier_1_canonical_rule.matched_triggers || []).join(", ") || "general"}
                  </div>
                </div>
              )}

              {stratifiedBundle.tier_2_combinatorial_matrix && (
                <div className="sandbox-result" style={{ borderLeft: "3px solid #e4b45a" }}>
                  <div className="sandbox-result-head">
                    <b>[TIER 2 · 648-MATRIX] {stratifiedBundle.tier_2_combinatorial_matrix.source_title}</b>
                    <span className="score-chip">
                      SCORE {stratifiedBundle.tier_2_combinatorial_matrix.hybrid_relevance_score?.toFixed(2) ?? "0.88"}
                    </span>
                  </div>
                  <p style={{ whiteSpace: "pre-line" }}>{stratifiedBundle.tier_2_combinatorial_matrix.content_text}</p>
                  <div className="card-subtitle" style={{ marginTop: 6 }}>
                    Doc ID: <code>{stratifiedBundle.tier_2_combinatorial_matrix.doc_id}</code>
                  </div>
                </div>
              )}

              {(stratifiedBundle.tier_3_empirical_studies || []).map((study, idx) => (
                <div className="sandbox-result" style={{ borderLeft: "3px solid #65a9ff" }} key={study.doc_id + idx}>
                  <div className="sandbox-result-head">
                    <b>
                      [TIER 3 · EMPIRICAL STUDY #{idx + 1}] {study.source_title}
                    </b>
                    <span className="score-chip">SCORE {study.hybrid_relevance_score?.toFixed(2) ?? "0.85"}</span>
                  </div>
                  <p>{study.content_text}</p>
                  <div className="card-subtitle" style={{ marginTop: 6 }}>
                    Doc ID: <code>{study.doc_id}</code> · Year: {study.publication_year ?? 2024}
                  </div>
                </div>
              ))}
            </div>
          )}

          {results.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <div className="micro-label" style={{ marginBottom: 6 }}>
                HYBRID RRF CANDIDATE PASSAGES ({results.length})
              </div>
              {results.map((r) => (
                <div className="sandbox-result" key={r.chunk_id}>
                  <div className="sandbox-result-head">
                    <b>{r.source_title}</b>
                    <span className="score-chip">HYBRID {r.retrieval_score.toFixed(2)}</span>
                  </div>
                  <p>{r.content_text}</p>
                  <details style={{ fontSize: 9, color: "#8390a1", lineHeight: 1.6, marginTop: 6 }}>
                    <summary style={{ cursor: "pointer" }}>Show 1,200-word parent section context</summary>
                    <p>{r.parent_context}</p>
                  </details>
                  <div className="card-subtitle" style={{ marginTop: 8 }}>
                    {r.document_id} · {r.evidence_tier} · {r.chunk_id}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      <div className="section-heading">
        <h2>Indexed Scientific Corpus</h2>
        <span className="micro-label">
          {apiReady
            ? `${documents.length} DOCUMENTS · T1: ${tier1Count} · T2: ${tier2Count} · T3: ${tier3Count}`
            : "WAITING FOR API"}
        </span>
      </div>

      <section className="card">
        {documents.length ? (
          documents.map((doc) => (
            <div className="kb-doc-row" key={doc.id}>
              <div className="doc-icon">{doc.file_name.split(".").pop()?.toUpperCase() || "MD"}</div>
              <div className="doc-meta">
                <b>{doc.source_title}</b>
                <small>
                  <code>{doc.id}</code> · {doc.evidence_tier} · {doc.modality_tags.join(" · ") || "GENERAL"} ·{" "}
                  {doc.chunk_count} chunks
                  {doc.rejection_reason ? ` · ${doc.rejection_reason}` : ""}
                </small>
              </div>
              <Badge
                tone={
                  doc.status === "INDEXED"
                    ? "green"
                    : doc.status.includes("REJECTED")
                      ? "rose"
                      : "amber"
                }
              >
                {doc.status === "INDEXED"
                  ? doc.dense_indexed
                    ? "QDRANT + SQLITE INDEXED"
                    : "SQLITE INDEXED"
                  : doc.status.replaceAll("_", " ")}
              </Badge>
            </div>
          ))
        ) : (
          <div className="empty-state">
            {apiReady
              ? "No custom documents uploaded yet. Built-in 3-Tier Canonical Manuals, Combinatorial Matrix, and Empirical Studies are active."
              : "Connect to the FastAPI backend to view the knowledge corpus."}
          </div>
        )}
      </section>
    </>
  );
}
