"""Privacy-conscious local JSONL telemetry for pipeline diagnostics."""

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "logs" / "rag_events.jsonl"
_LOCK = threading.Lock()


def record_query(question, response, supplier="All", rate="All"):
    """Append operational metadata without storing raw question text."""
    results = response.get("results", [])
    top = results[0] if results else {}
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": "rag_query",
        "query_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "query_length": len(question),
        "status": "abstained" if response.get("abstained") else "answered",
        "supplier_filter": supplier,
        "rate_filter": rate,
        "top_chunk_id": top.get("chunk_id"),
        "semantic_similarity": top.get("semantic_score"),
        "reranker_score": top.get("reranker_score"),
        "latency_seconds": round(response.get("elapsed", 0.0), 4),
    }
    try:
        with _LOCK:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        return False
    return True
