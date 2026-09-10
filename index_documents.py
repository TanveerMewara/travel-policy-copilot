"""Lesson 6: build and load a persistent local vector index."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from model_runtime import load_embedding_model
from search import POLICY_DIR, load_chunks
from semantic_search import MODEL_NAME, embed_chunks

INDEX_DIR = Path(__file__).resolve().parent / "data" / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
VECTORS_FILE = INDEX_DIR / "vectors.npy"
MANIFEST_FILE = INDEX_DIR / "manifest.json"


def source_fingerprint():
    """Hash source paths, contents, and model name to detect a stale index."""
    digest = hashlib.sha256(MODEL_NAME.encode("utf-8"))
    source_paths = list(POLICY_DIR.glob("*.md"))
    source_paths += list(POLICY_DIR.glob("*.pdf"))
    source_paths += list(POLICY_DIR.glob("*.metadata.json"))
    for path in sorted(source_paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_index(model=None):
    """Embed all chunks and persist vectors, chunk data, and a manifest."""
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No policy passages found in data/policies.")

    print(f"Loading embedding model: {MODEL_NAME}", flush=True)
    if model is None:
        model = load_embedding_model()
    vectors = embed_chunks(model, chunks).astype("float32")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_FILE, vectors, allow_pickle=False)
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "format_version": 1,
        "embedding_model": MODEL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_fingerprint": source_fingerprint(),
        "chunk_count": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_index():
    """Load a compatible, current index or explain how to rebuild it."""
    required = (CHUNKS_FILE, VECTORS_FILE, MANIFEST_FILE)
    if not all(path.exists() for path in required):
        raise SystemExit(
            "The vector index is missing. Run:\n.\\.venv\\Scripts\\python.exe index_documents.py"
        )

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    vectors = np.load(VECTORS_FILE, allow_pickle=False)

    if manifest.get("embedding_model") != MODEL_NAME:
        raise SystemExit("The embedding model changed. Rebuild the vector index.")
    if manifest.get("source_fingerprint") != source_fingerprint():
        raise SystemExit("Policy files changed. Rebuild the vector index.")
    if len(chunks) != len(vectors) or len(chunks) != manifest.get("chunk_count"):
        raise SystemExit("Index files disagree about chunk count. Rebuild the index.")
    if vectors.ndim != 2 or vectors.shape[1] != manifest.get("vector_dimension"):
        raise SystemExit("Stored vector dimensions are invalid. Rebuild the index.")
    return chunks, vectors, manifest


def main():
    manifest = build_index()
    print(
        f"Indexed {manifest['chunk_count']} chunks as "
        f"{manifest['vector_dimension']}-number vectors."
    )
    print(f"Index directory: {INDEX_DIR}")
    print(f"Source fingerprint: {manifest['source_fingerprint'][:12]}...")


if __name__ == "__main__":
    main()
