"""Consistent online setup and offline runtime loading for local models."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache" / "models"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-small"


def cached_snapshot(model_name):
    """Resolve a Hugging Face model to a concrete local snapshot directory."""
    model_cache = CACHE_DIR / f"models--{model_name.replace('/', '--')}" / "snapshots"
    snapshots = sorted(
        (path for path in model_cache.glob("*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        raise RuntimeError(
            f"Model '{model_name}' is not cached. Run:\n"
            ".\\.venv\\Scripts\\python.exe prepare_models.py"
        )
    return snapshots[0]


def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(cached_snapshot(EMBEDDING_MODEL_NAME)), device="cpu")


def load_reranker_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(str(cached_snapshot(RERANKER_MODEL_NAME)), device="cpu")


def load_generator_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    snapshot = cached_snapshot(GENERATOR_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    generator = AutoModelForSeq2SeqLM.from_pretrained(snapshot, local_files_only=True)
    return tokenizer, generator
