"""Download all public models required by the offline application runtime."""

from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model_runtime import CACHE_DIR, EMBEDDING_MODEL_NAME, GENERATOR_MODEL_NAME, RERANKER_MODEL_NAME


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}", flush=True)
    SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=str(CACHE_DIR))
    print(f"Downloading reranker: {RERANKER_MODEL_NAME}", flush=True)
    CrossEncoder(RERANKER_MODEL_NAME, cache_folder=str(CACHE_DIR))
    print(f"Downloading generator: {GENERATOR_MODEL_NAME}", flush=True)
    AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME, cache_dir=str(CACHE_DIR))
    AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME, cache_dir=str(CACHE_DIR))
    print("All models are ready for offline use.")


if __name__ == "__main__":
    main()
