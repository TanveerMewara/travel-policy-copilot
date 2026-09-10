"""Lesson 2: compare keyword retrieval with local embedding-based retrieval."""

import argparse

from model_runtime import EMBEDDING_MODEL_NAME, load_embedding_model
from search import load_chunks, retrieve

MODEL_NAME = EMBEDDING_MODEL_NAME


def embed_chunks(model, chunks):
    """Convert each passage, including its policy identity, into a vector."""
    texts = [
        f"Supplier: {c['metadata'].get('supplier', '')}\n"
        f"Rate: {c['metadata'].get('rate', '')}\n"
        f"Effective from: {c['metadata'].get('effective_from', '')}\n"
        f"Effective to: {c['metadata'].get('effective_to', '')}\n"
        f"{c['title']}\n{c['section']}\n{c['text']}"
        for c in chunks
    ]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def semantic_retrieve(question, chunks, model, chunk_vectors, top_k=3):
    """Compare the question vector with every passage vector."""
    if not chunks:
        return []
    question_vector = model.encode(question, normalize_embeddings=True)
    # For unit-length (normalized) vectors, dot product is cosine similarity.
    scores = chunk_vectors @ question_vector
    ranked_indices = scores.argsort()[::-1][:top_k]
    return [{**chunks[int(index)], "score": float(scores[index])} for index in ranked_indices]


def print_results(label, results):
    print(f"\n{label}")
    if not results:
        print("  No keyword matches found.")
    for position, result in enumerate(results, 1):
        print(f"{position}. {result['title']} / {result['section']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Source: {result['source']}")
        print(f"   Passage: {result['text']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to search for")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("Your question: ")
    if not question.strip():
        parser.error("Please enter a non-empty question.")
    chunks = load_chunks()
    if not chunks:
        parser.error("No policy passages found in data/policies.")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Semantic-search dependencies are missing or could not be imported.\n"
            "Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
            f"Import detail: {error}"
        ) from error

    print(f"Loading {MODEL_NAME} on CPU. The first run downloads model files.", flush=True)
    model = load_embedding_model()
    chunk_vectors = embed_chunks(model, chunks)
    print(f"Embedded {len(chunks)} fictional policy passages.")
    print(f"Vector array shape: {chunk_vectors.shape} (passages, numbers per passage)")

    print_results("KEYWORD SEARCH - score counts matching words", retrieve(question, chunks))
    print_results(
        "SEMANTIC SEARCH - score measures cosine similarity",
        semantic_retrieve(question, chunks, model, chunk_vectors),
    )
    print("\nThe two score scales are different; neither is an answer-confidence percentage.")
    print("Semantic search returns nearest passages even when none answers the question.")
    print("These are retrieved passages, not a generated answer.")


if __name__ == "__main__":
    main()
