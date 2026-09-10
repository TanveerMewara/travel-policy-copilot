"""Lesson 9: rerank retrieved candidates with a query-passage cross-encoder."""

import argparse
import sys

from hybrid_search import hybrid_retrieve
from index_documents import load_index
from model_runtime import RERANKER_MODEL_NAME, load_embedding_model, load_reranker_model

RERANKER_NAME = RERANKER_MODEL_NAME


def passage_text(chunk):
    metadata = chunk["metadata"]
    return (
        f"Supplier: {metadata.get('supplier', '')}\n"
        f"Rate: {metadata.get('rate', '')}\n"
        f"Effective from: {metadata.get('effective_from', '')}\n"
        f"Effective to: {metadata.get('effective_to', '')}\n"
        f"Policy: {chunk['title']}\n"
        f"Section: {chunk['section']}\n"
        f"{chunk['text']}"
    )


def rerank(question, candidates, model, top_k=3):
    """Score each question-passage pair jointly and reorder the candidates."""
    if not candidates:
        return []
    pairs = [(question, passage_text(candidate)) for candidate in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    rescored = [
        {**candidate, "retrieval_rank": rank, "reranker_score": float(score)}
        for rank, (candidate, score) in enumerate(zip(candidates, scores), 1)
    ]
    return sorted(rescored, key=lambda candidate: candidate["reranker_score"], reverse=True)[:top_k]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to search for")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("Your question: ")
    if not question.strip():
        parser.error("Please enter a non-empty question.")

    chunks, vectors, _manifest = load_index()
    embedding_model = load_embedding_model()
    candidates = hybrid_retrieve(
        question, chunks, embedding_model, vectors, top_k=min(10, len(chunks))
    )
    reranker_model = load_reranker_model()
    results = rerank(question, candidates, reranker_model, top_k=3)

    print(f"Retrieved {len(candidates)} candidates; showing 3 reranked results.")
    for position, result in enumerate(results, 1):
        print(f"\n{position}. {result['title']} / {result['section']}")
        print(f"   Before reranking: position {result['retrieval_rank']}")
        print(f"   Reranker score: {result['reranker_score']:.3f} (not a probability)")
        print(f"   Source: {result['source']}")
        print(f"   Passage: {result['text']}")


if __name__ == "__main__":
    main()
