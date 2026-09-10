"""Lesson 5: combine semantic and keyword rankings with reciprocal rank fusion."""

import argparse

from model_runtime import load_embedding_model
from search import load_chunks, retrieve
from semantic_search import embed_chunks, semantic_retrieve


def chunk_key(chunk):
    """Identify one passage consistently across retrieval methods."""
    return chunk["source"], chunk["section"]


def reciprocal_rank_fusion(semantic_results, keyword_results, top_k=3, rrf_k=60):
    """Fuse rankings without comparing their incompatible raw scores."""
    by_key = {chunk_key(result): dict(result) for result in semantic_results}
    fused_scores = {key: 0.0 for key in by_key}
    semantic_ranks = {}
    keyword_ranks = {}

    for rank, result in enumerate(semantic_results, 1):
        key = chunk_key(result)
        semantic_ranks[key] = rank
        fused_scores[key] += 1 / (rrf_k + rank)
        by_key[key]["semantic_score"] = result["score"]

    for rank, result in enumerate(keyword_results, 1):
        key = chunk_key(result)
        keyword_ranks[key] = rank
        if key not in by_key:
            by_key[key] = dict(result)
            fused_scores[key] = 0.0
        fused_scores[key] += 1 / (rrf_k + rank)
        by_key[key]["keyword_score"] = result["score"]

    ordered_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    return [
        {
            **by_key[key],
            "score": fused_scores[key],
            "semantic_rank": semantic_ranks.get(key),
            "keyword_rank": keyword_ranks.get(key),
            "semantic_score": by_key[key].get("semantic_score", 0.0),
            "keyword_score": by_key[key].get("keyword_score", 0),
        }
        for key in ordered_keys
    ]


def hybrid_retrieve(question, chunks, model, chunk_vectors, top_k=3):
    """Retrieve with both methods, then fuse their ranks."""
    semantic_results = semantic_retrieve(question, chunks, model, chunk_vectors, top_k=len(chunks))
    keyword_results = retrieve(question, chunks, top_k=len(chunks))
    return reciprocal_rank_fusion(semantic_results, keyword_results, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to search for")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("Your question: ")
    if not question.strip():
        parser.error("Please enter a non-empty question.")

    chunks = load_chunks()
    model = load_embedding_model()
    chunk_vectors = embed_chunks(model, chunks)
    results = hybrid_retrieve(question, chunks, model, chunk_vectors)

    print("\nHYBRID RESULTS")
    for position, result in enumerate(results, 1):
        semantic_rank = result["semantic_rank"] or "-"
        keyword_rank = result["keyword_rank"] or "-"
        print(f"\n{position}. {result['title']} / {result['section']}")
        print(f"   Semantic rank: {semantic_rank}; keyword rank: {keyword_rank}")
        print(f"   RRF score: {result['score']:.5f} (not a confidence score)")
        print(f"   Source: {result['source']}")
        print(f"   Passage: {result['text']}")


if __name__ == "__main__":
    main()
