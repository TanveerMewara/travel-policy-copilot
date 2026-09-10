"""Lesson 6: search stored passage vectors without re-embedding documents."""

import argparse

from hybrid_search import hybrid_retrieve
from index_documents import load_index
from model_runtime import load_embedding_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to search for")
    args = parser.parse_args()
    question = args.question if args.question is not None else input("Your question: ")
    if not question.strip():
        parser.error("Please enter a non-empty question.")

    chunks, vectors, manifest = load_index()
    model = load_embedding_model()
    results = hybrid_retrieve(question, chunks, model, vectors, top_k=3)

    print(f"Loaded {len(chunks)} stored vectors built at {manifest['created_at_utc']}.")
    print("Only the question was embedded during this search.")
    for position, result in enumerate(results, 1):
        metadata = result["metadata"]
        print(f"\n{position}. {result['title']} / {result['section']}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Supplier: {metadata.get('supplier', '-')}")
        print(f"   Rate: {metadata.get('rate', '-')}")
        print(
            f"   Effective: {metadata.get('effective_from', '-')} to "
            f"{metadata.get('effective_to', '-')}"
        )
        print(
            f"   Semantic rank: {result['semantic_rank']}; "
            f"keyword rank: {result['keyword_rank'] or '-'}"
        )
        print(f"   Passage: {result['text']}")


if __name__ == "__main__":
    main()
