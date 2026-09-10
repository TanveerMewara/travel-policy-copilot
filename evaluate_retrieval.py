"""Evaluate retrieval on separate development and held-out test questions."""

import json
from itertools import pairwise
from pathlib import Path

from hybrid_search import hybrid_retrieve
from model_runtime import load_embedding_model
from reranker import load_reranker_model, rerank
from search import load_chunks
from semantic_search import embed_chunks, semantic_retrieve

DATA_DIR = Path(__file__).resolve().parent / "data"
DEV_FILE = DATA_DIR / "evaluation_dev.json"
TEST_FILE = DATA_DIR / "evaluation_test.json"


def is_expected(result, case):
    return (
        result["source"] == case["expected_source"]
        and result["section"] == case["expected_section"]
    )


def best_threshold(scored_cases, score_key):
    """Choose a cutoff using development data only."""
    scores = sorted({case[score_key] for case in scored_cases})
    candidates = [scores[0] - 1e-6, scores[-1] + 1e-6]
    candidates += [(left + right) / 2 for left, right in pairwise(scores)]
    best = None
    for threshold in candidates:
        correct = sum((case[score_key] >= threshold) == case["answerable"] for case in scored_cases)
        result = (correct / len(scored_cases), threshold)
        if best is None or result[0] > best[0]:
            best = result
    return best[1], best[0]


def evaluate_split(label, cases, chunks, vectors, embedding_model, reranker_model):
    answerable_count = sum(case["expected_source"] is not None for case in cases)
    metrics = {
        "semantic": {"hit1": 0, "hit3": 0, "rr": 0.0},
        "hybrid": {"hit1": 0, "hit3": 0, "rr": 0.0},
        "reranked": {"hit1": 0, "hit3": 0, "rr": 0.0},
    }
    scored_cases = []

    print(f"\n{label.upper()} SET")
    print("CASE                       SEM   HYBRID  RERANK   SEM    CROSS")
    print("-" * 72)
    for case in cases:
        semantic_results = semantic_retrieve(
            case["question"], chunks, embedding_model, vectors, top_k=3
        )
        candidates = hybrid_retrieve(
            case["question"], chunks, embedding_model, vectors, top_k=min(10, len(chunks))
        )
        hybrid_results = candidates[:3]
        reranked_results = rerank(case["question"], candidates, reranker_model, top_k=3)
        answerable = case["expected_source"] is not None
        ranks = {}
        for name, results in (
            ("semantic", semantic_results),
            ("hybrid", hybrid_results),
            ("reranked", reranked_results),
        ):
            rank = next(
                (
                    index
                    for index, result in enumerate(results, 1)
                    if answerable and is_expected(result, case)
                ),
                None,
            )
            ranks[name] = rank
            if answerable:
                metrics[name]["hit1"] += rank == 1
                metrics[name]["hit3"] += rank is not None
                metrics[name]["rr"] += 1 / rank if rank else 0

        semantic_score = max(result["semantic_score"] for result in candidates)
        reranker_score = reranked_results[0]["reranker_score"]
        scored_cases.append(
            {
                "answerable": answerable,
                "semantic_score": semantic_score,
                "reranker_score": reranker_score,
            }
        )
        display_ranks = [
            str(ranks[name]) if answerable else "N/A" for name in ("semantic", "hybrid", "reranked")
        ]
        print(
            f"{case['id']:<26} {display_ranks[0]:^5} {display_ranks[1]:^8} "
            f"{display_ranks[2]:^7} {semantic_score:>6.3f} {reranker_score:>8.3f}"
        )

    print("\nMETHOD      HIT@1       HIT@3       MRR")
    for name in ("semantic", "hybrid", "reranked"):
        result = metrics[name]
        print(
            f"{name:<10}  {result['hit1']}/{answerable_count} "
            f"({result['hit1'] / answerable_count:.1%})  "
            f"{result['hit3']}/{answerable_count} "
            f"({result['hit3'] / answerable_count:.1%})  "
            f"{result['rr'] / answerable_count:.3f}"
        )
    return scored_cases


def classification_accuracy(scored_cases, threshold, score_key):
    correct = sum((case[score_key] >= threshold) == case["answerable"] for case in scored_cases)
    return correct / len(scored_cases)


def main():
    dev_cases = json.loads(DEV_FILE.read_text(encoding="utf-8"))
    test_cases = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    chunks = load_chunks()
    embedding_model = load_embedding_model()
    reranker_model = load_reranker_model()
    vectors = embed_chunks(embedding_model, chunks)

    dev_scores = evaluate_split(
        "development", dev_cases, chunks, vectors, embedding_model, reranker_model
    )
    test_scores = evaluate_split(
        "held-out test", test_cases, chunks, vectors, embedding_model, reranker_model
    )

    print("\nANSWERABILITY")
    print("SIGNAL       THRESHOLD   DEV ACCURACY   TEST ACCURACY")
    for score_key, label in (("semantic_score", "semantic"), ("reranker_score", "cross-encoder")):
        threshold, dev_accuracy = best_threshold(dev_scores, score_key)
        test_accuracy = classification_accuracy(test_scores, threshold, score_key)
        print(f"{label:<12} {threshold:>9.3f}   {dev_accuracy:>11.1%}   {test_accuracy:>12.1%}")
    print("Only the held-out result estimates behavior on questions that did not")
    print("participate in choosing the threshold.")


if __name__ == "__main__":
    main()
