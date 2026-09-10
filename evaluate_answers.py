"""Evaluate retrieved source, generated facts, citations, and lexical grounding."""

import json
import re
from pathlib import Path

from hybrid_search import hybrid_retrieve
from index_documents import load_index
from model_runtime import load_embedding_model, load_generator_model, load_reranker_model
from rag import build_prompt, finalize_answer, generate_answer
from reranker import rerank
from search import STOP_WORDS

CASES_FILE = Path(__file__).resolve().parent / "data" / "evaluation_answers.json"


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def grounded_token_precision(answer, evidence):
    answer_tokens = set(normalize(answer).split()) - STOP_WORDS
    evidence_tokens = set(normalize(evidence).split()) - STOP_WORDS
    return len(answer_tokens & evidence_tokens) / len(answer_tokens) if answer_tokens else 0.0


def main():
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    chunks, vectors, _manifest = load_index()
    embedding_model = load_embedding_model()
    reranker_model = load_reranker_model()
    tokenizer, generator = load_generator_model()

    source_hits = fact_hits = citation_hits = 0
    grounding_total = 0.0
    print("CASE                              SOURCE  FACTS  CITATION  GROUNDING")
    print("-" * 76)
    for case in cases:
        candidates = hybrid_retrieve(
            case["question"], chunks, embedding_model, vectors, top_k=min(10, len(chunks))
        )
        result = rerank(case["question"], candidates, reranker_model, top_k=1)[0]
        generated = generate_answer(build_prompt(case["question"], [result]), tokenizer, generator)
        answer = finalize_answer(generated, case["question"], result["text"])
        cited_answer = f"{answer.rstrip()} [1]" if answer else ""

        source_ok = result["source"] == case["expected_source"]
        normalized_answer = normalize(answer)
        facts_ok = all(normalize(term) in normalized_answer for term in case["expected_terms"])
        citation_ok = cited_answer.endswith("[1]")
        grounding = grounded_token_precision(answer, result["text"])
        source_hits += source_ok
        fact_hits += facts_ok
        citation_hits += citation_ok
        grounding_total += grounding
        print(
            f"{case['id']:<34} {source_ok!s:^6}  {facts_ok!s:^5}  "
            f"{citation_ok!s:^8}  {grounding:>8.1%}"
        )

    count = len(cases)
    print("\nANSWER METRICS")
    print(f"Source accuracy:        {source_hits / count:.1%}")
    print(f"Required-fact accuracy: {fact_hits / count:.1%}")
    print(f"Citation coverage:      {citation_hits / count:.1%}")
    print(f"Lexical grounding:      {grounding_total / count:.1%}")
    print("Lexical grounding is a diagnostic overlap measure, not a full faithfulness judge.")


if __name__ == "__main__":
    main()
