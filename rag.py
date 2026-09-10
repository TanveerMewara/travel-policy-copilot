"""Lesson 3: retrieve policy evidence and generate a grounded answer locally."""

import argparse
import re
import sys

from hybrid_search import hybrid_retrieve
from index_documents import load_index
from model_runtime import GENERATOR_MODEL_NAME, load_embedding_model, load_generator_model
from query_validation import missing_policy_year
from reranker import RERANKER_NAME, load_reranker_model, rerank

GENERATOR_NAME = GENERATOR_MODEL_NAME
# Calibrated by evaluate_retrieval.py on the tiny lesson dataset.
DEFAULT_MIN_RERANKER_SCORE = 1.696
# Backward-compatible name used by the UI from earlier lessons.
DEFAULT_MIN_SIMILARITY = DEFAULT_MIN_RERANKER_SCORE


def build_prompt(question, retrieved_chunks):
    """Place instructions, retrieved evidence, and the question in one prompt."""
    evidence_blocks = []
    for number, chunk in enumerate(retrieved_chunks, 1):
        evidence_blocks.append(
            f"SOURCE [{number}]\n"
            f"Policy: {chunk['title']}\n"
            f"Section: {chunk['section']}\n"
            f"Text: {chunk['text']}"
        )

    evidence = "\n\n".join(evidence_blocks)
    return f"""Answer the travel-policy question using only the context.
If the context does not contain the answer, reply: I don't have enough information in the provided policies.
Compare all times and numbers carefully before choosing the applicable rule.
Include the exact fee, refund, or time stated by that rule. Give one short sentence.

CONTEXT:
{evidence}

QUESTION: {question}
ANSWER:"""


def generate_answer(prompt, tokenizer, generator):
    """Ask the local language model to continue the prompt with an answer."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = generator.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        num_beams=2,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def finalize_answer(generated_answer, question, evidence):
    """Replace fragile short/numeric answers with the retrieved policy text."""
    answer = generated_answer.strip()
    meaningful_tokens = re.findall(r"[a-z0-9]+", answer.lower())
    evidence_has_conditions = any(
        phrase in evidence.lower() for phrase in ("at least", "within", "before", "after")
    )
    question_has_number = bool(re.search(r"\d", question))
    answer_has_number = bool(re.search(r"\d", answer))
    needs_fallback = len(meaningful_tokens) < 4 or (
        evidence_has_conditions and question_has_number and not answer_has_number
    )
    if not answer or needs_fallback:
        return f"According to the retrieved policy: {evidence.strip()}"
    return answer


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question about the policies")
    parser.add_argument(
        "--show-prompt", action="store_true", help="Print exactly what is sent to the generator"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="Abstain when the reranker score is below this value",
    )
    args = parser.parse_args()
    question = args.question if args.question is not None else input("Your question: ")
    if not question.strip():
        parser.error("Please enter a non-empty question.")

    try:
        import sentence_transformers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "RAG dependencies are missing or could not be imported.\n"
            "Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
            f"Import detail: {error}"
        ) from error

    chunks, chunk_vectors, _manifest = load_index()
    print(
        f"1. RETRIEVE: loaded {len(chunks)} stored passage vectors; encoding the question...",
        flush=True,
    )
    embedding_model = load_embedding_model()
    candidates = hybrid_retrieve(
        question, chunks, embedding_model, chunk_vectors, top_k=min(10, len(chunks))
    )
    for number, chunk in enumerate(candidates[:3], 1):
        print(
            f"   [{number}] {chunk['title']} / {chunk['section']} "
            f"(semantic rank {chunk['semantic_rank']}, "
            f"keyword rank {chunk['keyword_rank'] or '-'}, "
            f"similarity {chunk['semantic_score']:.3f})"
        )

    available_years = missing_policy_year(question, candidates)
    if available_years:
        print("\nCLARIFICATION NEEDED")
        print("Which policy year applies? Available years: " + ", ".join(available_years) + ".")
        print("The reranker and generator were not called.")
        return

    print(f"2. RERANK: loading {RERANKER_NAME}...", flush=True)
    reranker_model = load_reranker_model()
    retrieved = rerank(question, candidates, reranker_model, top_k=3)
    print(
        f"   Selected: {retrieved[0]['title']} / {retrieved[0]['section']} "
        f"(cross-encoder score {retrieved[0]['reranker_score']:.3f})"
    )

    if retrieved[0]["reranker_score"] < args.min_score:
        print(
            f"\nABSTAIN: reranker score {retrieved[0]['reranker_score']:.3f} "
            f"is below the development threshold {args.min_score:.3f}."
        )
        print("ANSWER")
        print("I don't have enough information in the provided policies.")
        print("\nThe generator was not called. Inspect the candidates above before")
        print("assuming the information is truly absent.")
        return

    # Lesson 3 uses only the best passage for generation. The other candidates
    # stay visible above so we can inspect retrieval without confusing this tiny
    # generator with competing policy rules.
    generation_context = retrieved[:1]
    prompt = build_prompt(question, generation_context)
    if args.show_prompt:
        print("\n3. AUGMENT: prompt sent to the language model\n")
        print(prompt)
    else:
        print("3. AUGMENT: added the highest-ranked passage to the model prompt.")

    print(f"4. GENERATE: loading {GENERATOR_NAME} on CPU...", flush=True)
    tokenizer, generator = load_generator_model()
    answer = generate_answer(prompt, tokenizer, generator)
    answer = finalize_answer(answer, question, generation_context[0]["text"])
    if answer and not answer.rstrip().endswith("[1]"):
        answer = f"{answer.rstrip()} [1]"

    print("\nANSWER")
    print(answer or "The model returned an empty answer.")

    print("\nSOURCES SUPPLIED TO THE MODEL")
    for number, chunk in enumerate(generation_context, 1):
        print(f"[{number}] {chunk['source']} — {chunk['title']} / {chunk['section']}")
        print(f"    {chunk['text']}")
    print("\nThis small learning model can still misunderstand evidence. Retrieval scores")
    print("and a source list do not prove factual correctness; evaluation comes next.")


if __name__ == "__main__":
    main()
