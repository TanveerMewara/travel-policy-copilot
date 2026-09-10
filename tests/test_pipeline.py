import json

import pytest

import telemetry
from document_ingestion import ingest_pdf_bytes, validate_document_id
from evaluate_answers import grounded_token_precision, normalize
from evaluate_retrieval import best_threshold, classification_accuracy
from hybrid_search import reciprocal_rank_fusion
from query_validation import missing_policy_year
from rag import build_prompt, finalize_answer
from search import chunk_page_text, load_chunks


def fake_chunk(chunk_id, score):
    return {
        "chunk_id": chunk_id,
        "source": f"data/policies/{chunk_id}.md",
        "section": "Cancellation",
        "title": chunk_id,
        "text": "Policy text",
        "metadata": {"supplier": "Hotel", "rate": "Standard Rate"},
        "score": score,
    }


def test_policy_loading_preserves_versions_and_metadata():
    chunks = load_chunks()
    assert len(chunks) == 14
    ids = {chunk["chunk_id"] for chunk in chunks}
    assert "harbor-standard-2025::cancellation" in ids
    assert "harbor-standard-2026::cancellation" in ids
    assert all(chunk["metadata"].get("supplier") for chunk in chunks)


def test_page_chunking_has_expected_overlap():
    text = " ".join(f"word{number}" for number in range(400))
    chunks = chunk_page_text(text, max_words=180, overlap_words=30)
    assert [len(chunk.split()) for chunk in chunks] == [180, 180, 100]
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]
    with pytest.raises(ValueError):
        chunk_page_text(text, max_words=30, overlap_words=30)


def test_rrf_rewards_results_found_by_both_methods():
    semantic = [fake_chunk("semantic-only", 0.9), fake_chunk("both", 0.8)]
    keyword = [fake_chunk("both", 4), fake_chunk("keyword-only", 3)]
    results = reciprocal_rank_fusion(semantic, keyword, top_k=3)
    assert results[0]["chunk_id"] == "both"
    assert results[0]["semantic_rank"] == 2
    assert results[0]["keyword_rank"] == 1


def test_conflicting_versions_request_a_year():
    candidates = [
        {
            "section": "Cancellation",
            "metadata": {
                "supplier": "Harbor Hotel",
                "rate": "Standard Rate",
                "effective_from": "2026-01-01",
            },
        },
        {
            "section": "Cancellation",
            "metadata": {
                "supplier": "Harbor Hotel",
                "rate": "Standard Rate",
                "effective_from": "2025-01-01",
            },
        },
    ]
    assert missing_policy_year("What is the cancellation rule?", candidates) == ["2025", "2026"]
    assert missing_policy_year("What is the 2026 cancellation rule?", candidates) == []


def test_pdf_validation_rejects_unsafe_or_invalid_input():
    with pytest.raises(ValueError):
        validate_document_id("../escape")
    with pytest.raises(ValueError, match="valid PDF header"):
        ingest_pdf_bytes(b"not a pdf", {"document_id": "invalid-pdf"})


def test_prompt_contains_only_supplied_evidence():
    chunk = fake_chunk("harbor", 1.0)
    chunk["text"] = "Cancellation costs one night."
    prompt = build_prompt("What is the fee?", [chunk])
    assert "Cancellation costs one night." in prompt
    assert "What is the fee?" in prompt
    assert "using only the context" in prompt


def test_short_numeric_answer_falls_back_to_policy_evidence():
    evidence = (
        "Cancellations at least 12 hours before check-in receive a full refund. "
        "Cancellations within 12 hours incur a fee equal to 25 percent of the first night."
    )
    answer = finalize_answer("a full refund", "What happens 8 hours before check-in?", evidence)
    assert "25 percent of the first night" in answer


def test_threshold_is_fit_without_test_data():
    dev = [
        {"answerable": True, "reranker_score": 4.0},
        {"answerable": True, "reranker_score": 3.0},
        {"answerable": False, "reranker_score": -1.0},
    ]
    threshold, accuracy = best_threshold(dev, "reranker_score")
    assert accuracy == 1.0
    assert -1.0 < threshold < 3.0
    assert classification_accuracy(dev, threshold, "reranker_score") == 1.0


def test_answer_diagnostics_are_normalized_and_grounded():
    assert normalize("Non-refundable!") == "non refundable"
    score = grounded_token_precision(
        "The charge is equal to the first night.",
        "Cancellations incur a charge equal to the first night.",
    )
    assert score >= 0.8


def test_telemetry_hashes_query_instead_of_storing_it(tmp_path, monkeypatch):
    log_file = tmp_path / "events.jsonl"
    monkeypatch.setattr(telemetry, "LOG_FILE", log_file)
    question = "private example question"
    assert telemetry.record_query(question, {"abstained": True, "results": []})
    raw = log_file.read_text(encoding="utf-8")
    event = json.loads(raw)
    assert question not in raw
    assert len(event["query_sha256"]) == 64
